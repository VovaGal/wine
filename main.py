import asyncio
import json
import signal
import sys
from pathlib import Path

import aio_pika
from aio_pika import DeliveryMode, Message
from config import Config, config
from loguru import logger
from minio import Minio
from pydantic import BaseModel, ValidationError
from wine_inference import WineInference


class TaskMessage(BaseModel):
    """Contract for incoming task messages: {"task_id": "...", "object_key": "..."}"""

    task_id: str
    object_key: str


def download_object(minio_client: Minio, bucket: str, object_key: str) -> bytes:
    response = None
    try:
        response = minio_client.get_object(bucket, object_key)
        return response.read()
    finally:
        if response is not None:
            response.close()
            response.release_conn()


class WineRecognitionWorker:
    def __init__(self, settings: Config):
        self.settings = settings

        self.worker: WineInference | None = None
        self.minio_client: Minio | None = None
        self.connection: aio_pika.abc.AbstractRobustConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None
        self.task_queue: aio_pika.abc.AbstractQueue | None = None

        self._stop_event = asyncio.Event()

    def logger_setup(self) -> None:
        logger.remove()

        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            level=self.settings.logging.level,
            format=log_format,
            serialize=self.settings.logging.serialize,
            backtrace=True,
            diagnose=False,
        )

        if self.settings.logging.file:
            log_path = Path(self.settings.logging.logs_directory) / self.settings.logging.file
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_path,
                level=self.settings.logging.level,
                format=log_format,
                serialize=self.settings.logging.serialize,
                backtrace=True,
                diagnose=False,
                encoding="utf-8",
            )

        logger.info("Logger setup complete with level: {}", self.settings.logging.level)

    async def prelude(self) -> None:
        logger.info("Starting worker prelude routine")

        logger.debug("Loading wine inference pipeline")
        self.worker = WineInference(
            catalog_path=self.settings.inference.catalog_path,
            checkpoint_path=self.settings.inference.checkpoint_path,
            device=self.settings.inference.device,
            craft_canvas_size=self.settings.inference.craft_canvas_size,
            recognition_passes=self.settings.inference.recognition_passes,
            allow_multilingual=self.settings.inference.allow_multilingual,
        )
        logger.info(
            "Inference initialized: catalog={}, checkpoint={}, device={}",
            self.settings.inference.catalog_path,
            self.settings.inference.checkpoint_path,
            self.settings.inference.device,
        )

        logger.debug("Creating Minio client")
        self.minio_client = Minio(
            self.settings.minio.endpoint,
            access_key=self.settings.minio.access_key,
            secret_key=self.settings.minio.secret_key,
            secure=self.settings.minio.secure,
        )

        logger.debug("Connecting to RabbitMQ")
        try:
            self.connection = await aio_pika.connect_robust(self.settings.rabbitmq.url)
        except Exception:
            logger.exception("Failed to connect to RabbitMQ")
            raise
        logger.debug("RabbitMQ connection established")

        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=self.settings.rabbitmq.prefetch_count)

        self.task_queue = await self.channel.declare_queue(
            self.settings.rabbitmq.publish_queue,
            durable=True,
        )
        await self.channel.declare_queue(
            self.settings.rabbitmq.consume_queue,
            durable=True,
        )

        logger.info("Worker prelude routine completed successfully")

    async def publish_result(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        await self.channel.default_exchange.publish(
            Message(
                body=body,
                content_type="application/json",
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=self.settings.rabbitmq.consume_queue,
        )

    @staticmethod
    def _error_response(task_id: str | None, error: str) -> dict:
        return {
            "task_id": task_id,
            "result": {
                "status": "error",
                "ocr_lines": [],
                "best_match": None,
                "alternatives": [],
                "candidates": [],
                "score_type": "uncalibrated_catalog_match_score",
                "error": error,
            },
        }

    async def handle_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process(requeue=False):
            try:
                task = TaskMessage.model_validate_json(message.body)
            except (ValidationError, ValueError) as exc:
                # Covers both malformed JSON and contract violations (missing/extra/wrong-typed fields).
                logger.error("Rejected malformed task message: {}", exc)
                return

            logger.info("Received request {}: object_key={}", task.task_id, task.object_key)

            try:
                image_bytes = await asyncio.to_thread(
                    download_object,
                    self.minio_client,
                    self.settings.minio.image_bucket,
                    task.object_key,
                )

                result = await asyncio.to_thread(
                    self.worker.predict,
                    image_bytes,
                    allow_multilingual=self.settings.inference.allow_multilingual,
                    top_k=self.settings.inference.top_k,
                )

                result = self._enrich_result(result)

                response = {
                    "task_id": task.task_id,
                    "result": result,
                }
                logger.info("Request {} finished: status={}", task.task_id, result.get("status"))

            except Exception:
                logger.exception("Recognition failed for request {}", task.task_id)
                response = self._error_response(task.task_id, "recognition_failed")

            await self.publish_result(response)
            logger.info(
                "Result for request {} published to '{}'",
                task.task_id,
                self.settings.rabbitmq.consume_queue,
            )

    async def consume(self) -> None:
        async with self.task_queue.iterator() as queue_iter:
            async for message in queue_iter:
                if self._stop_event.is_set():
                    break
                await self.handle_message(message)

    async def shutdown(self) -> None:
        logger.info("Shutting down worker")
        self._stop_event.set()
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
        logger.info("Worker shutdown complete")

    async def run(self) -> None:
        self.logger_setup()
        await self.prelude()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # add_signal_handler is unavailable on some platforms (e.g. Windows)
                pass

        logger.info(
            "Waiting for tasks in '{}'...",
            self.settings.rabbitmq.publish_queue,
        )

        try:
            await self.consume()
        finally:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()

    def _enrich_candidate(self, candidate: dict) -> dict:
        wine_id = str(candidate["wine_id"])
        product = self.worker.products.get(wine_id)

        if product is None:
            logger.warning(
                "Wine {} from inference result is missing in catalog",
                wine_id,
            )
            return candidate

        metadata = {
            key: value
            for key, value in candidate.items()
            if key not in product
        }

        return {
            **product,
            **metadata,
        }


    def _enrich_result(self, result: dict) -> dict:
        enriched = dict(result)

        if result.get("best_match") is not None:
            enriched["best_match"] = self._enrich_candidate(
                result["best_match"]
            )

        enriched["alternatives"] = [
            self._enrich_candidate(candidate)
            for candidate in result.get("alternatives", [])
        ]

        enriched["candidates"] = [
            self._enrich_candidate(candidate)
            for candidate in result.get("candidates", [])
        ]

        return enriched


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Wine Recognition Worker")
    parser.add_argument("--generate-config-json-string", help="Generate a JSON string of the config and exit.", action="store_true")

    args = parser.parse_args()

    if args.generate_config_json_string:
        print(Config.defaults().model_dump_json())
        return

    settings = config()
    worker = WineRecognitionWorker(settings)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()