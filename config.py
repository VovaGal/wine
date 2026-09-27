from contextvars import ContextVar
from functools import lru_cache
 
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
 
 
class AppSettings(BaseSettings):
 
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        if not _settings_sources_enabled.get():
            return (init_settings,)
 
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
 
    @classmethod
    def defaults(cls):
        token = _settings_sources_enabled.set(False)
 
        try:
            return cls()
        finally:
            _settings_sources_enabled.reset(token)
 
 
_settings_sources_enabled = ContextVar(
    "settings_sources_enabled",
    default=True,
)
 
 
class InferenceConfig(AppSettings):
    catalog_path: str = Field(default="dataset/wine_catalog.jsonl")
    checkpoint_path: str = Field(default="weights/parseq_wine_best.pt")
    device: str = Field(default="cuda:0")
    craft_canvas_size: int = Field(default=2560)
    recognition_passes: int = Field(default=3)
    allow_multilingual: bool = Field(default=True)
    top_k: int = Field(default=5)
 
 
class RabbitMQConfig(AppSettings):
    host: str = Field(default="localhost")
    port: int = Field(default=5672)
    username: str = Field(default="guest")
    password: str = Field(default="guest")
 
    publish_queue: str = Field(default="ocr.recognition.tasks")  # worker CONSUMES tasks from this queue
    consume_queue: str = Field(default="ocr.recognition.results")  # worker PUBLISHES results to this queue
 
    prefetch_count: int = Field(default=1)
 
    @property
    def url(self) -> str:
        return f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"
 
 
class MinioConfig(AppSettings):
    endpoint: str = Field(default="localhost:9000")
    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")
    secure: bool = Field(default=False)
    image_bucket: str = Field(default="images")
    # image_prefix removed: object_key already contains the full prefix, so it's not needed for downloading from minio
 
 
class LoggingConfig(AppSettings):
    level: str = Field(default="INFO")
    logs_directory: str = Field(default="./logs/")
    file: str | None = Field(default=None)
    serialize: bool = Field(default=False)
 
 
class Config(AppSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        extra="ignore",
    )
 
    worker_name: str = Field(default="ocr_inference_worker")
    worker_version: str = Field(default="0.0.1")
 
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    rabbitmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
 
 
@lru_cache
def config() -> Config:
    return Config()
