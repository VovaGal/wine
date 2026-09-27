"""Match OCR text to the five closest wines in a built catalog.

Dependencies: Python standard library + wine_normalizer.py. Load this object
once per backend worker, as you already do with CRAFT and PARSeq.

    matcher = WineMatcher("dataset/wine_catalog.jsonl")
    result = matcher.match(["Mont Blanc", "Cuvée"], top_k=5)

Accepts OCR strings or ``[{"text": "Mont Blanc", "confidence": 0.92}, ...]``.
``match_score`` ranks catalog entries; it is not a calibrated probability.
Unresolved results contain no product candidates: a weak shared-word overlap
must not be presented as an actual wine recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from wine_normalizer import display_text, search_forms, search_key, tokens


ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
FIELD_WEIGHTS = {
    "name": 0.65,
    "producer": 0.18,
    "grapes": 0.08,
    "region": 0.04,
    "style": 0.05,
}

# Only unambiguous terms printed on common bilingual labels. These are search
# equivalents, not changes to the original OCR or catalog wording.
EQUIVALENT_WORDS = {
    "sweet": ("сладкое", "сладкий"),
    "brut": ("брют",),
    "dry": ("сухое", "сухой"),
    "rose": ("розе",),
    "розе": ("rose",),
    "брют": ("brut",),
}


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None
    keys: tuple[str, ...]
    indices: tuple[int, ...]


def parse_ocr_lines(lines: Iterable[str | dict[str, Any]]) -> list[OCRLine]:
    """Accept raw runner output or structured OCR; reject invalid confidences."""
    if isinstance(lines, (str, bytes, dict)):
        raise TypeError("ocr_lines must be a list of strings or {text, confidence} objects")
    parsed: list[OCRLine] = []
    for item in lines:
        if isinstance(item, str):
            value, confidence = item, None
        elif isinstance(item, dict):
            value, confidence = item.get("text"), item.get("confidence")
            if not isinstance(value, str):
                raise ValueError("Every OCR entry must include a text string")
            if confidence is not None:
                if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
                    raise ValueError("OCR confidence must be a number in [0, 1]")
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("OCR confidence must be a finite number in [0, 1]")
        else:
            raise TypeError("OCR entries must be strings or {text, confidence} objects")
        text = display_text(value)
        keys = search_forms(text)
        if keys:
            parsed.append(OCRLine(text, confidence, keys, (len(parsed),)))
    return parsed


def query_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """Also compare neighboring OCR lines to product names split by CRAFT."""
    result = list(lines)

    def add_group(group: list[OCRLine]) -> None:
        text = " ".join(item.text for item in group)
        # Very long composites are almost always OCR bleed across label
        # regions and cost disproportionately much to compare.
        if len(text) > 42:
            return
        known = [item.confidence for item in group if item.confidence is not None]
        confidence = sum(known) / len(known) if len(known) == len(group) else None
        result.append(OCRLine(text, confidence, search_forms(text), tuple(
            index for item in group for index in item.indices
        )))

    for width in (2, 3):
        for start in range(len(lines) - width + 1):
            add_group(lines[start:start + width])
    # CRAFT can place a stray decorative line between two parts of a name.
    # Skip at most one intermediate line, keeping the search bounded.
    for start in range(len(lines) - 2):
        if min(len(lines[start].text), len(lines[start + 2].text)) < 5:
            continue
        add_group([lines[start], lines[start + 2]])
        if start + 3 < len(lines):
            add_group([lines[start], lines[start + 2], lines[start + 3]])
    return result


def grams(text: str) -> set[str]:
    """Character trigrams from each token, useful for one-letter OCR errors."""
    result: set[str] = set()
    for word in text.split():
        if len(word) >= 4 and ALPHA.search(word):
            result.update(word[index:index + 3] for index in range(len(word) - 2))
    return result


@lru_cache(maxsize=200_000)
def similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) > 1.6 * len(right) and len(left.split()) > 1:
        return 0.0
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    ratio = matcher.ratio() if matcher.quick_ratio() >= 0.62 else 0.0
    left_words, right_words = left.split(), right.split()
    # OCR often sees only part of a printed product name. Permit a short OCR
    # line to match a longer catalog title, but NEVER grant the same partial
    # bonus to a longer OCR phrase that merely contains a generic short name.
    if len(left_words) <= len(right_words):
        for size in {len(left_words), min(len(right_words), len(left_words) + 1)}:
            for start in range(len(right_words) - size + 1):
                window = " ".join(right_words[start:start + size])
                if 2 * min(len(left), len(window)) / (len(left) + len(window)) < 0.62:
                    continue
                part = SequenceMatcher(None, left, window, autojunk=False)
                if part.quick_ratio() >= 0.62:
                    ratio = max(ratio, part.ratio() * (
                        0.92 if len(right_words) > size else 1.0
                    ))
    return ratio


class WineMatcher:
    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path)
        self.products: dict[str, dict[str, Any]] = {}
        with self.catalog_path.open("r", encoding="utf-8-sig") as handle:
            for row_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid catalog JSON on line {row_number}") from exc
                if product.get("schema_version") != 1:
                    raise ValueError(f"Unsupported catalog schema on line {row_number}")
                wine_id = str(product.get("wine_id", ""))
                if not wine_id or wine_id in self.products:
                    raise ValueError(f"Missing or duplicate wine_id on line {row_number}")
                self.products[wine_id] = product
        if not self.products:
            raise ValueError(f"Empty wine catalog: {self.catalog_path}")
        self.producer_ids = {}
        for wine_id, product in self.products.items():
            key = search_key(product.get("producer"))
            if key:
                self.producer_ids.setdefault(key, []).append(wine_id)

        self.fields: dict[str, dict[str, tuple[str, ...]]] = {}
        self.alias_owners: dict[str, set[str]] = defaultdict(set)
        self.token_index: dict[str, set[str]] = defaultdict(set)
        self.gram_index: dict[str, set[str]] = defaultdict(set)
        for wine_id, product in self.products.items():
            aliases = [item["text"] for item in product.get("aliases", [])]
            fields = {
                "name": self._unique_forms(aliases or [product.get("name")]),
                "producer": self._unique_forms([product.get("producer")]),
                "grapes": self._unique_forms(product.get("grapes") or []),
                "region": self._unique_forms([product.get("region")]),
                "style": self._unique_forms(product.get("style") or []),
            }
            self.fields[wine_id] = fields
            for alias in fields["name"]:
                self.alias_owners[alias].add(wine_id)
            # Only identifying fields retrieve candidates; style/region often
            # appear on hundreds of unrelated wines.
            for field in ("name", "producer", "grapes"):
                for value in fields[field]:
                    for word in tokens(value):
                        if len(word) >= 3 and ALPHA.search(word):
                            self.token_index[word].add(wine_id)
                    for gram in grams(value):
                        self.gram_index[gram].add(wine_id)

    @staticmethod
    def _unique_forms(values: Iterable[Any]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(form for value in values for form in search_forms(value)))

    def specificity(self, key: str) -> float:
        """Common tokens (e.g. wine/riserva) should not identify a product."""
        corpus_size = len(self.products)
        word_scores = []
        for word in key.split():
            if len(word) < 3 or not ALPHA.search(word):
                continue
            appearances = len(self.token_index.get(word, ()))
            if not appearances:
                # An OCR typo absent from the catalog is not proof of rarity.
                continue
            value = math.log1p(corpus_size / (1 + appearances)) / math.log1p(corpus_size)
            word_scores.append(value)
        return max(0.35, max(word_scores, default=0.35))

    def retrieve(self, queries: list[OCRLine], limit: int = 80) -> list[str]:
        votes: Counter[str] = Counter()
        corpus_size = len(self.products)
        for query in queries:
            # Numeric-only lines are very common. They can help rerank only
            # when the catalog actually knows a vintage.
            if not any(ALPHA.search(key) for key in query.keys):
                continue
            per_line: Counter[str] = Counter()
            for key in query.keys:
                for word in set(key.split()):
                    if len(word) < 3 or not ALPHA.search(word):
                        continue
                    matches = self.token_index.get(word, ())
                    if matches:
                        weight = math.log1p(corpus_size / (1 + len(matches)))
                        per_line.update({wine_id: weight for wine_id in matches})
                for gram in grams(key):
                    matches = self.gram_index.get(gram, ())
                    if matches:
                        weight = 0.17 * math.log1p(corpus_size / (1 + len(matches)))
                        per_line.update({wine_id: weight for wine_id in matches})
            for wine_id, vote in sorted(
                per_line.items(), key=lambda pair: (-pair[1], pair[0])
            )[:35]:
                votes[wine_id] += vote
        return [wine_id for wine_id, _ in sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]

    def _best_evidence(
        self, queries: list[OCRLine], values: tuple[str, ...], field: str
    ) -> dict[str, Any] | None:
        best: tuple[float, OCRLine, str, float] | None = None
        for query in queries:
            for query_key in query.keys:
                for candidate in values:
                    likeness = similarity(query_key, candidate)
                    if likeness < 0.62:
                        continue
                    confidence_factor = (
                        1.0 if query.confidence is None
                        else 0.65 + 0.35 * query.confidence
                    )
                    # Measure rarity on the CATALOG alias: an OCR misspelling
                    # such as FRIZEN must not erase the rarity of FROZEN.
                    specificity = self.specificity(candidate) if field == "name" else 1.0
                    # A common single word appearing inside a long title is
                    # weak evidence even when that title contains rare words.
                    phrase_coverage = (
                        min(1.0, len(query_key.split()) / len(candidate.split()))
                        if field in {"name", "producer"} else 1.0
                    )
                    score = likeness * confidence_factor * specificity * phrase_coverage
                    if best is None or score > best[0]:
                        best = (score, query, candidate, likeness)
        if best is None:
            return None
        score, query, candidate, likeness = best
        return {
            "field": field,
            "ocr_text": query.text,
            "catalog_key": candidate,
            "ocr_indices": list(query.indices),
            "similarity": round(likeness, 3),
            "weighted_similarity": round(score, 3),
            "contribution": round(score * FIELD_WEIGHTS[field], 4),
        }

    def _title_token_evidence(
        self, lines: list[OCRLine], wine_id: str
    ) -> dict[str, Any] | None:
        """Recognize a catalog title printed as several separate OCR lines."""
        observations: dict[str, tuple[int, float]] = {}
        for index, line in enumerate(lines):
            for key in line.keys:
                for word in key.split():
                    if len(word) < 3 and not (len(word) >= 2 and word.isdigit()):
                        continue
                    confidence = line.confidence if line.confidence is not None else 1.0
                    if word not in observations or confidence > observations[word][1]:
                        observations[word] = (index, confidence)
                    for alternate in EQUIVALENT_WORDS.get(word, ()):
                        observations.setdefault(alternate, (index, confidence * 0.92))
        if not observations:
            return None
        best = None
        for alias in self.fields[wine_id]["name"]:
            words = [
                word for word in alias.split()
                if len(word) >= 3 or (len(word) >= 2 and word.isdigit())
            ]
            if not words:
                continue
            total_weight = 0.0
            supported_weight = 0.0
            matched_indices: set[int] = set()
            for word in words:
                weight = max(0.35, self.specificity(word))
                total_weight += weight
                best_word = max(
                    (
                        (similarity(observed, word), index, confidence)
                        for observed, (index, confidence) in observations.items()
                        if 2 * min(len(observed), len(word)) / (len(observed) + len(word)) >= 0.75
                    ),
                    default=(0.0, -1, 0.0),
                )
                likeness, index, confidence = best_word
                if likeness >= 0.78:
                    supported_weight += weight * likeness * (0.65 + 0.35 * confidence)
                    matched_indices.add(index)
            coverage = supported_weight / total_weight
            # The OCR may have read a single common term in an otherwise
            # unrelated title. Require a substantial share of this title.
            if coverage < 0.58 or not matched_indices:
                continue
            evidence = {
                "field": "name",
                "ocr_text": " | ".join(lines[index].text for index in sorted(matched_indices)),
                "catalog_key": alias,
                "ocr_indices": sorted(matched_indices),
                "similarity": round(coverage, 3),
                "weighted_similarity": round(coverage, 3),
                "contribution": round(coverage * FIELD_WEIGHTS["name"], 4),
                "matched_words": len(matched_indices),
            }
            if best is None or evidence["weighted_similarity"] > best["weighted_similarity"]:
                best = evidence
        return best

    def rank(
        self, wine_id: str, queries: list[OCRLine], lines: list[OCRLine],
        rare_words: set[str],
    ) -> dict[str, Any]:
        product = self.products[wine_id]
        evidence = []
        score = 0.0
        used_indices: set[int] = set()
        for field, weight in FIELD_WEIGHTS.items():
            item = self._best_evidence(queries, self.fields[wine_id][field], field)
            if field == "name":
                spread = self._title_token_evidence(lines, wine_id)
                if spread and (not item or spread["weighted_similarity"] > item["weighted_similarity"]):
                    item = spread
            if item:
                # The same OCR line is one piece of evidence. A bottle with
                # "Riesling" in both its name and grape field should not get
                # a second boost from that single printed word.
                indices = set(item["ocr_indices"])
                if indices.issubset(used_indices):
                    if field == "producer" and search_key(product.get("producer")) in self.fields[wine_id]["name"][0]:
                        # A printed brand can occur inside the product name.
                        # Give modest producer credit even when both point at
                        # the same OCR crop.
                        score += 0.10 * item["weighted_similarity"]
                    continue
                evidence.append(item)
                score += weight * item["weighted_similarity"]
                if field == "name" and "matched_words" in item:
                    # Resolve ties between a complete printed name and its
                    # shorter generic sibling (e.g. Cantiani Riesling vs.
                    # Cantiani Aligote Riesling).
                    score += min(0.08, 0.02 * item["matched_words"])
                used_indices.update(indices)

        # Vintage is supporting evidence only if the catalog has an explicit
        # year. In the supplied scrape, every year is null.
        year = product.get("year")
        if isinstance(year, int) and not isinstance(year, bool):
            for query in queries:
                if YEAR.search(query.text) and str(year) in YEAR.findall(query.text):
                    score += 0.03
                    evidence.append({
                        "field": "year", "ocr_text": query.text,
                        "catalog_key": str(year), "ocr_indices": list(query.indices),
                        "similarity": 1.0, "weighted_similarity": 1.0,
                        "contribution": 0.03,
                    })
                    break
        known_words = {
            word for field in ("name", "producer", "grapes")
            for value in self.fields[wine_id][field]
            for word in value.split()
        }
        missing_rare = [
            word for word in rare_words
            if not any(similarity(word, known) >= 0.85 for known in known_words)
        ]
        # A rare, correctly read brand on the label must count against an
        # unrelated wine that merely shares its varietal name.
        score = max(0.0, score - min(0.30, 0.12 * len(missing_rare)))
        return {
            "wine_id": wine_id,
            "name": product["name"],
            "producer": product.get("producer"),
            "image_url": product.get("image_url"),
            "match_score": round(min(1.0, score), 4),
            "evidence": evidence,
        }

    @staticmethod
    def evidence_coverage(lines: list[OCRLine], evidence: list[dict[str, Any]]) -> float:
        """Fraction of descriptive OCR lines explained by identifying fields."""
        descriptive = [
            index for index, line in enumerate(lines)
            if any(len(word) >= 4 and ALPHA.search(word) for word in line.keys[0].split())
        ]
        if not descriptive:
            return 0.0
        matched = {
            index for item in evidence
            if item["field"] in {"name", "producer", "grapes"}
            and item["similarity"] >= 0.78
            for index in item["ocr_indices"]
        }
        return len(matched.intersection(descriptive)) / len(descriptive)

    def match(self, ocr_lines: Iterable[str | dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        # If its better to show the producer shows itself first, then this block should be replaced with the commented
        # however that increase the total time by about 1s, 0.4s median - benchmarked.
        parsed = parse_ocr_lines(ocr_lines)
        source = [{"text": line.text, "confidence": line.confidence} for line in parsed]
        result = {
            "status": "unresolved",
            "ocr_lines": source,
            "best_match": None,
            "alternatives": [],
            "candidates": [],
            "score_type": "uncalibrated_catalog_match_score",
        }
        if not parsed:
            return result

        # parsed = parse_ocr_lines(ocr_lines)
        # source = [
        #     {"text": line.text, "confidence": line.confidence}
        #     for line in parsed
        # ]
        
        # result = {
        #     "status": "unresolved",
        #     "ocr_lines": source,
        #     "best_match": None,
        #     "alternatives": [],
        #     "candidates": [],
        #     "score_type": "uncalibrated_catalog_match_score",
        #     "recognized_producer": None,
        #     "producer_wines": [],
        # }
        
        # if not parsed:
        #     return result
        
        # producer_key = next(
        #     (
        #         key
        #         for line in parsed
        #         if line.confidence is None or line.confidence >= 0.80
        #         for key in line.keys
        #         if len(key) >= 5 and key in self.producer_ids
        #     ),
        #     None,
        # )
        
        # if producer_key:
        #     ids = self.producer_ids[producer_key]
        #     result["recognized_producer"] = self.products[ids[0]]["producer"]
        #     result["producer_wines"] = [
        #         {
        #             "wine_id": wine_id,
        #             "name": self.products[wine_id]["name"],
        #             "slug": self.products[wine_id].get("slug"),
        #             "image_url": self.products[wine_id].get("image_url"),
        #         }
        #         for wine_id in ids
        #     ]

        queries = query_lines(parsed)
        ids = self.retrieve(queries)
        if not ids:
            return result
        rare_words = {
            word for line in parsed for key in line.keys for word in key.split()
            if len(word) >= 5 and 0 < len(self.token_index.get(word, ())) <= 100
        }
        ranked = sorted(
            (self.rank(wine_id, queries, parsed, rare_words) for wine_id in ids),
            key=lambda item: (-item["match_score"], item["wine_id"]),
        )
        ranked = ranked[:top_k]
        first = ranked[0]
        first["evidence_coverage"] = round(
            self.evidence_coverage(parsed, first["evidence"]), 3
        )
        runner_up = ranked[1]["match_score"] if len(ranked) > 1 else 0.0
        name_evidence = next(
            (e for e in first["evidence"] if e["field"] == "name"), None
        )
        alias_owners = (
            self.alias_owners.get(name_evidence["catalog_key"], set())
            if name_evidence else set()
        )
        producer_supported = any(
            e["field"] == "producer" and e["similarity"] >= 0.85
            for e in first["evidence"]
        )
        name_identifies_product = bool(
            name_evidence and len(alias_owners) == 1
            and len(name_evidence["catalog_key"]) >= 8
            and name_evidence["weighted_similarity"] >= 0.80
        )
        unique_exact_alias = bool(
            name_evidence
            and name_evidence["catalog_key"] in search_forms(name_evidence["ocr_text"])
            and len(name_evidence["catalog_key"]) >= 6
            and self.specificity(name_evidence["catalog_key"]) >= 0.60
            and alias_owners == {first["wine_id"]}
        )
        unique_fuzzy_alias = False
        if name_evidence:
            alias = name_evidence["catalog_key"]
            ocr_key = search_key(name_evidence["ocr_text"])
            # Full phrase, similar length, and a rare token that survives a
            # one-character OCR error. A partial match on "Ruby" or "Blend"
            # does not qualify as a bottle identity.
            rare_token_seen = any(
                len(word) >= 5
                and 0 < len(self.token_index.get(word, ())) <= 3
                and any(similarity(typed, word) >= 0.75 for typed in ocr_key.split())
                for word in alias.split()
            )
            unique_fuzzy_alias = (
                len(alias) >= 8
                and len(ocr_key) >= 8
                and len(alias.split()) == len(ocr_key.split())
                and name_evidence["similarity"] >= 0.85
                and alias_owners == {first["wine_id"]}
                and rare_token_seen
            )
        if first["match_score"] >= 0.52 and name_evidence and (name_identifies_product or producer_supported) and (
            first["match_score"] - runner_up >= 0.08 or unique_exact_alias or unique_fuzzy_alias
        ) and (first["evidence_coverage"] >= 0.65 or unique_exact_alias or unique_fuzzy_alias) and (
            len(parsed) > 1 or name_evidence["weighted_similarity"] >= 0.75
        ):
            result["status"] = "matched"
        elif first["match_score"] >= 0.40 and name_evidence and first["evidence_coverage"] >= 0.60:
            result["status"] = "ambiguous"
        # Two catalog SKUs may have identical name and producer but different
        # grapes, which are not printed on the observed front label.
        if (
            result["status"] == "unresolved"
            and producer_supported and len(ranked) > 1
            and search_key(first["name"]) == search_key(ranked[1]["name"])
            and search_key(first["producer"]) == search_key(ranked[1]["producer"])
            and first["match_score"] >= 0.55
        ):
            result["status"] = "ambiguous"
        if len(alias_owners) >= 3 and not producer_supported:
            result["status"] = "unresolved"
        # A bare grape or style, without a brand, describes dozens of wines.
        near_top = sum(
            candidate["match_score"] >= first["match_score"] - 0.10
            for candidate in ranked
        )
        if (
            result["status"] == "ambiguous"
            and near_top >= min(5, top_k)
            and not any(e["field"] == "producer" for e in first["evidence"])
        ):
            result["status"] = "unresolved"
        if result["status"] == "unresolved":
            result["rejection_reason"] = (
                "No sufficiently supported product match in this catalog; "
                "check whether the label's wine is present."
            )
            return result
        result["best_match"] = first
        result["alternatives"] = ranked[1:]
        result["candidates"] = ranked
        return result


@lru_cache(maxsize=4)
def _load_catalog(path: str, modified_ns: int) -> WineMatcher:
    return WineMatcher(path)


def match_wines(
    ocr_lines: Iterable[str | dict[str, Any]],
    catalog_path: str | Path = "dataset/wine_catalog.jsonl",
    top_k: int = 5,
) -> dict[str, Any]:
    """Backend entrypoint: keep index resident and reload after catalog updates."""
    path = Path(catalog_path).expanduser().resolve()
    return _load_catalog(str(path), path.stat().st_mtime_ns).match(ocr_lines, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("dataset/wine_catalog.jsonl"))
    parser.add_argument("--line", action="append", required=True, help="One OCR line; repeat for each line")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(match_wines(args.line, args.catalog, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
