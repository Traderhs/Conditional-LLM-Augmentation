"""Generate the locked Binary Matched-Size Augmentation Experiment candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from lmstudio import (
    DEFAULT_LMSTUDIO_API_URL,
    EndpointConfig,
    LmStudioGenerationSettings,
    LmStudioRequestError,
    loaded_lmstudio_models,
    parse_lmstudio_endpoints,
    request_lmstudio_chat_content,
)


EXPERIMENT_NAME = "Offensive Language Matched-Size Augmentation Experiment"
EXPERIMENT_ID = "offensive_language_matched_size_experiment"
PATH_ID = "OffensiveLanguage"
PROTOCOL_VERSION = "journal-matched-size-offensive-v1"
PROMPT_ID = "offensive_binary_v1"
MODEL_ID = "google/gemma-4-31b"
DATABASE_SCHEMA_VERSION = "offensive-language-generation-v1"
REQUESTS_PER_CELL = 28_000
TOTAL_REQUESTS = 140_000
CANDIDATES_PER_SEED = 20
MAX_ATTEMPTS = 3
DEFAULT_SHARD_SIZE = 1_000
DEFAULT_GENERATION_CONCURRENCY = 12

TARGET_CELLS = (
    "en_binary_olid",
    "ko_binary_kold",
    "bn_binary_tb_olid",
    "ha_binary_hausahate",
    "ml_binary_dravidian_offensive",
)
CELL_LANGUAGE = {
    "en_binary_olid": "en",
    "ko_binary_kold": "ko",
    "bn_binary_tb_olid": "bn",
    "ha_binary_hausahate": "ha",
    "ml_binary_dravidian_offensive": "ml",
}
INHERITED_LABEL = {0: "non-offensive", 1: "offensive"}
ALLOWED_LABELS = {"non-offensive", "offensive"}

BINARY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "label": {"type": "string", "enum": ["non-offensive", "offensive"]},
    },
    "required": ["text", "label"],
    "additionalProperties": False,
}

COMMON_JSON_HEADER = (
    "IMPORTANT:\n"
    "You MUST output ONLY one valid JSON object.\n"
    "No introductions, no explanations, no code fences, no comments.\n"
    "The first character must be '{' and the last must be '}'.\n"
    'Schema: {"text": string, "label": "non-offensive|offensive"}\n\n'
)

PROMPT_BODIES = {
    "en": (
        "Output exactly one JSON object and nothing else.\n"
        'Format: {"text":"...", "label":"non-offensive|offensive"}\n'
        "Target label: {label}\n"
        "Target language: English\n"
        "Seed sentence: {seed_text_clipped_to_20_words}\n"
        "Generate exactly one new utterance from the same kind of online/social-text domain, "
        "about the same specific subject or communicative act as the seed, and consistent with "
        "the target offensive-language label. Preserve the seed's core meaning, target, and "
        "offensiveness level, but use substantially different wording and sentence structure.\n"
        "Rules:\n"
        "1. Use only the target language in the generated sentence.\n"
        "2. Keep the target label unambiguous; do not make non-offensive text offensive, and do "
        "not sanitize offensive text into non-offensive text.\n"
        "3. Do not make offensive content more extreme than the seed.\n"
        "4. Do not copy, closely paraphrase, or merely replace a few words from the seed.\n"
        "5. Do not invent facts, situations, real identities, events, or numbers absent from the "
        "seed. Preserve anonymization tokens such as @USER and URL when relevant.\n"
        "6. Do not introduce a different target, protected group, situation, or unrelated topic.\n"
        "7. Do not use meaningless strings, mixed languages, or excessive repetition.\n"
        "8. Output exactly one utterance with no more than 30 words."
    ),
    "ko": (
        "JSON 객체 1개만 출력하세요. 다른 텍스트는 출력하지 마세요.\n"
        '형식: {"text":"...", "label":"non-offensive|offensive"}\n'
        "목표 라벨: {label}\n"
        "목표 언어: 한국어\n"
        "시드 문장: {seed_text_clipped_to_20_words}\n"
        "시드와 같은 온라인/소셜 텍스트 맥락에서 동일한 구체적 주제나 발화 행위를 다루고, "
        "목표 공격적 언어 라벨과 일치하는 새로운 발화 1개를 생성하세요. 시드의 핵심 의미, "
        "공격 대상, 공격성 수준을 유지하되 충분히 다른 표현과 문장 구조를 사용하세요.\n"
        "규칙:\n"
        "1. 생성 문장에는 목표 언어만 사용하세요.\n"
        "2. 목표 라벨이 명확해야 합니다. non-offensive 문장을 offensive로 만들거나 offensive "
        "문장을 non-offensive로 순화하지 마세요.\n"
        "3. offensive 문장은 시드보다 더 극단적인 공격 표현으로 강화하지 마세요.\n"
        "4. 시드를 복사하거나 가깝게 의역하거나 일부 단어만 바꾸지 마세요.\n"
        "5. 시드에 없는 사실, 상황, 실제 인물, 사건, 숫자를 만들지 마세요. @USER, URL 같은 "
        "익명화 토큰은 관련되는 경우 유지하세요.\n"
        "6. 다른 공격 대상, 집단, 상황 또는 관련 없는 주제를 새로 도입하지 마세요.\n"
        "7. 무의미한 문자열, 다른 언어의 혼입 또는 과도한 반복을 사용하지 마세요.\n"
        "8. 정확히 발화 1개만, 30단어 이내로 작성하세요."
    ),
    "bn": (
        "শুধু একটি JSON অবজেক্ট আউটপুট করুন। অন্য কোনো লেখা আউটপুট করবেন না।\n"
        'ফরম্যাট: {"text":"...", "label":"non-offensive|offensive"}\n'
        "লক্ষ্য লেবেল: {label}\n"
        "লক্ষ্য ভাষা: বাংলা\n"
        "মূল বাক্য: {seed_text_clipped_to_20_words}\n"
        "মূল বাক্যের মতো একই অনলাইন/সামাজিক লেখার প্রেক্ষাপটে, একই নির্দিষ্ট বিষয় বা কথন-উদ্দেশ্য "
        "নিয়ে এবং লক্ষ্য offensive-language লেবেলের সঙ্গে সামঞ্জস্যপূর্ণ ঠিক একটি নতুন উক্তি তৈরি করুন। "
        "মূল অর্থ, লক্ষ্য এবং আক্রমণাত্মকতার মাত্রা বজায় রাখুন, তবে যথেষ্ট ভিন্ন শব্দ ও বাক্যগঠন ব্যবহার করুন।\n"
        "নিয়ম:\n"
        "1. তৈরি করা উক্তিতে শুধু লক্ষ্য ভাষা ব্যবহার করুন।\n"
        "2. লক্ষ্য লেবেল দ্ব্যর্থহীন রাখুন; non-offensive লেখাকে offensive করবেন না এবং offensive "
        "লেখাকে non-offensive করে নরম করবেন না।\n"
        "3. offensive লেখা মূল বাক্যের তুলনায় আরও চরম করবেন না।\n"
        "4. মূল বাক্য কপি করবেন না, খুব কাছাকাছি ভাষায় পুনর্লিখন করবেন না, বা কয়েকটি শব্দ শুধু বদলাবেন না।\n"
        "5. মূল বাক্যে নেই এমন তথ্য, পরিস্থিতি, বাস্তব পরিচয়, ঘটনা বা সংখ্যা তৈরি করবেন না; প্রাসঙ্গিক হলে "
        "@USER ও URL-এর মতো anonymization token বজায় রাখুন।\n"
        "6. নতুন লক্ষ্য, গোষ্ঠী, পরিস্থিতি বা সম্পর্কহীন বিষয় যোগ করবেন না।\n"
        "7. অর্থহীন লেখা, ভাষা মেশানো বা অতিরিক্ত পুনরাবৃত্তি ব্যবহার করবেন না।\n"
        "8. ঠিক একটি উক্তি লিখুন এবং সর্বোচ্চ ৩০টি শব্দ ব্যবহার করুন।"
    ),
    "ha": (
        "Fitar da abu guda ɗaya na JSON kawai. Kada ka fitar da wani rubutu.\n"
        'Tsari: {"text":"...", "label":"non-offensive|offensive"}\n'
        "Alamar da ake nufi: {label}\n"
        "Harshen da ake nufi: Hausa\n"
        "Jimlar tushe: {seed_text_clipped_to_20_words}\n"
        "Ƙirƙiri sabuwar magana guda ɗaya a irin wannan mahallin rubutun yanar gizo/sada zumunta, "
        "game da takamaiman batu ko aikin magana iri ɗaya da jimlar tushe, kuma mai dacewa da alamar "
        "offensive-language da ake nufi. Riƙe ainihin ma'ana, abin da ake nufi, da matakin cin mutunci, "
        "amma yi amfani da kalmomi da tsarin jimla masu bambanci sosai.\n"
        "Ka'idoji:\n"
        "1. Yi amfani da harshen da ake nufi kawai.\n"
        "2. Ka sa alamar ta kasance a sarari; kada ka mai da non-offensive zuwa offensive, kuma kada ka "
        "sassauta offensive ya zama non-offensive.\n"
        "3. Kada ka ƙara tsananta offensive fiye da jimlar tushe.\n"
        "4. Kada ka kwafi ko ka sake faɗar jimlar tushe da kalmomi kusan iri ɗaya.\n"
        "5. Kada ka ƙirƙiri bayanai, yanayi, ainihin mutane, abubuwan da suka faru, ko lambobi da babu su "
        "a tushe; ka riƙe alamomin @USER da URL idan sun dace.\n"
        "6. Kada ka gabatar da wani sabon abin hari, rukuni, yanayi, ko batu marar alaƙa.\n"
        "7. Kada ka yi amfani da rubutu marar ma'ana, haɗa harsuna, ko maimaitawa fiye da kima.\n"
        "8. Rubuta magana guda ɗaya tak kuma kada ta wuce kalmomi 30."
    ),
    "ml": (
        "ഒരു JSON ഒബ്ജക്റ്റ് മാത്രം ഔട്ട്പുട്ട് ചെയ്യുക. മറ്റൊരു എഴുത്തും ഔട്ട്പുട്ട് ചെയ്യരുത്.\n"
        'ഫോർമാറ്റ്: {"text":"...", "label":"non-offensive|offensive"}\n'
        "ലക്ഷ്യ ലേബൽ: {label}\n"
        "ലക്ഷ്യ ഭാഷ: മലയാളം\n"
        "സീഡ് വാക്യം: {seed_text_clipped_to_20_words}\n"
        "സീഡിനോട് സാമ്യമുള്ള ഓൺലൈൻ/സോഷ്യൽ ടെക്സ്റ്റ് സാഹചര്യത്തിൽ, അതേ നിർദ്ദിഷ്ട വിഷയമോ സംസാര പ്രവർത്തിയോ "
        "സംബന്ധിച്ചും ലക്ഷ്യ offensive-language ലേബലുമായി പൊരുത്തപ്പെടുന്നതുമായ കൃത്യമായി ഒരു പുതിയ ഉച്ചാരണം "
        "സൃഷ്ടിക്കുക. സീഡിന്റെ മുഖ്യ അർത്ഥം, ലക്ഷ്യം, ആക്രമണാത്മകതയുടെ നില എന്നിവ നിലനിർത്തുക, എന്നാൽ ഗണ്യമായി "
        "വ്യത്യസ്തമായ വാക്കുകളും വാക്യഘടനയും ഉപയോഗിക്കുക.\n"
        "നിയമങ്ങൾ:\n"
        "1. സൃഷ്ടിക്കുന്ന ഉച്ചാരണത്തിൽ ലക്ഷ്യ ഭാഷ മാത്രം ഉപയോഗിക്കുക.\n"
        "2. ലക്ഷ്യ ലേബൽ സംശയമില്ലാതെ നിലനിർത്തുക; non-offensive എഴുത്തിനെ offensive ആക്കരുത്, offensive "
        "എഴുത്തിനെ non-offensive ആയി മൃദൂകരിക്കരുത്.\n"
        "3. offensive ഉള്ളടക്കം സീഡിനേക്കാൾ കൂടുതൽ അതിരൂക്ഷമാക്കരുത്.\n"
        "4. സീഡ് പകർത്തുകയോ വളരെ അടുത്ത രീതിയിൽ പുനരാഖ്യാനം ചെയ്യുകയോ കുറച്ച് വാക്കുകൾ മാത്രം മാറ്റുകയോ ചെയ്യരുത്.\n"
        "5. സീഡിൽ ഇല്ലാത്ത വസ്തുതകൾ, സാഹചര്യങ്ങൾ, യഥാർത്ഥ വ്യക്തിത്വങ്ങൾ, സംഭവങ്ങൾ അല്ലെങ്കിൽ സംഖ്യകൾ സൃഷ്ടിക്കരുത്; "
        "പ്രസക്തമായാൽ @USER, URL പോലുള്ള anonymization token നിലനിർത്തുക.\n"
        "6. പുതിയ ആക്രമണ ലക്ഷ്യം, ഗ്രൂപ്പ്, സാഹചര്യം അല്ലെങ്കിൽ ബന്ധമില്ലാത്ത വിഷയം ചേർക്കരുത്.\n"
        "7. അർത്ഥമില്ലാത്ത എഴുത്ത്, ഭാഷാ കലർച്ച, അമിത ആവർത്തനം എന്നിവ ഉപയോഗിക്കരുത്.\n"
        "8. കൃത്യമായി ഒരു ഉച്ചാരണം മാത്രം എഴുതുക; പരമാവധി 30 വാക്കുകൾ."
    ),
}
PROMPT_TEMPLATES = {
    language: COMMON_JSON_HEADER + body + "\n"
    for language, body in PROMPT_BODIES.items()
}

STRICT_JSON_WRAPPER = "Return only the JSON object. Do not emit any text before or after it."
LABEL_MISMATCH_RETRY = "The previous output used the wrong label. Use exactly the target label shown above."
REPETITION_RETRY = (
    "The previous output entered a repetition loop. Generate a new natural sentence "
    "without repeated or meaningless text."
)
WORD_LIMIT_RETRY = "The previous output exceeded the word limit. Use no more than 30 words."
SURFACE_CORRUPTION_RETRY = (
    "The previous output contained corrupted text. Generate a natural sentence "
    "without malformed text."
)
INVALID_REASONS = {
    "empty_response", "json_parse_failure", "json_not_object", "unexpected_fields",
    "missing_text", "missing_label", "text_not_string", "label_not_string",
    "empty_generated_text", "invalid_label_value", "label_mismatch", "exact_copy",
    "repeated_token", "repeated_segment", "word_limit_exceeded", "surface_corruption",
}

GENERATION_SETTINGS = LmStudioGenerationSettings(
    temperature=0.8,
    top_p=0.9,
    max_tokens=256,
    repeat_penalty=1.0,
    reasoning_effort="none",
    stream=False,
)
GENERATION_CONFIG: dict[str, Any] = {
    "model": MODEL_ID,
    "reasoning_effort": "none",
    "temperature": 0.8,
    "top_p": 0.9,
    "repeat_penalty": 1.0,
    "max_tokens": 256,
    "stream": False,
    "response_format": "strict_json_schema",
    "seed_mode": "request_specific_attempt_seed",
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "offensive_binary_v1_response",
        "strict": True,
        "schema": BINARY_OUTPUT_SCHEMA,
    },
}


class ManifestValidationError(RuntimeError):
    """Raised before any model call when a locked input is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_log(message: str) -> None:
    print(f"[{utc_now()}] {message}", file=sys.stderr, flush=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_int(value: object, field: str) -> int:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ManifestValidationError(f"{field} is not an integer: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ManifestValidationError(f"{field} is not an integer: {value!r}")
    return int(number)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_copy(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def clip_words(value: str, count: int = 20) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()[:count])


def attempt_generation_seed(base_generation_seed: int, attempt_number: int) -> int:
    if attempt_number == 1:
        return int(base_generation_seed)
    used = {int(base_generation_seed) % (2**32)}
    derived = 0
    for current_attempt in range(2, attempt_number + 1):
        source = f"{int(base_generation_seed)}:{current_attempt}".encode("utf-8")
        derived = int.from_bytes(hashlib.sha256(source).digest()[:4], "big")
        while derived in used:
            derived = (derived + 1) % (2**32)
        used.add(derived)
    return derived


def render_prompt(template: str, inherited_label: str, seed_text: str) -> str:
    return template.replace("{label}", inherited_label).replace(
        "{seed_text_clipped_to_20_words}", clip_words(seed_text, 20)
    )


def detect_repetition_collapse(text: str) -> str | None:
    if re.search(r'(?:^|[\s"])([^\s"]+)(?:\s+\1){7}(?=\s|["}]|$)', text):
        return "repeated_token"
    for repeated in re.finditer(r"(.{1,4})\1{7,}", text):
        if len(repeated.group(0)) >= 32:
            return "repeated_segment"
    return None


def has_surface_corruption(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    if any(unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in text):
        return True
    if re.search(r"([/\\]{2})[^\W_]{1,4}\1", text, flags=re.UNICODE):
        return True
    return bool(re.search(r"(?<![\w/\\])([/\\])[^\W_]{1,4}\1(?![\w/\\])", text, flags=re.UNICODE))


def validate_final_json(
    final_response: str, seed_text: str, inherited_label: str
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    metadata = {"json_schema_valid": False, "generated_text": None, "generated_label_raw": None}
    if not isinstance(final_response, str) or not final_response.strip():
        return None, "empty_response", metadata
    repetition_reason = detect_repetition_collapse(final_response)
    if repetition_reason is not None:
        return None, repetition_reason, metadata
    try:
        value = json.loads(final_response.strip())
    except Exception:
        return None, "json_parse_failure", metadata
    if not isinstance(value, dict):
        return None, "json_not_object", metadata
    if set(value.keys()) != {"text", "label"}:
        return None, "unexpected_fields", metadata
    if "text" not in value:
        return None, "missing_text", metadata
    if "label" not in value:
        return None, "missing_label", metadata
    if not isinstance(value["text"], str):
        return None, "text_not_string", metadata
    if not isinstance(value["label"], str):
        return None, "label_not_string", metadata
    generated_text = value["text"].strip()
    generated_label_raw = value["label"]
    metadata.update(json_schema_valid=True, generated_text=generated_text, generated_label_raw=generated_label_raw)
    if not generated_text:
        return None, "empty_generated_text", metadata
    if generated_label_raw not in ALLOWED_LABELS:
        return None, "invalid_label_value", metadata
    if generated_label_raw != inherited_label:
        return None, "label_mismatch", metadata
    if normalize_for_copy(seed_text) == normalize_for_copy(generated_text):
        return None, "exact_copy", metadata
    if len(generated_text.split()) > 30:
        return None, "word_limit_exceeded", metadata
    if has_surface_corruption(generated_text):
        return None, "surface_corruption", metadata
    return {"text": generated_text, "label": generated_label_raw}, None, metadata


def validate_manifest_lock(manifest_root: Path) -> dict[str, Any]:
    lock_path = manifest_root / "MANIFEST_LOCK.json"
    if not lock_path.is_file():
        raise ManifestValidationError(f"missing manifest lock: {lock_path}")
    lock = read_json(lock_path)
    if not isinstance(lock, dict) or not lock:
        raise ManifestValidationError("MANIFEST_LOCK.json must be a non-empty object")
    errors: list[str] = []
    root = manifest_root.resolve()
    for relative, expected in sorted(lock.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append(f"invalid lock entry: {relative!r}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"lock path escapes manifest root: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing locked file: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"hash mismatch: {relative}")
    required = {"stage_a_generation_plan.csv"}
    for cell_id in TARGET_CELLS:
        required.update({f"{cell_id}/cell_manifest.json", f"{cell_id}/experiment_bank.csv"})
    for relative in sorted(required):
        if relative not in lock:
            errors.append(f"required file absent from lock: {relative}")
    if errors:
        raise ManifestValidationError("manifest validation failed:\n- " + "\n- ".join(errors))
    return {
        "verified": True,
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        "locked_file_count": len(lock),
    }


def load_and_validate_inputs(
    prepared_root: Path, manifest_root: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    lock_info = validate_manifest_lock(manifest_root)
    banks: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for cell_id in TARGET_CELLS:
        cell_manifest = read_json(manifest_root / cell_id / "cell_manifest.json")
        if not isinstance(cell_manifest, dict) or cell_manifest.get("cell_id") != cell_id:
            raise ManifestValidationError(f"{cell_id}: invalid cell manifest")
        if cell_manifest.get("protocol_version") != PROTOCOL_VERSION:
            raise ManifestValidationError(f"{cell_id}: protocol version mismatch")
        manifests[cell_id] = cell_manifest
        split_hashes = cell_manifest.get("prepared_split_sha256")
        if not isinstance(split_hashes, dict):
            raise ManifestValidationError(f"{cell_id}: prepared split hashes missing")
        for split in ("train", "dev", "test"):
            path = prepared_root / cell_id / f"{split}.csv"
            if not path.is_file() or sha256_file(path) != split_hashes.get(split):
                raise ManifestValidationError(f"{cell_id}: prepared {split} hash mismatch")

        bank_path = manifest_root / cell_id / "experiment_bank.csv"
        with bank_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"row_id", "_source_row", "_text_hash", "text", "label", "bank_class_position"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ManifestValidationError(f"{cell_id}: bank columns missing: {sorted(missing)}")
            rows: list[dict[str, Any]] = []
            for bank_index, raw in enumerate(reader):
                row_id = str(raw.get("row_id") or "").strip()
                text = normalize_text(raw.get("text"))
                text_hash = str(raw.get("_text_hash") or "").strip()
                label = parse_int(raw.get("label"), f"{cell_id}.label")
                source_row = parse_int(raw.get("_source_row"), f"{cell_id}._source_row")
                if not row_id or not text or text_hash != sha256_bytes(text.encode("utf-8")):
                    raise ManifestValidationError(f"{cell_id}: invalid bank row/text hash at {bank_index}")
                if label not in INHERITED_LABEL:
                    raise ManifestValidationError(f"{cell_id}: non-binary label at {bank_index}")
                rows.append({
                    "cell_id": cell_id,
                    "language": CELL_LANGUAGE[cell_id],
                    "experiment_bank_index": bank_index,
                    "real_row_id": row_id,
                    "prepared_row_index": source_row,
                    "seed_text": text,
                    "text_sha256": text_hash,
                    "label_int": label,
                })
        expected_n = parse_int(cell_manifest.get("experiment_bank_n"), f"{cell_id}.experiment_bank_n")
        if len(rows) != expected_n or expected_n != 1_400:
            raise ManifestValidationError(f"{cell_id}: bank size {len(rows)} != 1400")
        if len({row["real_row_id"] for row in rows}) != len(rows):
            raise ManifestValidationError(f"{cell_id}: duplicate bank row_id")
        expected_counts = {str(k): parse_int(v, f"{cell_id}.count") for k, v in cell_manifest["experiment_bank_label_counts"].items()}
        if dict(Counter(str(row["label_int"]) for row in rows)) != expected_counts:
            raise ManifestValidationError(f"{cell_id}: bank label counts mismatch")

        wanted = {row["prepared_row_index"]: row for row in rows}
        found: set[int] = set()
        with (prepared_root / cell_id / "train.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"text", "label"}.issubset(reader.fieldnames or []):
                raise ManifestValidationError(f"{cell_id}: prepared train lacks text/label")
            for source_row, raw in enumerate(reader):
                bank_row = wanted.get(source_row)
                if bank_row is None:
                    continue
                text = normalize_text(raw.get("text"))
                label = parse_int(raw.get("label"), f"{cell_id}.prepared.label")
                if text != bank_row["seed_text"] or sha256_bytes(text.encode("utf-8")) != bank_row["text_sha256"]:
                    raise ManifestValidationError(f"{cell_id}: prepared text mismatch at {source_row}")
                if label != bank_row["label_int"]:
                    raise ManifestValidationError(f"{cell_id}: prepared label mismatch at {source_row}")
                found.add(source_row)
        if found != set(wanted):
            raise ManifestValidationError(f"{cell_id}: prepared rows missing from train")
        banks[cell_id] = rows

    plan_path = manifest_root / "stage_a_generation_plan.csv"
    plan: list[dict[str, Any]] = []
    with plan_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protocol_version", "cell_id", "real_row_id", "label", "candidate_index", "global_generation_index", "generation_seed"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ManifestValidationError(f"generation plan columns missing: {sorted(missing)}")
        for raw in reader:
            cell_id = str(raw.get("cell_id") or "").strip()
            if cell_id not in TARGET_CELLS:
                raise ManifestValidationError(f"generation plan contains unexpected cell: {cell_id}")
            plan.append({
                "protocol_version": str(raw.get("protocol_version") or "").strip(),
                "cell_id": cell_id,
                "language": CELL_LANGUAGE[cell_id],
                "real_row_id": str(raw.get("real_row_id") or "").strip(),
                "label_int": parse_int(raw.get("label"), "generation plan label"),
                "candidate_index": parse_int(raw.get("candidate_index"), "candidate_index"),
                "global_generation_index": parse_int(raw.get("global_generation_index"), "global_generation_index"),
                "generation_seed": parse_int(raw.get("generation_seed"), "generation_seed"),
            })
    if len(plan) != TOTAL_REQUESTS:
        raise ManifestValidationError(f"generation plan has {len(plan)} rows, expected {TOTAL_REQUESTS}")
    indices = [row["global_generation_index"] for row in plan]
    if len(set(indices)) != TOTAL_REQUESTS or set(indices) != set(range(TOTAL_REQUESTS)):
        raise ManifestValidationError("global_generation_index is duplicated or has gaps")
    seeds = [row["generation_seed"] for row in plan]
    if len(set(seeds)) != TOTAL_REQUESTS:
        raise ManifestValidationError("generation_seed is duplicated")
    bank_indexes = {cell_id: {row["real_row_id"]: row for row in rows} for cell_id, rows in banks.items()}
    keys: set[tuple[str, str, int]] = set()
    candidates_by_seed: dict[tuple[str, str], set[int]] = defaultdict(set)
    counts = Counter(row["cell_id"] for row in plan)
    for row in plan:
        cell_id = row["cell_id"]
        bank_row = bank_indexes[cell_id].get(row["real_row_id"])
        if row["protocol_version"] != PROTOCOL_VERSION:
            raise ManifestValidationError(f"{cell_id}: generation plan protocol mismatch")
        if bank_row is None:
            raise ManifestValidationError(f"{cell_id}: unknown real_row_id {row['real_row_id']}")
        if row["label_int"] != bank_row["label_int"]:
            raise ManifestValidationError(f"{cell_id}: plan label mismatch for {row['real_row_id']}")
        if row["candidate_index"] not in range(CANDIDATES_PER_SEED):
            raise ManifestValidationError(f"{cell_id}: invalid candidate index")
        key = (cell_id, row["real_row_id"], row["candidate_index"])
        if key in keys:
            raise ManifestValidationError(f"duplicate generation key: {key}")
        keys.add(key)
        candidates_by_seed[(cell_id, row["real_row_id"])].add(row["candidate_index"])
        row.update(
            seed_text=bank_row["seed_text"],
            inherited_label=INHERITED_LABEL[row["label_int"]],
            prompt_id=PROMPT_ID,
            prompt_sha256=sha256_bytes(PROMPT_TEMPLATES[row["language"]].encode("utf-8")),
        )
    if any(counts[cell_id] != REQUESTS_PER_CELL for cell_id in TARGET_CELLS):
        raise ManifestValidationError(f"per-cell generation counts invalid: {dict(counts)}")
    for cell_id, bank_rows in banks.items():
        for bank_row in bank_rows:
            observed = candidates_by_seed[(cell_id, bank_row["real_row_id"])]
            if observed != set(range(CANDIDATES_PER_SEED)):
                raise ManifestValidationError(f"{cell_id}: incomplete candidates for {bank_row['real_row_id']}")
    plan.sort(key=lambda row: row["global_generation_index"])
    lock_info["generation_plan_sha256"] = sha256_file(plan_path)
    return lock_info, banks, plan


def write_prompt_templates(output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for cell_id in TARGET_CELLS:
        template = PROMPT_TEMPLATES[CELL_LANGUAGE[cell_id]]
        path = output_root / "prompts" / f"{cell_id}_{PROMPT_ID}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template, encoding="utf-8", newline="\n")
        hashes[cell_id] = sha256_bytes(template.encode("utf-8"))
    return hashes


def metadata_contract(lock_info: Mapping[str, Any], prompt_hashes: Mapping[str, str]) -> dict[str, str]:
    response_schema_hash = sha256_bytes(canonical_json(BINARY_OUTPUT_SCHEMA).encode("utf-8"))
    values: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "manifest_lock_hash": lock_info["lock_sha256"],
        "generation_plan_hash": lock_info["generation_plan_sha256"],
        "prompt_hashes": dict(prompt_hashes),
        "generation_request_configuration": GENERATION_CONFIG,
        "generation_model_identifier": MODEL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "response_schema_hash": response_schema_hash,
        "max_attempts": MAX_ATTEMPTS,
    }
    return {key: canonical_json(value) for key, value in values.items()}


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS requests (
            global_generation_index INTEGER PRIMARY KEY,
            protocol_version TEXT NOT NULL,
            cell_id TEXT NOT NULL,
            language TEXT NOT NULL,
            real_row_id TEXT NOT NULL,
            seed_text TEXT NOT NULL,
            label_int INTEGER NOT NULL,
            inherited_label TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            generation_seed INTEGER NOT NULL UNIQUE,
            prompt_id TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','valid','failed')),
            attempt_count INTEGER NOT NULL DEFAULT 0,
            generated_text TEXT,
            generated_label_raw TEXT,
            terminal_failure_reason TEXT,
            endpoint TEXT,
            model_id TEXT,
            completed_at TEXT,
            attempts_json TEXT,
            UNIQUE(cell_id, real_row_id, candidate_index)
        );
        CREATE TABLE IF NOT EXISTS attempts (
            global_generation_index INTEGER NOT NULL,
            attempt INTEGER NOT NULL,
            attempt_generation_seed INTEGER NOT NULL,
            endpoint TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_variant TEXT NOT NULL,
            raw_response TEXT,
            final_response_text TEXT,
            validation_reason TEXT,
            json_schema_valid INTEGER NOT NULL,
            generated_text TEXT,
            generated_label_raw TEXT,
            backend_error TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            PRIMARY KEY(global_generation_index, attempt),
            FOREIGN KEY(global_generation_index) REFERENCES requests(global_generation_index)
        );
        """
    )
    return connection


def initialize_database(
    connection: sqlite3.Connection,
    plan: Sequence[Mapping[str, Any]],
    contract: Mapping[str, str],
    resume: bool,
) -> None:
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    request_count = connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    if resume:
        if not existing or request_count == 0:
            raise RuntimeError("--resume requires an existing initialized generation database")
        mismatches = [key for key, value in contract.items() if existing.get(key) != value]
        if mismatches:
            raise RuntimeError(f"resume metadata mismatch: {', '.join(sorted(mismatches))}")
        if request_count != TOTAL_REQUESTS:
            raise RuntimeError(f"resume database request count {request_count} != {TOTAL_REQUESTS}")
        return
    if existing or request_count:
        raise RuntimeError("generation database already exists; use --resume")
    created_at = utc_now()
    with connection:
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", [*contract.items(), ("created_at", canonical_json(created_at))])
        connection.executemany(
            """INSERT INTO requests(
                global_generation_index, protocol_version, cell_id, language, real_row_id,
                seed_text, label_int, inherited_label, candidate_index, generation_seed,
                prompt_id, prompt_sha256, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'pending')""",
            [(
                row["global_generation_index"], row["protocol_version"], row["cell_id"], row["language"],
                row["real_row_id"], row["seed_text"], row["label_int"], row["inherited_label"],
                row["candidate_index"], row["generation_seed"], row["prompt_id"], row["prompt_sha256"],
            ) for row in plan],
        )


def status_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    overall = dict(connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status"))
    cells: dict[str, dict[str, int]] = {}
    for cell_id in TARGET_CELLS:
        values = dict(connection.execute("SELECT status, COUNT(*) FROM requests WHERE cell_id=? GROUP BY status", (cell_id,)))
        cells[cell_id] = {status: int(values.get(status, 0)) for status in ("valid", "failed", "pending")}
    valid = int(overall.get("valid", 0))
    failed = int(overall.get("failed", 0))
    pending = int(overall.get("pending", 0))
    return {"completed": valid + failed, "valid": valid, "failed": failed, "pending": pending, "cells": cells}


def pending_requests(connection: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    indices = [row[0] for row in connection.execute(
        "SELECT global_generation_index FROM requests WHERE status='pending' ORDER BY global_generation_index"
    )]
    connection.row_factory = sqlite3.Row
    for index in indices:
        row = connection.execute(
            "SELECT * FROM requests WHERE global_generation_index=? AND status='pending'", (index,)
        ).fetchone()
        if row is not None:
            yield dict(row)
    connection.row_factory = None


class GemmaGenerator:
    def __init__(self, api_url: str, model: str, api_key: str | None, timeout_seconds: int) -> None:
        self.api_url = api_url
        self.model_id = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, seed: int) -> str:
        return request_lmstudio_chat_content(
            api_url=self.api_url,
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            settings=GENERATION_SETTINGS,
            seed=seed,
            response_format=RESPONSE_FORMAT,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )


def retry_instruction(reason: str | None) -> tuple[str, str]:
    if reason == "label_mismatch":
        return "label_mismatch_retry", LABEL_MISMATCH_RETRY
    if reason in {"repeated_token", "repeated_segment"}:
        return "repetition_retry", REPETITION_RETRY
    if reason == "word_limit_exceeded":
        return "word_limit_retry", WORD_LIMIT_RETRY
    if reason == "surface_corruption":
        return "surface_corruption_retry", SURFACE_CORRUPTION_RETRY
    return "strict_json_retry", STRICT_JSON_WRAPPER


def generate_one_request(backend: GemmaGenerator, request: Mapping[str, Any]) -> dict[str, Any]:
    base_prompt = render_prompt(PROMPT_TEMPLATES[str(request["language"])], str(request["inherited_label"]), str(request["seed_text"]))
    attempts: list[dict[str, Any]] = []
    retry_reason: str | None = None
    valid: dict[str, Any] | None = None
    terminal_reason: str | None = None
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        prompt_variant = "base"
        prompt = base_prompt
        if attempt_number > 1:
            name, instruction = retry_instruction(retry_reason)
            prompt_variant = f"{name}_{attempt_number - 1}"
            prompt += "\n\n" + instruction
        attempt_seed = attempt_generation_seed(int(request["generation_seed"]), attempt_number)
        started_at = utc_now()
        record: dict[str, Any] = {
            "attempt": attempt_number,
            "attempt_generation_seed": attempt_seed,
            "endpoint": backend.api_url,
            "model_id": backend.model_id,
            "prompt_variant": prompt_variant,
            "raw_response": None,
            "final_response_text": None,
            "validation_reason": None,
            "json_schema_valid": False,
            "generated_text": None,
            "generated_label_raw": None,
            "backend_error": None,
            "started_at": started_at,
            "completed_at": None,
        }
        try:
            response = backend.generate(prompt, attempt_seed)
            parsed, reason, metadata = validate_final_json(response, str(request["seed_text"]), str(request["inherited_label"]))
            record.update(
                raw_response=response,
                final_response_text=response,
                validation_reason=reason,
                json_schema_valid=bool(metadata["json_schema_valid"]),
                generated_text=metadata["generated_text"],
                generated_label_raw=metadata["generated_label_raw"],
            )
            retry_reason = reason
            terminal_reason = reason
            if parsed is not None and reason is None:
                valid = parsed
        except LmStudioRequestError as exc:
            record.update(raw_response=exc.raw_response, backend_error=f"{type(exc).__name__}: {exc}")
            retry_reason = "backend_error"
            terminal_reason = "backend_error"
        except Exception as exc:
            record.update(backend_error=f"{type(exc).__name__}: {exc}")
            retry_reason = "backend_error"
            terminal_reason = "backend_error"
        record["completed_at"] = utc_now()
        attempts.append(record)
        if valid is not None:
            break
    return {
        "request": dict(request),
        "status": "valid" if valid is not None else "failed",
        "attempts": attempts,
        "generated_text": valid["text"] if valid else None,
        "generated_label_raw": valid["label"] if valid else None,
        "terminal_failure_reason": None if valid else terminal_reason,
        "endpoint": backend.api_url,
        "model_id": backend.model_id,
        "completed_at": utc_now(),
    }


def commit_result(connection: sqlite3.Connection, result: Mapping[str, Any]) -> None:
    request = result["request"]
    attempts = result["attempts"]
    with connection:
        connection.executemany(
            """INSERT INTO attempts(
                global_generation_index, attempt, attempt_generation_seed, endpoint, model_id,
                prompt_variant, raw_response, final_response_text, validation_reason,
                json_schema_valid, generated_text, generated_label_raw, backend_error,
                started_at, completed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(
                request["global_generation_index"], row["attempt"], row["attempt_generation_seed"],
                row["endpoint"], row["model_id"], row["prompt_variant"], row["raw_response"],
                row["final_response_text"], row["validation_reason"], int(row["json_schema_valid"]),
                row["generated_text"], row["generated_label_raw"], row["backend_error"],
                row["started_at"], row["completed_at"],
            ) for row in attempts],
        )
        cursor = connection.execute(
            """UPDATE requests SET status=?, attempt_count=?, generated_text=?,
                generated_label_raw=?, terminal_failure_reason=?, endpoint=?, model_id=?,
                completed_at=?, attempts_json=?
                WHERE global_generation_index=? AND status='pending'""",
            (
                result["status"], len(attempts), result["generated_text"], result["generated_label_raw"],
                result["terminal_failure_reason"], result["endpoint"], result["model_id"],
                result["completed_at"], canonical_json(attempts), request["global_generation_index"],
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"duplicate/non-pending result for {request['global_generation_index']}")


def run_pending_generation(
    connection: sqlite3.Connection,
    endpoints: Sequence[EndpointConfig],
    api_key: str | None,
    timeout_seconds: int,
) -> None:
    backends = [GemmaGenerator(url, model, api_key, timeout_seconds) for url, model, _ in endpoints]
    executors = [ThreadPoolExecutor(max_workers=concurrency) for _, _, concurrency in endpoints]
    slots = [index for index, (_, _, concurrency) in enumerate(endpoints) for _ in range(concurrency)]
    iterator = iter(pending_requests(connection))
    futures: dict[Future[dict[str, Any]], int] = {}
    start = perf_counter()
    initial = status_summary(connection)

    def submit_one(endpoint_index: int) -> bool:
        try:
            request = next(iterator)
        except StopIteration:
            return False
        future = executors[endpoint_index].submit(generate_one_request, backends[endpoint_index], request)
        futures[future] = endpoint_index
        return True

    try:
        for endpoint_index in slots:
            submit_one(endpoint_index)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint_index = futures.pop(future)
                result = future.result()
                commit_result(connection, result)
                summary = status_summary(connection)
                elapsed = perf_counter() - start
                newly_completed = summary["completed"] - initial["completed"]
                rate = newly_completed / elapsed if elapsed > 0 else 0.0
                eta = summary["pending"] / rate if rate > 0 else None
                request = result["request"]
                message = (
                    "generation progress | "
                    f"completed={summary['completed']}/{TOTAL_REQUESTS} | valid={summary['valid']} | "
                    f"failed={summary['failed']} | pending={summary['pending']} | cell_id={request['cell_id']} | "
                    f"global_generation_index={request['global_generation_index']} | "
                    f"candidate_index={request['candidate_index']} | endpoint={result['endpoint']} | "
                    f"attempt_count={len(result['attempts'])} | elapsed_seconds={elapsed:.1f} | "
                    f"requests_per_second={rate:.3f} | ETA={eta:.1f}"
                    if eta is not None
                    else "generation progress | ETA=unknown"
                )
                progress_log(message)
                submit_one(endpoint_index)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def terminal_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute("SELECT * FROM requests WHERE status!='pending' ORDER BY global_generation_index")]
    connection.row_factory = None
    for row in rows:
        row["attempts"] = json.loads(row.pop("attempts_json") or "[]")
    return rows


def output_record(row: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": row["protocol_version"],
        "cell_id": row["cell_id"],
        "language": row["language"],
        "real_row_id": row["real_row_id"],
        "label_int": row["label_int"],
        "inherited_label": row["inherited_label"],
        "candidate_index": row["candidate_index"],
        "global_generation_index": row["global_generation_index"],
        "generation_seed": row["generation_seed"],
        "attempt_count": row["attempt_count"],
        "prompt_id": row["prompt_id"],
        "prompt_sha256": row["prompt_sha256"],
        "model_id": row["model_id"],
        "generation_config": GENERATION_CONFIG,
    }
    if row["status"] == "valid":
        return {**base, "generated_text": row["generated_text"], "generated_label_raw": row["generated_label_raw"]}
    return {**base, "failure_reason": row["terminal_failure_reason"], "attempts": row["attempts"]}


def raw_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **output_record(row),
        "status": row["status"],
        "seed_text": row["seed_text"],
        "attempts": row["attempts"],
        "completed_at": row["completed_at"],
    }


def clear_shards(folder: Path) -> None:
    if folder.is_dir():
        for path in folder.glob("part_*.jsonl"):
            path.unlink()


def write_shards(folder: Path, rows: Sequence[Mapping[str, Any]], shard_size: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    clear_shards(folder)
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        path = folder / f"part_{shard_index:06d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows[start : start + shard_size]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def scope_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = [attempt for row in rows for attempt in row["attempts"]]
    valid_rows = [row for row in rows if row["status"] == "valid"]
    failed_rows = [row for row in rows if row["status"] == "failed"]
    first_success = sum(row["status"] == "valid" and row["attempt_count"] == 1 for row in rows)
    reason_counts = Counter(str(attempt.get("validation_reason")) for attempt in attempts if attempt.get("validation_reason"))
    terminal_counts = Counter(str(row["terminal_failure_reason"]) for row in failed_rows)
    endpoint_counts = Counter(str(row["endpoint"]) for row in rows)
    agreement_count = sum(
        row["generated_label_raw"] == row["inherited_label"]
        for row in valid_rows
    )
    return {
        "requested": len(rows),
        "valid": len(valid_rows),
        "failed": len(failed_rows),
        "pending": 0,
        "first_attempt_success_count": first_success,
        "first_attempt_success_rate": first_success / len(rows) if rows else None,
        "attempt_count_distribution": dict(sorted(Counter(str(row["attempt_count"]) for row in rows).items())),
        "validation_reason_attempt_counts": dict(sorted(reason_counts.items())),
        "terminal_failure_reason_counts": dict(sorted(terminal_counts.items())),
        "generated_label_agreement_count": agreement_count,
        "generated_label_agreement_rate": (
            agreement_count / len(valid_rows)
            if valid_rows
            else None
        ),
        "exact_copy_count": reason_counts.get("exact_copy", 0),
        "repeated_token_count": reason_counts.get("repeated_token", 0),
        "repeated_segment_count": reason_counts.get("repeated_segment", 0),
        "word_limit_exceeded_count": reason_counts.get("word_limit_exceeded", 0),
        "surface_corruption_count": reason_counts.get("surface_corruption", 0),
        "endpoint_counts": dict(sorted(endpoint_counts.items())),
    }


def generation_lock_files(generation_root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(generation_root.rglob("*")):
        if not path.is_file() or path.name in {"GENERATION_LOCK.json", "generation_status.sqlite3", "generation_status.sqlite3-wal", "generation_status.sqlite3-shm"}:
            continue
        files[path.relative_to(generation_root).as_posix()] = sha256_file(path)
    return files


def export_generation(
    connection: sqlite3.Connection,
    generation_root: Path,
    lock_info: Mapping[str, Any],
    prompt_hashes: Mapping[str, str],
    endpoints: Sequence[EndpointConfig],
    shard_size: int,
) -> dict[str, Any]:
    summary = status_summary(connection)
    if summary["pending"] != 0 or summary["completed"] != TOTAL_REQUESTS:
        raise RuntimeError(f"cannot export incomplete generation state: {summary}")
    rows = terminal_rows(connection)
    for cell_id in TARGET_CELLS:
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        write_shards(generation_root / "raw_outputs" / cell_id, [raw_record(row) for row in cell_rows], shard_size)
        write_shards(generation_root / "valid_outputs" / cell_id, [output_record(row) for row in cell_rows if row["status"] == "valid"], shard_size)
        write_shards(generation_root / "failures" / cell_id, [output_record(row) for row in cell_rows if row["status"] == "failed"], shard_size)
    report = {
        "report_version": "binary-matched-size-generation-v1",
        "generated_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "generation_model_identifier": MODEL_ID,
        "generation_request_configuration": GENERATION_CONFIG,
        "endpoints": [{"api_url": url, "model": model, "max_concurrency": concurrency} for url, model, concurrency in endpoints],
        "prompt_hashes": dict(prompt_hashes),
        "manifest_lock_hash": lock_info["lock_sha256"],
        "generation_plan_hash": lock_info["generation_plan_sha256"],
        "cells": {cell_id: scope_stats([row for row in rows if row["cell_id"] == cell_id]) for cell_id in TARGET_CELLS},
        "overall": scope_stats(rows),
    }
    write_json(generation_root / "generation_report.json", report)
    report_fields = ["scope", "cell_id", *report["overall"].keys()]
    report_rows = []
    for cell_id in TARGET_CELLS:
        row = {"scope": "cell", "cell_id": cell_id, **report["cells"][cell_id]}
        report_rows.append({key: canonical_json(value) if isinstance(value, dict) else value for key, value in row.items()})
    overall_row = {"scope": "overall", "cell_id": "__all__", **report["overall"]}
    report_rows.append({key: canonical_json(value) if isinstance(value, dict) else value for key, value in overall_row.items()})
    write_csv(generation_root / "generation_report.csv", report_rows, report_fields)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    lock = {
        "lock_version": "binary-matched-size-generation-v1",
        "created_at": utc_now(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "manifest_lock_hash": lock_info["lock_sha256"],
        "generation_plan_hash": lock_info["generation_plan_sha256"],
        "files": generation_lock_files(generation_root),
    }
    write_json(generation_root / "GENERATION_LOCK.json", lock)
    return report


def verify_lock(root: Path, lock_name: str) -> dict[str, Any]:
    lock_path = root / lock_name
    if not lock_path.is_file():
        raise RuntimeError(f"missing lock: {lock_path}")
    lock = read_json(lock_path)
    if not isinstance(lock, dict) or not isinstance(lock.get("files"), dict):
        raise RuntimeError(f"invalid lock: {lock_path}")
    errors = []
    for relative, expected in lock["files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(relative)
    if errors:
        raise RuntimeError(f"lock verification failed for {lock_path}: {errors[:10]}")
    return lock


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--prepared-root", type=Path, default=project_root / "Data" / "Prepared" / "OffensiveLanguage")
    parser.add_argument("--manifest-root", type=Path, default=project_root / "Data" / "ExperimentManifests" / "OffensiveLanguage" / "v1")
    parser.add_argument("--output-root", type=Path, default=project_root / "Results" / PATH_ID)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--export-shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.export_shard_size <= 0:
        raise SystemExit("timeout and export shard size must be positive")
    try:
        endpoint_args = args.endpoint or [[DEFAULT_LMSTUDIO_API_URL, args.model, str(DEFAULT_GENERATION_CONCURRENCY)]]
        endpoints = parse_lmstudio_endpoints(endpoint_args, args.model, DEFAULT_LMSTUDIO_API_URL)
        if any(model != MODEL_ID for _, model, _ in endpoints):
            raise ValueError(f"all generation endpoints must use {MODEL_ID}")
        prepared_root = args.prepared_root.resolve()
        manifest_root = args.manifest_root.resolve()
        experiment_root = args.output_root.resolve()
        generation_root = experiment_root / "Generation"
        lock_info, _banks, plan = load_and_validate_inputs(prepared_root, manifest_root)
        prompt_hashes = write_prompt_templates(generation_root)
        contract = metadata_contract(lock_info, prompt_hashes)
        print(json.dumps({
            "mode": "resume" if args.resume else "new",
            "requests": len(plan),
            "per_cell": dict(Counter(row["cell_id"] for row in plan)),
            "prompt_hashes": prompt_hashes,
            "generation_config": GENERATION_CONFIG,
            "expected_output_root": str(generation_root),
        }, ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0
        database_path = generation_root / "generation_status.sqlite3"
        connection = open_database(database_path)
        try:
            initialize_database(connection, plan, contract, args.resume)
            progress_log(f"generation {'resume' if args.resume else 'new run'} status | {status_summary(connection)}")
            with loaded_lmstudio_models(endpoints, api_key=args.api_key, timeout_seconds=args.timeout_seconds, log=progress_log):
                run_pending_generation(connection, endpoints, args.api_key, args.timeout_seconds)
            report = export_generation(connection, generation_root, lock_info, prompt_hashes, endpoints, args.export_shard_size)
        finally:
            connection.close()
        print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
        return 0
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
