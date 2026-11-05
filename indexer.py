"""조각과 임베딩을 index/ 에 저장한다. 바뀐 것이 없으면 API 를 부르지 않고 파일에서 불러온다.

실행:  python indexer.py            필요할 때만 생성
       python indexer.py --rebuild  강제로 다시 생성

다시 만드는 조건 (meta.json 과 비교):
  문서 내용(doc_hash) / 임베딩 모델 / 차원 / 조각 설정 / 임베딩 입력 텍스트(inputs_hash) 중 하나라도 다를 때,
  또는 파일이 없거나 embeddings.npy 크기가 맞지 않을 때.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime

import numpy as np

import chunker
import config
import embedder
import gemini_api

EMB_PATH = config.INDEX_DIR / "embeddings.npy"
META_PATH = config.INDEX_DIR / "meta.json"

# 비교할 항목과 다를 때 보여 줄 이유
CHECK_KEYS = [
    ("doc_hash", "문서 내용 변경"),
    ("embed_model", "임베딩 모델 변경"),
    ("dim", "차원 변경"),
    ("chunk_settings", "조각 설정 변경"),
    ("inputs_hash", "조각 내용 변경"),
]


def sha256_files(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def sha256_texts(texts):
    return hashlib.sha256("\n\0".join(texts).encode("utf-8")).hexdigest()


def build_meta(files, inputs):
    return {
        "doc_hash": sha256_files(files),
        "files": [f.name for f in files],
        "embed_model": config.EMBED_MODEL,
        "dim": config.EMBED_DIM,
        "normalized": True,
        "chunk_settings": {"max_chars": config.CHUNK_MAX_CHARS, "min_chars": config.CHUNK_MIN_CHARS},
        "num_chunks": len(inputs),
        "inputs_hash": sha256_texts(inputs),
    }


def rebuild_reason(meta):
    """다시 만들어야 하면 이유(문자열), 아니면 None."""
    if not EMB_PATH.exists() or not META_PATH.exists():
        return "인덱스 파일 없음"
    old = json.loads(META_PATH.read_text(encoding="utf-8"))
    for key, reason in CHECK_KEYS:
        if old.get(key) != meta[key]:
            return reason
    shape = np.load(EMB_PATH, mmap_mode="r").shape
    if shape != (meta["num_chunks"], meta["dim"]):
        return f"embeddings.npy 크기 불일치 {shape}"
    return None


def load_index(rebuild=False):
    """(조각 목록, 임베딩 행렬) 을 돌려준다. 행 i 는 조각 번호 i+1 의 벡터."""
    chunks, files = chunker.build_chunks()
    chunker.save(chunks, files)
    inputs = [embedder.doc_input(c.path_full, c.text) for c in chunks]
    meta = build_meta(files, inputs)

    reason = "--rebuild 지정" if rebuild else rebuild_reason(meta)
    if reason is None:
        print("인덱스 불러옴 (API 호출 없음)")
        return chunks, np.load(EMB_PATH)

    print(f"인덱스 생성: {reason} → 조각 {len(chunks)}개 임베딩")
    matrix = np.asarray(embedder.embed_texts(inputs), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)   # 길이 1로 맞추면 코사인 유사도 = 내적
    np.save(EMB_PATH, matrix)
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    # meta 는 임베딩 저장이 끝난 뒤에 쓴다 → 중간에 실패하면 다음 실행에서 다시 만든다
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks, matrix


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="조각 임베딩 인덱스 생성/불러오기")
    parser.add_argument("--rebuild", action="store_true", help="강제로 다시 생성")
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        chunks, emb = load_index(rebuild=args.rebuild)
    except gemini_api.DailyQuotaExceeded as e:
        sys.exit(f"오늘 한도 소진: 내일 다시 실행해 주세요. ({e})")
    print(f"조각 {len(chunks)}개 · 임베딩 {emb.shape[0]}행 × {emb.shape[1]}차원 · "
          f"API 호출 {gemini_api.api_calls}회 · {time.perf_counter() - start:.1f}초")
