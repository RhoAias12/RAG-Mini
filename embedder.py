"""Gemini 임베딩 호출.

- 조각을 작은 묶음으로 나눠 보내고, 묶음 사이에 대기해 분당 토큰 한도를 넘지 않게 한다.
- 429·재시도 처리는 gemini_api.call_with_retry 가 맡는다.

단독 실행:  python embedder.py   → 짧은 문장 1개로 모델이 동작하는지 확인 (API 1회)
"""
import sys
import time

from google.genai import types

import config
import gemini_api


# ---------------------------------------------------------------- 입력 형식
# gemini-embedding-2 는 task_type 대신 텍스트 앞 접두어로 용도를 알려 준다 (공식 문서 권장 형식)
def doc_input(title, text):
    return f"title: {title} | text: {text}"


def query_input(question):
    return f"task: search result | query: {question}"


def estimate_tokens(text):
    return len(text)    # 어림값: 글자 수 (실제 토큰 수보다 크게 잡히는 쪽)


# ---------------------------------------------------------------- 임베딩
def _embed_request(texts):
    """한 요청으로 여러 텍스트를 임베딩한다.
    gemini-embedding-2 는 문자열 목록을 그대로 넣으면 '하나로 합친' 임베딩 1개를 돌려주므로,
    텍스트마다 Content 로 감싸서 보내야 텍스트별 임베딩을 받는다 (공식 문서)."""
    resp = gemini_api.call_with_retry(
        lambda: gemini_api.get_client().models.embed_content(
            model=config.EMBED_MODEL,
            contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in texts],
            config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
        ),
        label="임베딩",
    )
    vectors = [e.values for e in resp.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(f"보낸 텍스트 {len(texts)}개, 받은 임베딩 {len(vectors)}개 — 개수 불일치")
    return vectors


def make_batches(texts):
    """조각 수(EMBED_BATCH_SIZE)와 어림 토큰 수 둘 다 넘지 않게 묶는다."""
    batches, cur, cur_tokens = [], [], 0
    for t in texts:
        tok = estimate_tokens(t)
        if cur and (len(cur) >= config.EMBED_BATCH_SIZE
                    or cur_tokens + tok > config.EMBED_MAX_TOKENS_PER_REQUEST):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += tok
    if cur:
        batches.append(cur)
    return batches


def wait_after(batch):
    """방금 보낸 묶음 크기만큼 쉬면 분당 토큰이 (한도 × 안전계수) 를 넘지 않는다.
    예) 3,000토큰 묶음, 분당 24,000 허용 → 60 × 3000/24000 = 7.5초"""
    tokens = sum(estimate_tokens(t) for t in batch)
    by_tpm = 60 * tokens / (config.EMBED_TPM_LIMIT * config.EMBED_TPM_SAFETY)
    by_rpm = 60 / config.EMBED_RPM_LIMIT
    return max(by_tpm, by_rpm)


def embed_texts(texts):
    batches = make_batches(texts)
    vectors = []
    for i, batch in enumerate(batches, start=1):
        print(f"  묶음 {i}/{len(batches)}: 조각 {len(batch)}개, 어림 {sum(map(estimate_tokens, batch))}토큰")
        vectors.extend(_embed_request(batch))
        if i < len(batches):
            wait = wait_after(batch)
            print(f"    {wait:.1f}초 대기 (분당 한도 보호)")
            time.sleep(wait)
    return vectors


def embed_query(question):
    return _embed_request([query_input(question)])[0]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        vec = _embed_request([doc_input("none", "임베딩 모델 동작 확인용 짧은 문장입니다.")])[0]
    except gemini_api.DailyQuotaExceeded as e:
        sys.exit(f"오늘 한도 소진: {e}")
    norm = sum(v * v for v in vec) ** 0.5
    print(f"모델 {config.EMBED_MODEL} 동작 확인: 차원 {len(vec)}, 벡터 길이(norm) {norm:.4f}, "
          f"API 호출 {gemini_api.api_calls}회")
