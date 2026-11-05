"""근거 조각으로 답변을 만든다. 공급자는 config.LLM_PROVIDER 로 고른다.

지금은 Gemini 만 구현. Claude 는 요청 시 _claude_answer 를 추가한다 (키 이름 RAG_CLAUDE_KEY).

단독 실행:  python llm.py   → 짧은 요청 1개로 답변 모델이 동작하는지 확인 (API 1회)
"""
import sys
import time

from google.genai import types

import config
import gemini_api

NOT_FOUND = "문서에서 찾을 수 없습니다"

INSTRUCTION = (
    "아래 근거만으로 한국어로 답하고, 근거에 없으면 "
    f"'{NOT_FOUND}'라고만 답하라. "
    "추측하거나 근거에 없는 숫자·이름을 만들지 마라. "
    "마크다운 서식 기호(**, # 등) 없이 일반 텍스트로 답하라."
)


def build_prompt(question, chunks):
    evidence = "\n\n".join(
        f"[근거 {i}] (제목 경로: {c.path_full})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"{INSTRUCTION}\n\n### 근거\n{evidence}\n\n### 질문\n{question}"


def generate_answer(question, chunks):
    """질문과 근거 조각 목록(chunker.Chunk)을 받아 답변 문자열을 돌려준다."""
    prompt = build_prompt(question, chunks)
    if config.LLM_PROVIDER == "gemini":
        return _gemini_answer(prompt)
    if config.LLM_PROVIDER == "claude":
        raise NotImplementedError("Claude 답변은 아직 구현되지 않았습니다 (요청 시 추가).")
    raise ValueError(f"알 수 없는 LLM_PROVIDER: {config.LLM_PROVIDER!r} (gemini 또는 claude)")


# ---------------------------------------------------------------- Gemini
_last_call = 0.0


def _wait_for_slot():
    """직전 답변 요청 후 LLM_DELAY_SEC 가 지나지 않았으면 남은 시간만큼 기다린다."""
    global _last_call
    remaining = config.LLM_DELAY_SEC - (time.monotonic() - _last_call)
    if _last_call and remaining > 0:
        time.sleep(remaining)
    _last_call = time.monotonic()


def _gemini_answer(prompt):
    _wait_for_slot()
    resp = gemini_api.call_with_retry(
        lambda: gemini_api.get_client().models.generate_content(
            model=config.GEMINI_LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=config.LLM_TEMPERATURE),
        ),
        label="답변",
    )
    text = (resp.text or "").strip()
    return text or "(빈 응답: 모델이 답변을 돌려주지 않았습니다)"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        reply = _gemini_answer("동작 확인입니다. '확인'이라고만 답하세요.")
    except gemini_api.DailyQuotaExceeded as e:
        sys.exit(f"오늘 한도 소진: {e}")
    print(f"모델 {config.GEMINI_LLM_MODEL} 동작 확인: 응답 {reply!r}, API 호출 {gemini_api.api_calls}회")
