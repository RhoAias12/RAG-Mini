"""Gemini API 공통 부분: 클라이언트 생성, 429 구분, 재시도. 임베딩과 답변 생성이 함께 쓴다."""
import json
import os
import re
import sys
import time

from google import genai
from google.genai import errors

import config


class DailyQuotaExceeded(Exception):
    """오늘 쓸 수 있는 요청 수를 다 썼다. 재시도해도 소용없음."""


api_calls = 0          # 이번 실행에서 보낸 API 요청 수 (재시도 포함)
_client = None


def get_client():
    """키는 config.py 가 .env 에서 불러 둔 환경변수에서 읽는다. 키 값은 절대 출력하지 않는다."""
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY 가 설정되지 않았습니다. .env.example 을 참고해 .env 를 만들어 주세요.")
        _client = genai.Client(api_key=key)
    return _client


def _error_text(err):
    return json.dumps(err.details, ensure_ascii=False) if err.details else str(err)


def _is_daily_quota(err):
    # 429 응답의 QuotaFailure 에 '...PerDay...' 할당량 ID 가 들어 있으면 일일 한도
    return "PerDay" in _error_text(err)


def _server_retry_delay(err):
    # 서버가 RetryInfo 로 "retryDelay": "34s" 처럼 기다릴 시간을 알려 주면 그 값을 따른다
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', _error_text(err))
    return float(m.group(1)) if m else 0.0


def call_with_retry(fn, label):
    """fn() 을 호출한다. 429(분당)·5xx 는 간격을 늘려 재시도, 429(일일)는 즉시 DailyQuotaExceeded."""
    global api_calls
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            api_calls += 1
            return fn()
        except errors.APIError as e:
            if e.code == 429 and _is_daily_quota(e):
                raise DailyQuotaExceeded(f"{label}: {e.message}") from e
            retryable = e.code == 429 or (e.code is not None and e.code >= 500)
            if not retryable or attempt == config.MAX_RETRIES:
                raise
            wait = max(config.BACKOFF_BASE_SEC * 2 ** attempt, _server_retry_delay(e))
            print(f"  [{label}] {e.code} {e.status} → {wait:.0f}초 후 재시도 "
                  f"({attempt + 1}/{config.MAX_RETRIES})")
            time.sleep(wait)
