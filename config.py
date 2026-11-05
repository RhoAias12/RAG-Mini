"""프로젝트 설정값 모음. 값을 바꿀 때는 이 파일만 고치면 된다."""
from pathlib import Path

ROOT = Path(__file__).parent

# --- 경로 ---
DATA_DIR = ROOT / "data" / "private"   # 원본 문서 폴더 (.md, .txt 를 모두 읽음)
INDEX_DIR = ROOT / "index"             # 조각·임베딩 저장 폴더 (git 제외)
DOC_EXTENSIONS = (".md", ".txt")

# --- 조각(chunk) 나누기 ---
# 길이는 "임베딩에 넣는 정제된 텍스트"의 len() 기준 (공백·줄바꿈 포함)
CHUNK_MAX_CHARS = 800   # 이보다 긴 섹션만 더 잘게 나눈다 (병합으로 인한 초과는 허용)
CHUNK_MIN_CHARS = 80    # 이보다 짧은 일반 조각은 이웃 조각과 합친다 (표 조각은 병합하지 않음)
                        # 80: Part 1~4 도입부(43~60자)가 각자의 '핵심 분석 결과'에 합쳐지도록

# --- 임베딩 (Gemini) ---
# 무료 등급 한도 (사용자가 AI Studio에서 확인): RPM 100, TPM 30K, RPD 1K
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
EMBED_BATCH_SIZE = 10            # 한 요청에 보내는 조각 수
EMBED_MAX_TOKENS_PER_REQUEST = 6000   # 공식 한도 8,192 토큰보다 여유 있게
EMBED_RPM_LIMIT = 100
EMBED_TPM_LIMIT = 30_000
EMBED_TPM_SAFETY = 0.8           # 한도의 80%까지만 사용 → 분당 약 24K
# 토큰 수는 글자 수로 어림잡는다 (한국어는 보통 토큰 수 < 글자 수라 넉넉한 쪽으로 계산됨)

# --- 재시도 (429 분당 한도 / 5xx 서버 오류) ---
MAX_RETRIES = 5                  # 최대 재시도 횟수
BACKOFF_BASE_SEC = 2             # 2 → 4 → 8 → 16 → 32초
# 429 중 '일일 한도 소진'은 재시도하지 않고 멈춘다

# --- 답변 생성 (다음 단계에서 구현) ---
LLM_PROVIDER = "gemini"          # "gemini" 또는 "claude" (claude는 요청 시 추가, 키 이름 RAG_CLAUDE_KEY)
GEMINI_LLM_MODEL = "gemini-3.5-flash-lite"           # 무료: RPM 15, RPD 500
GEMINI_LLM_FALLBACK_MODEL = "gemini-3.1-flash-lite"  # 예비
LLM_DELAY_SEC = 5                # 답변 요청 사이 대기
