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
