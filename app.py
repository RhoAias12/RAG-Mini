"""문서 근거 챗봇 CLI.

실행:  python app.py              질문 입력 루프 (종료: exit / quit / 종료 / Ctrl+C)
       python app.py --k 6        근거 조각 6개 사용
       python app.py --rebuild    인덱스 강제 재생성 후 시작
"""
import argparse
import sys

from google.genai import errors

import config
import gemini_api
from llm import generate_answer
from retriever import Retriever

PREVIEW_CHARS = 150


def print_result(answer, hits):
    print("\n[답변]")
    print(answer)
    print("\n[근거 조각]")
    for rank, h in enumerate(hits, start=1):
        c = h.chunk
        flat = c.text.replace("\n", " ")
        preview = flat[:PREVIEW_CHARS] + ("…" if len(flat) > PREVIEW_CHARS else "")
        print(f"  {rank}. [{c.id}] {c.path_short} · 유사도 {h.score:.3f}")
        print(f"     {preview}")
    print("\n[출처] " + ", ".join(sorted({h.chunk.file for h in hits})))


def main():
    # Windows 터미널에서 한글이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="문서 근거 RAG 챗봇")
    parser.add_argument("--k", type=int, default=config.TOP_K, help=f"근거 조각 수 (기본 {config.TOP_K})")
    parser.add_argument("--rebuild", action="store_true", help="인덱스 강제 재생성")
    args = parser.parse_args()

    try:
        retriever = Retriever(rebuild=args.rebuild)
    except gemini_api.DailyQuotaExceeded as e:
        sys.exit(f"오늘 한도 소진: 내일 다시 실행해 주세요. ({e})")
    print(f"준비 완료 · 조각 {len(retriever.chunks)}개 · k={args.k} · "
          f"답변 모델 {config.GEMINI_LLM_MODEL} · 종료: exit")

    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit", "종료"):
            break
        if not question:
            continue
        try:
            hits = retriever.search(question, k=args.k)
            answer = generate_answer(question, [h.chunk for h in hits])
        except gemini_api.DailyQuotaExceeded as e:
            print(f"\n오늘 한도 소진: 내일 다시 실행해 주세요. ({e})")
            break
        except errors.APIError as e:
            print(f"\nAPI 오류 {e.code} {e.status}: {e.message}")
            continue
        print_result(answer, hits)


if __name__ == "__main__":
    main()
