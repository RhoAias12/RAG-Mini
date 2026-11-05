"""평가 질문을 일괄 실행해 결과를 CSV 로 저장한다. 정답 판정·LLM 채점은 하지 않는다.

실행:  python eval.py                         eval/questions.tsv 실행 (중단된 실행이 있으면 이어서)
       python eval.py --new                   이어서 하지 않고 새로 시작
       python eval.py --questions X.tsv --out-dir DIR

입력 TSV 열:  번호, 질문, 유형, 기대조각(쉼표 구분, 함정 질문은 빈칸)
출력:  results_{버전}_{YYYYMMDD_HHMM}.csv   (UTF-8-BOM, 엑셀에서 바로 열림)
       run_info_{버전}_{YYYYMMDD_HHMM}.json  실행 설정 기록
       progress_{버전}.json                  진행 중일 때만 존재 (끝나면 삭제)

중간에 멈춰도(일일 한도 소진 등) 다시 실행하면 같은 CSV 에 남은 번호만 이어서 채운다.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from google.genai import errors

import config
import gemini_api
import indexer
from llm import generate_answer
from retriever import Retriever

EVAL_DIR = config.ROOT / "eval"
COLUMNS = ["번호", "질문", "유형", "답변", "검색조각번호", "유사도", "기대조각",
           "적중_하나이상", "적중_전부", "사용모델", "판정", "오답원인"]
NA = "해당없음"


def read_questions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["기대조각"] = (r.get("기대조각") or "").strip()
    return rows


def parse_ids(text):
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def hit_marks(expected, retrieved):
    """(하나이상, 전부) 를 O / X / 해당없음 으로."""
    if not expected:
        return NA, NA
    found = set(expected) & set(retrieved)
    return ("O" if found else "X"), ("O" if found == set(expected) else "X")


def done_numbers(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return {row["번호"] for row in csv.DictReader(f)}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_or_resume(args, questions_hash, doc_hash):
    """이어서 할 실행이 있으면 그 파일들을, 없으면 새 파일 경로를 돌려준다."""
    progress_path = args.out_dir / f"progress_{config.SETTINGS_VERSION}.json"
    if progress_path.exists() and not args.new:
        p = json.loads(progress_path.read_text(encoding="utf-8"))
        same = (p["questions_hash"] == questions_hash and p["doc_hash"] == doc_hash
                and Path(p["results"]).exists())
        if same:
            print(f"이어서 실행: {Path(p['results']).name}")
            return progress_path, Path(p["results"]), Path(p["run_info"]), False
        print("이전 진행 기록이 현재 질문/문서와 달라 새로 시작합니다.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    results = args.out_dir / f"results_{config.SETTINGS_VERSION}_{stamp}.csv"
    run_info = args.out_dir / f"run_info_{config.SETTINGS_VERSION}_{stamp}.json"
    progress_path.write_text(json.dumps({
        "results": str(results), "run_info": str(run_info),
        "questions_hash": questions_hash, "doc_hash": doc_hash,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return progress_path, results, run_info, True


def write_run_info(path, args, retriever, doc_hash, total, done, started_at, finished=False):
    info = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds") if finished else None,
        "settings_version": config.SETTINGS_VERSION,
        "num_chunks": len(retriever.chunks),
        "k": config.TOP_K,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.GEMINI_LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "doc_hash": doc_hash,
        "questions_file": Path(args.questions).name,
        "questions_total": total,
        "questions_done": done,
    }
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="평가 질문 일괄 실행")
    parser.add_argument("--questions", default=EVAL_DIR / "questions.tsv", type=Path)
    parser.add_argument("--out-dir", default=EVAL_DIR, type=Path)
    parser.add_argument("--new", action="store_true", help="이어서 하지 않고 새로 시작")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    questions = read_questions(args.questions)
    retriever = Retriever()
    doc_hash = json.loads(indexer.META_PATH.read_text(encoding="utf-8"))["doc_hash"]
    progress_path, results_path, run_info_path, is_new = start_or_resume(
        args, sha256_file(args.questions), doc_hash)

    if is_new or not results_path.exists():
        with open(results_path, "w", encoding="utf-8-sig", newline="") as f:   # BOM 은 처음 한 번만
            csv.writer(f).writerow(COLUMNS)
        started_at = datetime.now().isoformat(timespec="seconds")
    else:
        started_at = json.loads(run_info_path.read_text(encoding="utf-8"))["started_at"]

    done = done_numbers(results_path)
    todo = [q for q in questions if q["번호"] not in done]
    write_run_info(run_info_path, args, retriever, doc_hash, len(questions), len(done), started_at)
    print(f"질문 {len(questions)}개 중 완료 {len(done)}개 · 남은 {len(todo)}개 · "
          f"설정 {config.SETTINGS_VERSION} (k={config.TOP_K}, {config.GEMINI_LLM_MODEL})")

    start = time.perf_counter()
    for q in todo:
        try:
            hits = retriever.search(q["질문"], k=config.TOP_K)
            answer = generate_answer(q["질문"], [h.chunk for h in hits])
        except gemini_api.DailyQuotaExceeded as e:
            print(f"\n오늘 한도 소진: 내일 python eval.py 로 이어서 실행하세요. ({e})")
            sys.exit(1)
        except errors.APIError as e:
            print(f"\nAPI 오류 {e.code} {e.status}: {e.message}\n다시 실행하면 {q['번호']}번부터 이어집니다.")
            sys.exit(1)

        retrieved = [h.chunk.id for h in hits]
        any_hit, all_hit = hit_marks(parse_ids(q["기대조각"]), retrieved)
        row = [q["번호"], q["질문"], q["유형"], answer,
               ", ".join(map(str, retrieved)),
               ", ".join(f"{h.score:.3f}" for h in hits),
               q["기대조각"], any_hit, all_hit, config.GEMINI_LLM_MODEL, "", ""]
        with open(results_path, "a", encoding="utf-8", newline="") as f:      # 한 줄씩 바로 저장
            csv.writer(f).writerow(row)
        done.add(q["번호"])
        print(f"  {q['번호']}번 완료 · 검색 {retrieved}")

    write_run_info(run_info_path, args, retriever, doc_hash, len(questions), len(done),
                   started_at, finished=True)
    progress_path.unlink(missing_ok=True)

    with open(results_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    scored = [r for r in rows if r["적중_하나이상"] != NA]
    print(f"\n완료 · {time.perf_counter() - start:.0f}초 · API 호출 {gemini_api.api_calls}회")
    print(f"적중_하나이상 O: {sum(r['적중_하나이상'] == 'O' for r in scored)}/{len(scored)} · "
          f"적중_전부 O: {sum(r['적중_전부'] == 'O' for r in scored)}/{len(scored)} "
          f"(함정 {len(rows) - len(scored)}개 제외)")
    print(f"저장: {results_path}\n      {run_info_path}")


if __name__ == "__main__":
    main()
