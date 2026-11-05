"""문서를 읽어 검색용 조각(chunk)으로 나눈다. API 호출 없음.

실행:  python chunker.py
결과:  index/chunks.json          프로그램이 읽는 조각 목록
       index/chunks_readable.md   사람이 검수하는 조각 목록

나누는 순서
  1) 제목(# ~ ####) 단위로 섹션을 나눈다.
  2) 섹션 안의 마크다운 표는 "헤더 + 데이터 행 1개"씩 따로 조각으로 만든다.
  3) 표가 아닌 글이 CHUNK_MAX_CHARS 를 넘으면
     굵은 소제목 → 목록 항목/빈 줄 → 문장 순서로 경계를 찾아 자른다.
  4) CHUNK_MIN_CHARS 보다 짧은 일반 조각은 다음(없으면 이전) 일반 조각과 합친다.
     표 조각은 합치지도, 합쳐 받지도 않는다. 병합 때문에 MAX 를 넘는 것은 허용.
"""
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")  # |:---:|:---|
HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")                              # --- 구분선
BOLD_LINE_RE = re.compile(r"^\*\*[^*].*\*\*\s*$")                               # **굵은 소제목** 한 줄
TOP_ITEM_RE = re.compile(r"^([*-]|\d+\.)\s+")                                   # 들여쓰기 없는 목록 항목
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: int
    file: str          # 원본 파일명
    path: list         # 제목 경로 (전체 제목)
    kind: str          # "text" 또는 "table"
    raw: str           # 원문 (마크다운 그대로, 화면 표시용)
    text: str          # 정제문 (서식 제거, 임베딩용)
    merged: list = field(default_factory=list)   # 합쳐 들어온 짧은 조각 설명

    @property
    def path_full(self):
        return " > ".join(self.path)

    @property
    def path_short(self):
        return " > ".join(shorten_title(t) for t in self.path)


# ---------------------------------------------------------------- 텍스트 정제
def clean_text(text):
    """임베딩용: 마크다운 서식 기호와 <br> 을 지우고 공백을 정리한다."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"<br\s*/?>", " ", line, flags=re.I)
        line = re.sub(r"^\s*>\s?", "", line)           # > 인용
        line = re.sub(r"^\s*[*-]\s+", "", line)         # * 목록 기호
        line = line.replace("*", "").replace("`", "")   # **굵게**, *기울임*, `코드`
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def shorten_title(title):
    """화면 표시용 축약 제목. 예) '4. 최종 결과물: …' → '4', 'Part 3. 방어력 분석' → 'Part 3'"""
    for pattern in (r"^(\d+(?:\.\d+)*)\.\s",    # 4. / 3.2.
                    r"^(Part \d+)",              # Part 3
                    r"^(\(\d+\))",               # (1)
                    r"^(\[[^\]]+\])"):           # [Process 1]
        m = re.match(pattern, title)
        if m:
            return m.group(1)
    return re.split(r"\s*[:(|]", title)[0].strip() or title


# ---------------------------------------------------------------- 1) 제목 단위 섹션
def split_sections(lines):
    """(제목 경로, 본문 줄 목록) 리스트를 돌려준다.

    파일에 # 제목이 하나뿐이면 그것은 '문서 제목'으로 보고 경로에서 뺀다.
    (모든 조각 앞에 같은 제목이 붙으면 구분에 도움이 안 되므로)
    문서 제목 바로 아래 서문만 경로가 [문서 제목] 이 된다.
    """
    h1_count = sum(1 for l in lines if re.match(r"^#\s", l))
    doc_title = None
    stack = []                      # [(레벨, 제목)]
    sections = []
    body = []

    def flush():
        if any(l.strip() for l in body):
            path = [t for _, t in stack] or ([doc_title] if doc_title else ["(제목 없음)"])
            sections.append((path, list(body)))
        body.clear()

    for line in lines:
        m = HEADING_RE.match(line)
        if not m:
            body.append(line)
            continue
        flush()
        level, title = len(m.group(1)), clean_text(m.group(2))
        if level == 1 and h1_count == 1:
            doc_title = title
            stack = []
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    flush()
    return sections


# ---------------------------------------------------------------- 2) 표 / 글 블록 분리
def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):            # 끝 | 이 빠진 행도 처리
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def split_blocks(body):
    """본문을 [("text", 줄들), ("table", 줄들), ...] 로 순서대로 나눈다."""
    blocks, text, i = [], [], 0
    while i < len(body):
        line = body[i]
        is_table = (line.strip().startswith("|") and i + 1 < len(body)
                    and TABLE_SEP_RE.match(body[i + 1]))
        if is_table:
            if text:
                blocks.append(("text", text))
                text = []
            table = []
            while i < len(body) and body[i].strip().startswith("|"):
                table.append(body[i])
                i += 1
            blocks.append(("table", table))
            continue
        if not HR_RE.match(line):          # --- 구분선은 버린다
            text.append(line)
        i += 1
    if any(l.strip() for l in text):
        blocks.append(("text", text))
    return blocks


def table_rows(table_lines):
    """표 → (원문, 정제문) 리스트. 행마다 헤더를 반복한다."""
    header_line, sep_line, rows = table_lines[0], table_lines[1], table_lines[2:]
    headers = [clean_text(h) for h in split_row(header_line)]
    result = []
    for row in rows:
        if not row.strip():
            continue
        cells = split_row(row)
        raw = "\n".join([header_line.rstrip(), sep_line.rstrip(), row.rstrip()])
        pairs = [f"{h}: {clean_text(c)}" for h, c in zip(headers, cells) if clean_text(c)]
        result.append((raw, "\n".join(pairs)))
    return result


# ---------------------------------------------------------------- 3) 긴 글 자르기
def clen(raw):
    return len(clean_text(raw))


def pack(pieces, max_len, sep="\n"):
    """조각들을 순서대로 이어 붙이되 max_len 을 넘기 직전에 끊는다."""
    out, cur = [], []
    for p in pieces:
        if cur and clen(sep.join(cur + [p])) > max_len:
            out.append(sep.join(cur))
            cur = []
        cur.append(p)
    if cur:
        out.append(sep.join(cur))
    return out


def make_units(lines):
    """빈 줄 / 굵은 소제목 / 들여쓰기 없는 목록 항목에서 새 단위를 시작한다.
    들여쓴 하위 항목과 이어지는 줄은 앞 단위에 붙는다."""
    units, cur = [], []
    for line in lines:
        if not line.strip():
            if cur:
                units.append(cur)
                cur = []
            continue
        if cur and (BOLD_LINE_RE.match(line) or TOP_ITEM_RE.match(line)):
            units.append(cur)
            cur = []
        cur.append(line)
    if cur:
        units.append(cur)
    return ["\n".join(u) for u in units]


def split_text(lines, max_len):
    raw = "\n".join(lines).strip("\n")
    if clen(raw) <= max_len:
        return [raw]

    # 굵은 소제목(**1. …**)을 기준으로 그룹을 만든다
    groups, cur = [], []
    for unit in make_units(lines):
        if cur and BOLD_LINE_RE.match(unit.splitlines()[0]):
            groups.append(cur)
            cur = []
        cur.append(unit)
    if cur:
        groups.append(cur)

    items = []
    for g in groups:
        g_raw = "\n".join(g)
        if clen(g_raw) <= max_len:
            items.append(g_raw)
            continue
        for unit in pack(g, max_len):              # 그룹이 길면 단위로
            if clen(unit) <= max_len:
                items.append(unit)
            else:                                   # 단위도 길면 문장으로
                items.extend(pack(SENTENCE_END_RE.split(unit), max_len, sep=" "))
    return pack(items, max_len)


# ---------------------------------------------------------------- 4) 짧은 조각 병합
def merge_into(small, target, before):
    """small 을 target 의 앞(before=True) 또는 뒤에 붙인다.
    small 의 제목 경로가 target 경로에 포함되지 않으면 small 의 제목을 본문에 남긴다."""
    raw = small.raw
    if target.path[:len(small.path)] != small.path:
        raw = f"{small.path[-1]}\n{raw}"
    target.raw = f"{raw}\n{target.raw}" if before else f"{target.raw}\n{raw}"
    target.text = clean_text(target.raw)
    preview = small.text.replace("\n", " ")[:30]
    target.merged.append(f"{small.path_short} ({len(small.text)}자, {'앞' if before else '뒤'}에 붙음): {preview}…")
    target.merged.extend(small.merged)


def merge_small(chunks, min_len):
    result, pending = [], None
    for c in chunks:
        if pending is not None:
            if c.kind == "text" and c.file == pending.file:
                merge_into(pending, c, before=True)
            else:
                result.append(pending) if not _merge_back(result, pending) else None
            pending = None
        if c.kind == "text" and len(c.text) < min_len:
            pending = c
            continue
        result.append(c)
    if pending is not None and not _merge_back(result, pending):
        result.append(pending)
    return result


def _merge_back(result, small):
    if result and result[-1].kind == "text" and result[-1].file == small.file:
        merge_into(small, result[-1], before=False)
        return True
    return False


# ---------------------------------------------------------------- 전체 실행
def build_chunks(data_dir=config.DATA_DIR):
    files = sorted(p for p in Path(data_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in config.DOC_EXTENSIONS)
    if not files:
        sys.exit(f"문서가 없습니다: {data_dir}")

    chunks = []
    for f in files:
        file_chunks = []
        lines = f.read_text(encoding="utf-8").splitlines()
        for path, body in split_sections(lines):
            for kind, block in split_blocks(body):
                if kind == "table":
                    pieces = table_rows(block)
                else:
                    pieces = [(r, clean_text(r)) for r in split_text(block, config.CHUNK_MAX_CHARS)]
                for raw, text in pieces:
                    if text:
                        file_chunks.append(Chunk(0, f.name, path, kind, raw, text))
        chunks.extend(merge_small(file_chunks, config.CHUNK_MIN_CHARS))

    for i, c in enumerate(chunks, start=1):
        c.id = i
    return chunks, files


def save(chunks, files):
    config.INDEX_DIR.mkdir(exist_ok=True)
    data = [{**asdict(c), "path_full": c.path_full, "path_short": c.path_short} for c in chunks]
    (config.INDEX_DIR / "chunks.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 조각 목록",
        "",
        f"- 원본 파일: {', '.join(f.name for f in files)}",
        f"- 총 {len(chunks)}개 · 길이는 임베딩용 정제문 기준 글자 수",
        "- 아래 내용은 임베딩에 실제로 들어가는 정제문입니다 (서식 기호 제거됨).",
        "",
    ]
    for c in chunks:
        tags = [c.kind, f"{len(c.text)}자"]
        if c.merged:
            tags.append(f"짧은 조각 {len(c.merged)}개 병합")
        lines += [
            "---",
            "",
            f"## [{c.id}] {c.path_short}",
            "",
            f"- 제목 경로: {c.path_full}",
            f"- 파일: {c.file} · " + " · ".join(tags),
        ]
        lines += [f"- 병합됨: {m}" for m in c.merged]
        lines += [
            "",
            "```text",
            c.text,
            "```",
            "",
        ]
    (config.INDEX_DIR / "chunks_readable.md").write_text("\n".join(lines), encoding="utf-8")


def report(chunks):
    lengths = [len(c.text) for c in chunks]
    print(f"조각 수: {len(chunks)}")
    print(f"길이(정제문 글자 수): 최소 {min(lengths)} / 최대 {max(lengths)} / "
          f"평균 {sum(lengths) / len(lengths):.0f}")
    print(f"표 조각 수: {sum(c.kind == 'table' for c in chunks)}")
    merged = [c for c in chunks if c.merged]
    print(f"병합: 짧은 조각 {sum(len(c.merged) for c in merged)}개가 {len(merged)}개 조각에 합쳐짐")
    for c in merged:
        for m in c.merged:
            print(f"  → [{c.id}] {c.path_short} 에 병합: {m}")
    over = [c for c in chunks if len(c.text) > config.CHUNK_MAX_CHARS]
    print(f"{config.CHUNK_MAX_CHARS}자 초과: " +
          (", ".join(f"[{c.id}] {len(c.text)}자" for c in over) if over else "없음"))
    short = [c for c in chunks if len(c.text) < config.CHUNK_MIN_CHARS]
    if short:
        print(f"{config.CHUNK_MIN_CHARS}자 미만으로 남은 조각: " +
              ", ".join(f"[{c.id}]({c.kind}, {len(c.text)}자)" for c in short))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 터미널 한글 깨짐 방지
    chunks, files = build_chunks()
    save(chunks, files)
    report(chunks)
    print(f"저장: {config.INDEX_DIR / 'chunks.json'}, {config.INDEX_DIR / 'chunks_readable.md'}")
