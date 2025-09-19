import re
from typing import List, Tuple, Optional, Dict
import numpy as np

from .models import Segment

try:
    from .config import EMO_WINDOW_CHARS
except Exception:
    EMO_WINDOW_CHARS = 100

try:
    from .config import ROLE_SEARCH_WINDOW
except Exception:
    ROLE_SEARCH_WINDOW = 100

from .text_processing import normalize_text, split_sentences_keep_quotes

COLON_CLASS = r'[:：]'
QUOTE_OPEN_CLASS = r'[“"「『]'
QUOTE_CLOSE_CLASS = r'[”"」』]'
SENT_ENDERS = "。！？….!?;；\n"

def _find_all_quote_spans_global(text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in re.finditer(rf'{QUOTE_OPEN_CLASS}.+?{QUOTE_CLOSE_CLASS}', text)]

def _inside_any_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    for L, R in spans:
        if L <= pos < R:
            return True
    return False

def _strip_quotes(s: str) -> str:
    return re.sub(rf'{QUOTE_OPEN_CLASS}|{QUOTE_CLOSE_CLASS}', '', s)

def _emo_text(full_text: str, abs_L: int, abs_R: int, window: int) -> str:
    L = max(0, abs_L - window); R = min(len(full_text), abs_R + window)
    before = _strip_quotes(full_text[L:abs_L]).strip()
    mid    = _strip_quotes(full_text[abs_L:abs_R]).strip()
    after  = _strip_quotes(full_text[abs_R:R]).strip()
    emo = " ".join(x for x in (before, mid, after) if x)
    emo = re.sub(r'\s+', ' ', emo)
    return emo[:max(60, window * 2)]

def _nearest_role_by_chars(full_text: str, abs_L: int, names: List[str], window: int = ROLE_SEARCH_WINDOW) -> str:
    L = max(0, abs_L - window); R = min(len(full_text), abs_L + window)
    ctx = full_text[L:R]
    best_name, best_dist = "找不到", 10**12
    for name in names or []:
        if not name:
            continue
        for m in re.finditer(re.escape(name), ctx):
            hit = L + m.start()
            dist = abs(hit - abs_L)
            if dist < best_dist:
                best_dist, best_name = dist, name
    return best_name

def _union_intervals(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans: return []
    spans = sorted([(max(0, L), max(L, R)) for L, R in spans if R > L], key=lambda x: x[0])
    merged = [spans[0]]
    for L, R in spans[1:]:
        pL, pR = merged[-1]
        if L <= pR:
            merged[-1] = (pL, max(pR, R))
        else:
            merged.append((L, R))
    return merged

def _complement(total_len: int, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    spans = _union_intervals(spans)
    res, cur = [], 0
    for L, R in spans:
        if cur < L:
            res.append((cur, L))
        cur = max(cur, R)
    if cur < total_len:
        res.append((cur, total_len))
    return res

def _sentence_colon_tail_span(sent_text: str, sent_abs_start: int, quote_spans_global: List[Tuple[int, int]]) -> Optional[Tuple[int,int]]:
    m = re.search(COLON_CLASS, sent_text)
    if not m:
        return None
    colon_pos_abs = sent_abs_start + m.start()
    if _inside_any_span(colon_pos_abs, quote_spans_global):
        return None
    abs_L = sent_abs_start + m.end()
    abs_R = sent_abs_start + len(sent_text)
    tail = re.search(rf'[{SENT_ENDERS}\s]+$', sent_text[m.end():])
    if tail:
        abs_R = sent_abs_start + m.end() + tail.start()
    if abs_R <= abs_L:
        return None
    return (abs_L, abs_R)

def extract_all_segments(raw_text: str, names: List[str]) -> List[Segment]:
    """
    - 台词区间：引号内整句；或“冒号（不在引号内）后的句子尾部”。
    - 旁白：全文扣除台词区间后的连续块（不按句号切）。
    - 台词 speaker：±ROLE_SEARCH_WINDOW 字内最近名字；无则“找不到”。
    - emo：台词=窗口前后+自身；旁白=文本本身。
    """
    text = normalize_text(raw_text)

    sents = split_sentences_keep_quotes(text)  # [(sentence_text, flag, abs_start)]
    quote_spans_global = _find_all_quote_spans_global(text)

    dialog_spans_abs = []

    for sent_text, _flag, abs_start in sents:
        abs_end = abs_start + len(sent_text)
        mid = (abs_start + abs_end) // 2
        if _inside_any_span(mid, quote_spans_global):
            dialog_spans_abs.append((abs_start, abs_end, {"rule": "in_quotes_sentence", "content_mode": "sentence"}))

    for sent_text, _flag, abs_start in sents:
        span = _sentence_colon_tail_span(sent_text, abs_start, quote_spans_global)
        if span:
            L, R = span
            dialog_spans_abs.append((L, R, {"rule": "colon_tail", "content_mode": "slice"}))

    dialog_only_intervals = _union_intervals([(L, R) for (L, R, _m) in dialog_spans_abs])

    segments: List[Segment] = []

    for L, R, meta in sorted(dialog_spans_abs, key=lambda x: (x[0], x[1])):
        content = _strip_quotes(text[L:R]).strip()
        if not content:
            continue
        speaker = _nearest_role_by_chars(text, L, names, ROLE_SEARCH_WINDOW)
        emo = _emo_text(text, L, R, EMO_WINDOW_CHARS)
        segments.append(Segment(
            seq=0, kind="dialog", speaker=speaker, text=content,
            emo_text=emo, start_idx=L, meta=meta
        ))

    narration_spans = _complement(len(text), dialog_only_intervals)
    for L, R in narration_spans:
        content = text[L:R].strip()
        if not content or not re.search(r'\S', content):
            continue
        segments.append(Segment(
            seq=0, kind="narration", speaker="旁白", text=content,
            emo_text=content, start_idx=L, meta={"rule": "narration_block"}
        ))

    segments.sort(key=lambda s: s.start_idx)
    for i, seg in enumerate(segments, start=1):
        seg.seq = i
    return segments
