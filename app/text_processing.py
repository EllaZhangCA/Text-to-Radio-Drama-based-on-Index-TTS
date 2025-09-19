\
import re
from typing import List, Tuple

# Canonical Chinese quotes
CN_L = "“"
CN_R = "”"

# All quote pairs we want to recognize
QUOTE_OPEN_SET = '“"「『'
QUOTE_CLOSE_SET = '”"」』'

def normalize_text(s: str) -> str:
    """
    Normalize mixed Chinese/English quotes and whitespace.
    - Convert 「」/『』 to Chinese style “”
    - Convert ASCII double quotes "..." into paired “...”
      (best-effort pairing by toggling)
    - Collapse excessive spaces (including fullwidth space)
    """
    # Unify East Asian quotes to Chinese quotes
    s = s.replace("「", CN_L).replace("」", CN_R)
    s = s.replace("『", CN_L).replace("』", CN_R)

    # Pair ASCII quotes into Chinese quotes (toggle method)
    out = []
    open_flag = True
    for ch in s:
        if ch == '"':
            out.append(CN_L if open_flag else CN_R)
            open_flag = not open_flag
        else:
            out.append(ch)
    s = "".join(out)

    # Unify spaces
    s = re.sub(r'[ \t\u3000]+', ' ', s)
    return s

def split_sentences_keep_quotes(s: str) -> List[Tuple[str, bool, int]]:
    """
    Sentence splitter recognizing BOTH Chinese and English enders.
    Returns: [(sentence_text, is_quote_sentence, start_index), ...]
    Enders: 。！？；… . ! ? ; and newline
    """
    res = []
    i = 0
    buf = []
    start = 0
    ENDERS = "。！？；….!?;\n"
    for ch in s:
        buf.append(ch)
        if ch in ENDERS:
            part = "".join(buf).strip()
            if part:
                is_quote = _has_any_quotes(part)
                res.append((part, is_quote, start))
            i += 1
            start = i
            buf = []
        else:
            i += 1
    if buf:
        part = "".join(buf).strip()
        if part:
            is_quote = _has_any_quotes(part)
            res.append((part, is_quote, start))
    return res

def _has_any_quotes(part: str) -> bool:
    return any(q in part for q in (CN_L, CN_R, '"', '「', '」', '『', '』'))

def extract_quote_text(part: str) -> str:
    """
    Extract text inside ANY double-quote style:
    “ ... ”  or  " ... "  or  「 ... 」  or  『 ... 』
    """
    # Try unified Chinese quotes first
    m = re.search(r'“(.+?)”', part)
    if m:
        return m.group(1).strip()
    # Try ASCII double quotes
    m = re.search(r'"(.+?)"', part)
    if m:
        return m.group(1).strip()
    # Try East Asian corner quotes
    m = re.search(r'「(.+?)」', part)
    if m:
        return m.group(1).strip()
    m = re.search(r'『(.+?)』', part)
    if m:
        return m.group(1).strip()
    return ""
