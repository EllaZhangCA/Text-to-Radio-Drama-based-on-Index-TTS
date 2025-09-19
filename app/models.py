from dataclasses import dataclass
from typing import Dict

@dataclass
class Segment:
    seq: int                 # global sequence number
    kind: str                # "dialog" or "narration"
    speaker: str             # speaker name or "旁白"
    text: str                # target text to synthesize
    emo_text: str            # emotion hint (prev + self + next)
    start_idx: int           # start char index in original text
    meta: Dict              # misc info (rule used, etc.)
