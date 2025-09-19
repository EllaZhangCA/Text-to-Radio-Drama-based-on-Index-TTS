from typing import List
from pydub import AudioSegment
from .config import SAMPLE_RATE

def concat_all_wavs(file_list: List[str], silence_ms: int, out_path: str) -> str:
    combined = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    silence = AudioSegment.silent(duration=max(0, int(silence_ms)), frame_rate=SAMPLE_RATE)
    for fp in file_list:
        seg = AudioSegment.from_file(fp).set_frame_rate(SAMPLE_RATE)
        combined += seg + silence
    combined.export(out_path, format="wav")
    return out_path
