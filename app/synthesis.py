# app/synthesis.py
import os
from typing import List, Tuple, Dict

from .models import Segment
from .tts_wrapper import TTSWrapper


def _label_for(seg: Segment) -> str:
    """
    统一的文件命名：{seq:05d}_{kind}_{speaker_prefix}_{hash4}
    - 防止同名覆盖
    - 便于后续合并按 seq 排序
    """
    import re, hashlib
    spk = (seg.speaker or "").strip() or ("旁白" if seg.kind == "narration" else "未命名")
    spk_clean = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", spk)[:16] or "spk"
    h = hashlib.md5((seg.text or "").encode("utf-8", errors="ignore")).hexdigest()[:4]
    kind_tag = "dlg" if seg.kind == "dialog" else "nar"
    return f"{int(seg.seq):05d}_{kind_tag}_{spk_clean}_{h}"


def _preflight_checks(segments: List[Segment], voice_map: Dict[str, str], human_kind_name: str):
    """
    合成前的检查：
    1) Segments 非空
    2) 每条都能映射到参考音频（角色台词=>seg.speaker；旁白=>固定 '旁白' 或 'Narrator'）
    """
    if not segments:
        raise ValueError(f"{human_kind_name} 空列表，无法生成。")

    missing = set()
    for seg in segments:
        if seg.kind == "dialog":
            key = (seg.speaker or "").strip() or "找不到"
        else:
            # 旁白的键统一成“旁白”，build_voice_map 内部也会做中英互补
            key = "旁白"

        # voice_map 里必须有
        if not voice_map.get(key):
            missing.add(key)

    if missing:
        # 旁白如果没绑定，也提示
        raise ValueError(
            f"{human_kind_name} 生成前置检查失败：以下说话人未在参考音频映射中绑定：{sorted(list(missing))}\n"
            f"请在“角色与旁白”页把这些名字与参考音频绑定（旁白需要键名“旁白”或“Narrator”）。"
        )


def batch_synthesize(
    segments: List[Segment],
    voice_map: Dict[str, str],
    emo_alpha: float,
    out_dir: str,
    checkpoints_dir: str | None = None,  # ← 新增参数，允许从 GUI 传入
) -> List[Tuple[str, int, str]]:
    """
    批量合成：
    - segments: 片段对象列表
    - voice_map: 角色名/旁白名 -> 参考音频路径
    - emo_alpha: 情绪强度（传递给上游）
    - out_dir: 输出目录
    - checkpoints_dir: IndexTTS2 的 checkpoints 路径（含 config.yaml 等），可为 None（走默认）
    返回：[(out_path, seq, label), ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    kind = segments[0].kind if segments else "dialog"
    _preflight_checks(segments, voice_map, "角色台词" if kind == "dialog" else "旁白")

    # —— 用用户给的 checkpoints_dir 初始化 —— #
    tts = TTSWrapper(checkpoints_dir=checkpoints_dir)

    results: List[Tuple[str, int, str]] = []
    for seg in segments:
        label = _label_for(seg)
        out_path = os.path.join(out_dir, f"{label}.wav")

        # 选择映射键：台词用说话人；旁白固定用“旁白”（build_voice_map 已经做过中英互补）
        speaker_key = seg.speaker.strip() if seg.kind == "dialog" else "旁白"
        if not speaker_key:
            speaker_key = "找不到"  # 若台词没识别到说话人

        ref_audio = voice_map.get(speaker_key)
        if not ref_audio:
            # 再尝试旁白兜底
            if seg.kind == "narration":
                ref_audio = voice_map.get("Narrator") or voice_map.get("旁白")
            elif speaker_key == "找不到":
                # 如果“找不到”也没有，就试试旁白
                ref_audio = voice_map.get("旁白") or voice_map.get("Narrator")

        # 到这里如果仍然没有 ref_audio，说明用户没在 GUI 里绑定，直接抛错更直观
        if not ref_audio:
            raise ValueError(
                f"未找到参考音频：'{speaker_key}'。请在“角色与旁白”页绑定参考音频后再试。"
            )

        # 统一调用 TTS
        tts.synthesize(
            text=seg.text,
            emo_text=seg.emo_text or seg.text,
            speaker_name=speaker_key,
            ref_audio_path=ref_audio,
            emo_alpha=float(emo_alpha),
            out_path=out_path,
        )
        results.append((out_path, int(seg.seq), label))

    return results
