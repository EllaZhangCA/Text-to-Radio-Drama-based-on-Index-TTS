# app/tts_wrapper.py
import os
from typing import Optional
from .config import DEFAULT_EMO_ALPHA

ENV_CKPT = "INDEXTTS_CHECKPOINTS"  # 环境变量名：指向包含 config.yaml 的目录


class MissingIndexTTS(Exception):
    pass


def _to_bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


class TTSWrapper:
    """
    - 优先使用 GUI 传入的 checkpoints_dir；否则用环境变量 INDEXTTS_CHECKPOINTS；
      两者都没有时，回退到 D:\\index-tts\\checkpoints。
    - 只调用 indextts.infer_v2.IndexTTS2，一次性按固定签名初始化；
    - 只提供 synth()（并给出 synthesize() 别名）。
    """

    def __init__(self, checkpoints_dir: Optional[str] = None):
        try:
            from indextts.infer_v2 import IndexTTS2
        except Exception as e:
            raise MissingIndexTTS(
                "未找到 IndexTTS2，请先安装并确保可从 Python 导入 indextts.infer_v2。"
            ) from e

        # 解析 checkpoints 路径：GUI 参数 > 环境变量 > 默认
        ckpt_dir = (
            checkpoints_dir
            or os.environ.get(ENV_CKPT)
            or r"D:\index-tts\checkpoints"
        )
        ckpt_dir = os.path.abspath(ckpt_dir)
        cfg_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise RuntimeError(
                f"未找到配置文件：{cfg_path}\n"
                f"请在 GUI 顶部的“IndexTTS2 checkpoints 路径”里填入正确目录，或设置环境变量 {ENV_CKPT}。"
            )

        # 允许用环境变量覆盖这些开关（默认与原写法一致）
        use_fp16 = _to_bool(os.environ.get("INDEXTTS_USE_FP16"), True)
        use_cuda_kernel = _to_bool(os.environ.get("INDEXTTS_USE_CUDA_KERNEL"), True)
        use_deepspeed = _to_bool(os.environ.get("INDEXTTS_USE_DEEPSPEED"), True)

        # 直接按固定签名初始化
        self.tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=ckpt_dir,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
        )

    def synth(
        self,
        speaker_audio: str,
        text: str,
        out_path: str,
        emo_text: Optional[str] = None,
        emo_alpha: float = DEFAULT_EMO_ALPHA,
    ) -> str:
        """
        直接调用上游 infer()，不做多余分支。
        参数名与当前 index-tts 的 infer 保持一致：
          - spk_audio_prompt: 参考音频
          - text: 待合成文本
          - output_path: 输出路径
        可选：
          - use_emo_text / emo_text / emo_alpha
        """
        kwargs = dict(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=out_path,
            verbose=False,
        )
        if emo_text and emo_text.strip():
            kwargs.update(
                dict(use_emo_text=True, emo_text=emo_text, emo_alpha=float(emo_alpha))
            )

        self.tts.infer(**kwargs)
        return out_path

    # 兼容旧调用：提供 synthesize 别名
    def synthesize(
        self,
        text: str,
        emo_text: Optional[str],
        speaker_name: str,          # 仅为兼容签名，不参与最终 infer（index-tts 走参考音频）
        ref_audio_path: str,
        emo_alpha: float,
        out_path: str,
    ) -> str:
        return self.synth(
            speaker_audio=ref_audio_path,
            text=text,
            out_path=out_path,
            emo_text=emo_text,
            emo_alpha=emo_alpha,
        )
