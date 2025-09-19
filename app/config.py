# Global configuration for the project
import os

# Output folder for generated clips and merged audio
OUT_DIR = os.environ.get("NR_OUT_DIR", "radio_out")

# Audio settings
SAMPLE_RATE = int(os.environ.get("NR_SAMPLE_RATE", "24000"))
DEFAULT_SILENCE_MS = int(os.environ.get("NR_SILENCE_MS", "150"))

# Emotion strength for text-based emotion mode in IndexTTS2
DEFAULT_EMO_ALPHA = float(os.environ.get("NR_EMO_ALPHA", "0.6"))

# Max characters for narration chunk
MAX_NARRATION_CHARS = int(os.environ.get("NR_MAX_NARRATION_CHARS", "160"))

# Quote symbols
CN_QUOTE_L = "“"
CN_QUOTE_R = "”"

# Speech verbs used to detect speaking actions
SPEECH_VERBS = r"(说道|说|问|答|回道|喊道|叫道|笑道|低声道|冷冷地说|沉声道|叹道|轻声道)"
EMO_WINDOW_CHARS = int(os.environ.get("NR_EMO_WINDOW_CHARS", "100"))