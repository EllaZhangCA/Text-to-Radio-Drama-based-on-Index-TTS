在 app/config.py 里新增这一行常量（若尚未存在）：

# Added for char-based emotion context window
EMO_WINDOW_CHARS = int(os.environ.get("NR_EMO_WINDOW_CHARS", "100"))

