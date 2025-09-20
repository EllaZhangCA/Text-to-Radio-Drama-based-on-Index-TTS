# app/gui.py
import os, json, time
from typing import List, Any
import gradio as gr
import threading  # 串行化编辑

from .config import OUT_DIR, DEFAULT_EMO_ALPHA, DEFAULT_SILENCE_MS
from .models import Segment
from .dialogue_extraction import extract_all_segments
from .merging import concat_all_wavs
from .synthesis import batch_synthesize

# ========= 全局缓存（segments & roles） =========
SEG_CACHE: List[dict] = []   # 片段缓存
LAST_EDIT_TS = 0.0
EDIT_LOCK = threading.Lock()

ROLES_CACHE: List[List[str]] = [[""], [""]]  # 角色名单缓存（只 1 列：name）
ROLES_LOCK = threading.Lock()

# ---------------- 工具函数 ----------------
def _to_rows(maybe_df):
    if maybe_df is None:
        return []
    try:
        import pandas as pd
        if isinstance(maybe_df, pd.DataFrame):
            return pd.DataFrame(maybe_df).fillna("").astype(str).values.tolist()
    except Exception:
        pass
    if isinstance(maybe_df, list):
        rows = []
        for r in maybe_df:
            if isinstance(r, list):
                rows.append(["" if (x is None) else str(x) for x in r])
            else:
                rows.append(["" if (r is None) else str(r)])
        return rows
    return []

def _segments_table_data(segs: List[Segment]):
    table = []
    for s in segs:
        table.append([
            s.seq,
            "角色台词" if s.kind == "dialog" else "旁白",
            s.speaker,
            s.text,
            s.emo_text or ""
        ])
    return table

def _canonicalize_segments(obj: Any) -> List[dict]:
    if isinstance(obj, dict) and "segments" in obj:
        obj = obj["segments"]
    seg_dicts = []
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind not in ("dialog", "narration"):
                continue
            seg_dicts.append({
                "seq": int(item.get("seq", 0) or 0),
                "kind": kind,
                "speaker": item.get("speaker", "旁白" if kind == "narration" else "未判定"),
                "text": item.get("text", ""),
                "emo_text": item.get("emo_text", ""),
                "start_idx": int(item.get("start_idx", 0) or 0),
                "meta": item.get("meta", {}),
            })
    seg_dicts.sort(key=lambda x: (x.get("start_idx", 0), x.get("seq", 0)))
    for i, d in enumerate(seg_dicts, start=1):
        d["seq"] = i
    return seg_dicts


# ---------------- 文案（中英切换） ----------------
STRINGS = {
    "zh": {
        "title": "小说转广播剧（基于 IndexTTS2）",
        "warn_prefix": "**提示：**检测到 {folder} 中存在上次生成留下的音频文件，建议删除该文件夹里的音频后再继续使用。",
        "novel": "上传小说（.txt）",
        "emo_alpha": "情绪强度",
        "silence": "全局拼接静音 (ms)",
        "tab_roles": "角色与旁白",
        "roles_table_title": "角色名单（通过下方输入添加；默认展示两行空位）",
        "roles_table_name": "name",
        "role_input": "输入角色名",
        "add_role": "加入角色",
        "remove_role_sel": "选择要移除的角色",
        "remove_role": "移除角色",
        "upload_refs": "上传角色参考音频（可多文件）",
        "mapping_table": "参考音频绑定表（第一列路径、第二列角色名，旁白请写“旁白”或“Narrator”）",
        "narr_audio": "旁白参考音频（可选；也可在上表中写一行 name=旁白 或 Narrator）",
        "tab_extract": "抓取台词与预览",
        "extract_all": "抓取全部台词",
        "seg_table_headers": ["seq","类型","说话人","文本","情绪提示"],
        "seg_json_preview": "Segments JSON（预览区）",
        "save_edits": "保存表格修改",
        "tab_batch": "批量生成与试听",
        "gen_roles": "生成全部角色台词",
        "gen_narr": "生成全部旁白",
        "seq_input": "输入序号（seq）",
        "gen_one_role": "生成某条台词（按 seq）",
        "gen_one_narr": "生成某条旁白（按 seq）",
        "role_files": "角色音频清单",
        "narr_files": "旁白音频清单",
        "pick_file": "选择试听",
        "preview": "试听",
        "json_batch_title": "Segments JSON（自动同步，可粘贴导入，一个 JSON 自动识别旁白/角色）",
        "json_batch": "Segments JSON（批量生成区）",
        "import_json": "从 JSON 导入 Segments",
        "tab_merge": "合并输出",
        "merge": "合并为整部广播剧",
        "merge_preview": "合并试听",
        "lang": "界面语言 Language",
        "zh": "中文",
        "en": "English",
        "ckpt_path": "IndexTTS2 checkpoints 路径（包含 config.yaml、gpt.pth、s2mel.pth 等）",
    },
    "en": {
        "title": "Novel to Radio (Based on IndexTTS2)",
        "warn_prefix": "**Notice:** Found residual audio files in {folder}. It is recommended to delete them before proceeding.",
        "novel": "Upload novel (.txt)",
        "emo_alpha": "Emotion strength",
        "silence": "Global concat silence (ms)",
        "tab_roles": "Roles & Narrator",
        "roles_table_title": "Role list (add via input below; two empty rows shown by default)",
        "roles_table_name": "name",
        "role_input": "Enter role name",
        "add_role": "Add role",
        "remove_role_sel": "Pick a role to remove",
        "remove_role": "Remove role",
        "upload_refs": "Upload role reference audios (multiple)",
        "mapping_table": "Binding table (col1=path, col2=role; use '旁白' or 'Narrator' for narrator)",
        "narr_audio": "Narrator reference audio (optional; or add a row in table with name='旁白'/'Narrator')",
        "tab_extract": "Extract & Preview",
        "extract_all": "Extract all segments",
        "seg_table_headers": ["seq","Type","Speaker","Text","Emotion hint"],
        "seg_json_preview": "Segments JSON (preview)",
        "save_edits": "Save table edits",
        "tab_batch": "Batch Synthesis & Preview",
        "gen_roles": "Synthesize all dialog lines",
        "gen_narr": "Synthesize all narration",
        "seq_input": "Seq to generate",
        "gen_one_role": "Synthesize one dialog (by seq)",
        "gen_one_narr": "Synthesize one narration (by seq)",
        "role_files": "Role audio list",
        "narr_files": "Narration audio list",
        "pick_file": "Pick to preview",
        "preview": "Preview",
        "json_batch_title": "Segments JSON (auto-synced; paste here; single JSON auto-detects dialog/narration)",
        "json_batch": "Segments JSON (batch area)",
        "import_json": "Import Segments from JSON",
        "tab_merge": "Merge Output",
        "merge": "Merge into full drama",
        "merge_preview": "Merged preview",
        "lang": "Language 界面语言",
        "zh": "中文",
        "en": "English",
        "ckpt_path": "IndexTTS2 checkpoints path (folder containing config.yaml, gpt.pth, s2mel.pth, ...)",
    }
}

def t(lang, key):
    return STRINGS[lang][key]


# ---------------- 主函数 ----------------
def launch_app():
    with gr.Blocks(title=t("zh","title")) as demo:
        # 语言状态
        lang_state = gr.State("zh")

        header = gr.Markdown("## " + t("zh","title"))

        # 启动自检提示（默认隐藏）
        warning_box = gr.Markdown("", visible=False)

        def _check_residual_outdir():
            AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            folder = OUT_DIR
            has_residual = False
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    ext = os.path.splitext(name)[1].lower()
                    if ext in AUDIO_EXTS:
                        has_residual = True
                        break
            if has_residual:
                msg = t("zh","warn_prefix").format(folder=os.path.abspath(folder))
                return gr.update(value=msg, visible=True)
            else:
                return gr.update(value="", visible=False)

        with gr.Row():
            novel_file = gr.File(label=t("zh","novel"), file_types=[".txt"])
            emo_alpha = gr.Slider(0.0, 1.0, value=DEFAULT_EMO_ALPHA, step=0.05, label=t("zh","emo_alpha"))
            silence_ms = gr.Slider(0, 2000, value=DEFAULT_SILENCE_MS, step=10, label=t("zh","silence"))
            lang_sel = gr.Dropdown(choices=["中文","English"], value="中文", label=t("zh","lang"))
            default_ckpt = os.environ.get("INDEXTTS_CHECKPOINTS")
            ckpt_dir = gr.Textbox(label=t("zh","ckpt_path"), value=default_ckpt)

        # 角色 & 音频绑定
        with gr.Tab(t("zh","tab_roles")):
            roles_title = gr.Markdown("### " + t("zh","roles_table_title"))

            # 1) 角色名单展示（只读）
            roles_df = gr.Dataframe(
                headers=[t("zh","roles_table_name")], datatype=["str"],
                value=ROLES_CACHE, row_count=(2, "dynamic"),
                col_count=(1, "fixed"), interactive=False, label=""
            )

            # 2) 加入/移除 控件
            with gr.Row():
                role_name_input = gr.Textbox(label=t("zh","role_input"), placeholder="小明 / 小红 / Narrator ...")
                btn_add_role = gr.Button(t("zh","add_role"))
            with gr.Row():
                role_remove_sel = gr.Dropdown(choices=[], label=t("zh","remove_role_sel"))
                btn_remove_role = gr.Button(t("zh","remove_role"))

            upload_refs_lbl = gr.Markdown("### " + t("zh","upload_refs"))
            role_audio_files = gr.Files(label=t("zh","upload_refs"), file_count="multiple", type="filepath")
            mapping_df = gr.Dataframe(
                headers=["audio_path", "name"], datatype=["str", "str"],
                value=[], row_count=(0, "dynamic"),
                col_count=(2, "fixed"), interactive=True,
                label=t("zh","mapping_table")
            )

            narrator_audio = gr.Audio(label=t("zh","narr_audio"), type="filepath")

        # 抓取 & 预览
        with gr.Tab(t("zh","tab_extract")):
            btn_extract = gr.Button(t("zh","extract_all"))
            seg_table = gr.Dataframe(
                headers=t("zh","seg_table_headers"),
                datatype=["number", "str", "str", "str", "str"],
                row_count=(0, "dynamic"), col_count=(5, "fixed"), interactive=True
            )
            seg_json = gr.Textbox(label=t("zh","seg_json_preview"), lines=6)
            btn_save_edits = gr.Button(t("zh","save_edits"))

        # 批量生成 & 试听
        with gr.Tab(t("zh","tab_batch")):
            with gr.Row():
                btn_gen_roles = gr.Button(t("zh","gen_roles"))
                btn_gen_narr = gr.Button(t("zh","gen_narr"))
                seq_input = gr.Number(label=t("zh","seq_input"), precision=0, value=1)
                btn_gen_one_role = gr.Button(t("zh","gen_one_role"))
                btn_gen_one_narr = gr.Button(t("zh","gen_one_narr"))

            role_files_table = gr.Dataframe(
                headers=["seq", "path", "label"], datatype=["number", "str", "str"],
                value=[], row_count=(0, "dynamic"), col_count=(3, "fixed"),
                interactive=True, label=t("zh","role_files")
            )
            narr_files_table = gr.Dataframe(
                headers=["seq", "path", "label"], datatype=["number", "str", "str"],
                value=[], row_count=(0, "dynamic"), col_count=(3, "fixed"),
                interactive=True, label=t("zh","narr_files")
            )

            pick_file = gr.Dropdown(label=t("zh","pick_file"), choices=[])
            audio_preview = gr.Audio(label=t("zh","preview"), interactive=False)

            json_batch_title_md = gr.Markdown("### " + t("zh","json_batch_title"))
            segments_json_batch = gr.Textbox(label=t("zh","json_batch"), lines=8)
            import_segments_json_btn = gr.Button(t("zh","import_json"))

        # 合并
        with gr.Tab(t("zh","tab_merge")):
            btn_merge = gr.Button(t("zh","merge"))
            merged_audio = gr.Audio(label=t("zh","merge_preview"), interactive=False)

        # -------- 状态 --------
        seg_state = gr.State([])
        role_items_state = gr.State([])
        narr_items_state = gr.State([])

        # -------- 语言切换 --------
        def switch_lang(sel):
            lang = "zh" if sel == "中文" else "en"
            updates = [
                gr.update(value="## " + t(lang,"title")),                  # header
                gr.update(label=t(lang,"novel")),                          # novel_file
                gr.update(label=t(lang,"emo_alpha")),                      # emo_alpha
                gr.update(label=t(lang,"silence")),                        # silence_ms
                gr.update(label=t(lang,"lang")),                           # lang_sel
                gr.update(value="### " + t(lang,"roles_table_title")),     # roles_title
                gr.update(headers=[t(lang,"roles_table_name")]),           # roles_df headers
                gr.update(label=t(lang,"role_input"), placeholder="小明 / 小红 / Narrator ..."),  # role_name_input
                gr.update(value=t(lang,"add_role")),                       # btn_add_role
                gr.update(label=t(lang,"remove_role_sel")),                # role_remove_sel
                gr.update(value=t(lang,"remove_role")),                    # btn_remove_role
                gr.update(value="### " + t(lang,"upload_refs")),           # upload_refs_lbl
                gr.update(label=t(lang,"upload_refs")),                    # role_audio_files
                gr.update(label=t(lang,"mapping_table")),                  # mapping_df
                gr.update(label=t(lang,"narr_audio")),                     # narrator_audio
                gr.update(value=t(lang,"extract_all")),                    # btn_extract
                gr.update(headers=t(lang,"seg_table_headers")),            # seg_table headers
                gr.update(label=t(lang,"seg_json_preview")),               # seg_json
                gr.update(value=t(lang,"save_edits")),                     # btn_save_edits
                gr.update(value=t(lang,"gen_roles")),                      # btn_gen_roles
                gr.update(value=t(lang,"gen_narr")),                       # btn_gen_narr
                gr.update(label=t(lang,"seq_input")),                      # seq_input
                gr.update(value=t(lang,"gen_one_role")),                   # btn_gen_one_role
                gr.update(value=t(lang,"gen_one_narr")),                   # btn_gen_one_narr
                gr.update(label=t(lang,"role_files")),                     # role_files_table
                gr.update(label=t(lang,"narr_files")),                     # narr_files_table
                gr.update(label=t(lang,"pick_file")),                      # pick_file
                gr.update(label=t(lang,"preview")),                        # audio_preview
                gr.update(value="### " + t(lang,"json_batch_title")),      # json_batch_title_md
                gr.update(label=t(lang,"json_batch")),                     # segments_json_batch
                gr.update(value=t(lang,"import_json")),                    # import_segments_json_btn
                gr.update(value=t(lang,"merge")),                          # btn_merge
                gr.update(label=t(lang,"merge_preview")),                  # merged_audio
                gr.update(label=t(lang,"ckpt_path")),                      # ckpt path
            ]
            return [lang] + updates

        lang_sel.change(
            switch_lang, inputs=[lang_sel],
            outputs=[
                lang_state, header, novel_file, emo_alpha, silence_ms, lang_sel,
                roles_title, roles_df, role_name_input, btn_add_role, role_remove_sel, btn_remove_role,
                upload_refs_lbl, role_audio_files, mapping_df, narrator_audio,
                btn_extract, seg_table, seg_json, btn_save_edits,
                btn_gen_roles, btn_gen_narr, seq_input, btn_gen_one_role, btn_gen_one_narr,
                role_files_table, narr_files_table, pick_file, audio_preview,
                json_batch_title_md, segments_json_batch, import_segments_json_btn, btn_merge, merged_audio, ckpt_dir
            ]
        )

        # -------- 角色：加入 & 移除 --------
        def _roles_list() -> List[str]:
            return [r[0].strip() for r in ROLES_CACHE if r and r[0].strip()]

        def _refresh_roles_views():
            # 展示 DataFrame 与下拉框
            view_rows = [[n] for n in (_roles_list() or ["",""])]
            if len(view_rows) < 2:
                view_rows += [[""]] * (2 - len(view_rows))
            choices = _roles_list()
            return gr.update(value=view_rows), gr.update(choices=choices, value=None)

        def add_role(name, cur_mapping):
            name = (name or "").strip()
            if not name:
                return gr.update(), gr.update(), gr.update(value="")  # no-op
            with ROLES_LOCK:
                names = _roles_list()
                if name not in names:
                    ROLES_CACHE.append([name])
            # 显示 DataFrame 与下拉刷新
            roles_view_u, remove_sel_u = _refresh_roles_views()
            # 同步：映射表追加一行 [空路径, name]
            rows = _to_rows(cur_mapping)
            rows.append(["", name])
            return roles_view_u, remove_sel_u, gr.update(value=""), rows

        btn_add_role.click(
            add_role,
            inputs=[role_name_input, mapping_df],
            outputs=[roles_df, role_remove_sel, role_name_input, mapping_df]
        )

        def remove_role(name_to_remove, cur_mapping):
            name = (name_to_remove or "").strip()
            if not name:
                return gr.update(), gr.update(), cur_mapping
            # 从缓存删除
            with ROLES_LOCK:
                kept = []
                for r in ROLES_CACHE:
                    if r and r[0].strip() != name:
                        kept.append([r[0]])
                ROLES_CACHE.clear()
                ROLES_CACHE.extend(kept if kept else [[""],[""]])
            # 显示 DataFrame 与下拉刷新
            roles_view_u, remove_sel_u = _refresh_roles_views()
            # 同步：映射表删除所有该角色的行
            rows = [r for r in _to_rows(cur_mapping) if not (len(r)>=2 and (r[1] or "").strip()==name)]
            return roles_view_u, remove_sel_u, rows

        btn_remove_role.click(
            remove_role,
            inputs=[role_remove_sel, mapping_df],
            outputs=[roles_df, role_remove_sel, mapping_df]
        )

        # 角色参考音频 → 绑定表（仅把路径追加到第一列，第二列由“加入角色”负责填）
        def on_files_append_to_mapping(files, cur_table):
            rows = _to_rows(cur_table)
            seen = {r[0] for r in rows if r and len(r) >= 1}
            for fp in (files or []):
                if fp and fp not in seen:
                    rows.append([fp, ""])
                    seen.add(fp)
            return rows

        role_audio_files.change(on_files_append_to_mapping, inputs=[role_audio_files, mapping_df], outputs=[mapping_df])

        # 抓取片段
        def on_extract(novel, _roles_df_unused):
            """抓取时以 ROLES_CACHE 为准（不依赖 UI 的快照）。"""
            global SEG_CACHE
            if not novel:
                SEG_CACHE = []
                return [], "[]", [], "[]"
            with open(novel.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            with ROLES_LOCK:
                names = [r[0].strip() for r in ROLES_CACHE if r and r[0].strip()]
            segs = extract_all_segments(text, names)
            SEG_CACHE = [s.__dict__ for s in segs]
            js = json.dumps(SEG_CACHE, ensure_ascii=False)
            return _segments_table_data(segs), js, SEG_CACHE, js

        btn_extract.click(on_extract, inputs=[novel_file, roles_df],
                          outputs=[seg_table, seg_json, seg_state, segments_json_batch])

        # —— Segments：后台随改随存 + 手动刷新 ——
        def _merge_from_table_into_cache(table_rows: List[List[str]]):
            global SEG_CACHE, LAST_EDIT_TS
            rows = _to_rows(table_rows)
            edit_map = {}
            for r in rows:
                if not r or r[0] in ("", None):
                    continue
                try:
                    idx = int(r[0])
                except Exception:
                    continue
                edit_map[idx] = r
            with EDIT_LOCK:
                new_cache = []
                for d in (SEG_CACHE or []):
                    seq = int(d.get("seq", 0) or 0)
                    if seq in edit_map:
                        r = edit_map[seq]
                        typ = str(r[1]).strip()
                        d["kind"] = "dialog" if typ.lower() in ("角色台词","type","dialog","对话","角色","dialogue","dialog line") else "narration"
                        d["speaker"] = r[2]
                        d["text"] = r[3]
                        d["emo_text"] = r[4]
                    new_cache.append(d)
                SEG_CACHE = new_cache
                LAST_EDIT_TS = time.time()
                js = json.dumps(SEG_CACHE, ensure_ascii=False)
            return js, SEG_CACHE, js

        def on_table_edit(table):
            return _merge_from_table_into_cache(table)

        seg_table.change(on_table_edit, inputs=[seg_table],
                         outputs=[seg_json, seg_state, segments_json_batch])

        def on_save_edits(_table_unused):
            global SEG_CACHE
            with EDIT_LOCK:
                segs = [Segment(**d) for d in (SEG_CACHE or [])]
                js = json.dumps(SEG_CACHE or [], ensure_ascii=False)
                return js, (SEG_CACHE or []), _segments_table_data(segs), js

        btn_save_edits.click(on_save_edits, inputs=[seg_table],
                             outputs=[seg_json, seg_state, seg_table, segments_json_batch])

        # 其它表：不回显，避免覆盖
        mapping_df.change(lambda m: None, inputs=[mapping_df], outputs=[])
        role_files_table.change(lambda _: None, inputs=[role_files_table], outputs=[])
        narr_files_table.change(lambda _: None, inputs=[narr_files_table], outputs=[])

        def import_segments_json(js_text):
            global SEG_CACHE
            try:
                data = json.loads(js_text or "[]")
            except Exception:
                data = []
            SEG_CACHE = _canonicalize_segments(data)
            segs = [Segment(**d) for d in SEG_CACHE]
            js = json.dumps(SEG_CACHE, ensure_ascii=False)
            return js, SEG_CACHE, _segments_table_data(segs), js

        import_segments_json_btn.click(import_segments_json, inputs=[segments_json_batch],
                                       outputs=[seg_json, seg_state, seg_table, segments_json_batch])

        def build_voice_map(mapping_rows, narrator_fp):
            vm = {}
            for r in _to_rows(mapping_rows):
                if r and len(r) >= 2 and r[0] and r[1]:
                    vm[r[1].strip()] = r[0]
            if narrator_fp and "旁白" not in vm:
                vm["旁白"] = narrator_fp
            if "旁白" in vm and "找不到" not in vm:
                vm["找不到"] = vm["旁白"]
            if "Narrator" in vm and "旁白" not in vm:
                vm["旁白"] = vm["Narrator"]
            if "Narrator" not in vm and "旁白" in vm:
                vm["Narrator"] = vm["旁白"]
            return vm

        # --- 批量生成（全部） ---
        def gen_roles(mapping_rows, narrator_fp, emo_alpha, _segs_ignored, ckpt_dir):
            seg_objs = [Segment(**d) for d in (SEG_CACHE or []) if d["kind"] == "dialog"]
            if not seg_objs:
                return [], [], gr.update(choices=[])
            outs = batch_synthesize(seg_objs, build_voice_map(mapping_rows, narrator_fp), emo_alpha, OUT_DIR, ckpt_dir)
            return [[seq, path, label] for (path, seq, label) in outs], outs, gr.update(choices=[lab for _, _, lab in outs])


        def gen_narr(mapping_rows, narrator_fp, emo_alpha, _segs_ignored, ckpt_dir):
            seg_objs = [Segment(**d) for d in (SEG_CACHE or []) if d["kind"] == "narration"]
            if not seg_objs:
                return [], [], gr.update(choices=[])
            outs = batch_synthesize(seg_objs, build_voice_map(mapping_rows, narrator_fp), emo_alpha, OUT_DIR, ckpt_dir)
            return [[seq, path, label] for (path, seq, label) in outs], outs, gr.update(choices=[lab for _, _, lab in outs])

        btn_gen_roles.click(gen_roles, inputs=[mapping_df, narrator_audio, emo_alpha, seg_state, ckpt_dir],
                            outputs=[role_files_table, role_items_state, pick_file])
        btn_gen_narr.click(gen_narr, inputs=[mapping_df, narrator_audio, emo_alpha, seg_state, ckpt_dir],
                           outputs=[narr_files_table, narr_items_state, pick_file])

        # --- 单条生成（按 seq） ---
        def gen_one(mapping_rows, narrator_fp, emo_alpha, _segs_ignored, seq, kind, ckpt_dir):
            try:
                seq = int(seq)
            except Exception:
                return [], [], gr.update(choices=[])
            targets = [Segment(**d) for d in (SEG_CACHE or []) if d["seq"] == seq and d["kind"] == kind]
            if not targets:
                return [], [], gr.update(choices=[])
            outs = batch_synthesize(targets, build_voice_map(mapping_rows, narrator_fp), emo_alpha, OUT_DIR, ckpt_dir)
            table_rows = [[s, p, l] for (p, s, l) in outs]
            choices = [l for _, _, l in outs]
            return table_rows, outs, gr.update(choices=choices)

        btn_gen_one_role.click(
            gen_one,
            inputs=[mapping_df, narrator_audio, emo_alpha, seg_state, seq_input, gr.State("dialog"), ckpt_dir],
            outputs=[role_files_table, role_items_state, pick_file]
        )
        btn_gen_one_narr.click(
            gen_one,
            inputs=[mapping_df, narrator_audio, emo_alpha, seg_state, seq_input, gr.State("narration"), ckpt_dir],
            outputs=[narr_files_table, narr_items_state, pick_file]
        )

        def on_pick(pick, role_items, narr_items):
            all_items = (role_items or []) + (narr_items or [])
            mapping = {label: fp for fp, _, label in all_items}
            return mapping.get(pick)

        pick_file.change(on_pick, inputs=[pick_file, role_items_state, narr_items_state],
                         outputs=[audio_preview])

        def on_merge(role_table, narr_table, silence):
            items = [(int(r[0]), r[1]) for r in _to_rows(role_table) + _to_rows(narr_table) if r and r[0] and r[1]]
            items.sort(key=lambda x: x[0])
            files = [p for _, p in items]
            if not files:
                return None
            os.makedirs(OUT_DIR, exist_ok=True)
            outfp = os.path.join(OUT_DIR, "final_merged.wav")
            concat_all_wavs(files, silence_ms=int(silence), out_path=outfp)
            return outfp

        btn_merge.click(on_merge, inputs=[role_files_table, narr_files_table, silence_ms],
                        outputs=[merged_audio])

        # 启动时的残留检测（必须在 Blocks 内）
        demo.load(_check_residual_outdir, outputs=[warning_box])

    # 兼容不同 Gradio 版本的 queue 签名
    try:
        demo = demo.queue(concurrency_count=1, max_size=64)
    except TypeError:
        try:
            demo = demo.queue(max_size=64)
        except TypeError:
            demo.queue()

    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
