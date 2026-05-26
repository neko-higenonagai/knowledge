import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

# --- Windows CUDA/cuDNN DLL優先探索の設定 ---
import os
import sys
if os.name == "nt":
    original_path = os.environ.get("PATH", "")
    nvidia_bins = []
    for p in sys.path:
        if not p:
            continue
        nvidia_dir = os.path.join(p, "nvidia")
        if os.path.exists(nvidia_dir):
            for sub in os.listdir(nvidia_dir):
                bin_dir = os.path.join(nvidia_dir, sub, "bin")
                if os.path.exists(bin_dir):
                    nvidia_bins.append(bin_dir)
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
    if nvidia_bins:
        os.environ["PATH"] = os.pathsep.join(nvidia_bins + original_path.split(os.pathsep))

# --- ONNX Runtime の警告ログをグローバルに抑制するパッチ (ライセンス準拠・パッケージ変更なし) ---
try:
    import onnxruntime
    _orig_session_options_init = onnxruntime.SessionOptions.__init__
    def _patched_session_options_init(self, *args, **kwargs):
        _orig_session_options_init(self, *args, **kwargs)
        self.log_severity_level = 3
    onnxruntime.SessionOptions.__init__ = _patched_session_options_init
except Exception:
    pass

VERSION = "0.0.2"

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import sqlite3
import datetime
import threading
import subprocess
import io
import argparse
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageTk

from config_ai_common import ensure_config, resolve_rag_base_path, resolve_rag_db_path
from embedding_manager import EmbeddingManager
from rag_ft_common import SudachiNounExtractor, fts_sync_single_record, embedding_sync_single_record, logical_path_parts, background_embedding_catchup

# ===========================================================================
# Configuration & Constants
# ===========================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "scripts" / "config_ai.txt"

# ===========================================================================
# Search Logic
# ===========================================================================

class IndexManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.extractor = SudachiNounExtractor()

    def ensure_db(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_texts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                date       TEXT,
                path       TEXT,
                filename   TEXT,
                page       INTEGER,
                text       TEXT,
                updated_at TEXT DEFAULT '',
                fts_synced INTEGER DEFAULT 0,
                UNIQUE(path, filename, page)
            )
        """)
        # embedding カラムの追加（マイグレーション）
        try:
            cursor.execute("ALTER TABLE ocr_texts ADD COLUMN embedding BLOB DEFAULT NULL")
        except sqlite3.OperationalError:
            pass

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS ocr_texts_fts USING fts5(
                filename UNINDEXED,
                page UNINDEXED,
                path UNINDEXED,
                category1 UNINDEXED,
                category2 UNINDEXED,
                text,
                tokenize = 'unicode61'
            )
        """)
        conn.commit()
        return conn

    def search(self, query_text, mode="index", limit=100, cat1=None, cat2=None, cat3=None):
        conn = self.ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            
            # Prepare filter criteria
            filter_sql = ""
            filter_params = []

            def get_sql_for_selection(sel_list):
                # sel_list is [ (name, is_folder), ... ]
                if not sel_list:
                    return "", []
                
                last_name, last_is_folder = sel_list[-1]
                prefix_path = "/".join([x[0] for x in sel_list[:-1]])
                
                if last_is_folder:
                    # Folder selection: match path or path prefix
                    target_path = "/".join([x[0] for x in sel_list])
                    return " AND (path = ? OR path LIKE ?)", [target_path, f"{target_path}/%"]
                else:
                    # File selection: match path and filename
                    return " AND path = ? AND filename = ?", [prefix_path, last_name]

            # Determine which levels are selected
            selected = []
            if cat1: selected.append(cat1)
            if cat2: selected.append(cat2)
            if cat3: selected.append(cat3)
            
            filter_sql, filter_params = get_sql_for_selection(selected)

            if mode == "index":
                tokens = self.extractor.extract_nouns(query_text)
                if not tokens:
                    # Sudachi でトークンが抽出できなかった場合（未知語や1文字ひらがな等）、
                    # 文字単位で分割して FTS5 の部分一致を助ける
                    tokens = list(query_text.replace(" ", ""))

                match_query = " AND ".join(['"' + t.replace('"', '""') + '"' for t in tokens])

                sql = f"""
                    SELECT t.id, t.date, t.path, t.filename, t.page, t.text, f.score
                    FROM (
                        SELECT rowid, bm25(ocr_texts_fts, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) AS score
                        FROM ocr_texts_fts
                        WHERE text MATCH ?
                    ) AS f
                    JOIN ocr_texts AS t ON t.id = f.rowid
                    WHERE 1=1
                    {filter_sql.replace("path", "t.path").replace("filename", "t.filename")}
                    ORDER BY score ASC
                    LIMIT ?
                """
                params = [match_query] + filter_params + [limit]
                rows = cursor.execute(sql, params).fetchall()
            else:
                # LIKE search - スペース区切りで AND 検索
                like_parts = query_text.split()
                if not like_parts:
                    return []
                
                like_sql = " AND ".join(["text LIKE ?" for _ in like_parts])
                sql = f"""
                    SELECT id, date, path, filename, page, text, 0.0 AS score
                    FROM ocr_texts
                    WHERE {like_sql}
                    {filter_sql}
                    ORDER BY id DESC
                    LIMIT ?
                """
                params = [f"%{p}%" for p in like_parts] + filter_params + [limit]
                rows = cursor.execute(sql, params).fetchall()
            # Convert rows to dicts while connection is still open
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            if "no such table" in str(e).lower():
                return []
            print(f"Database error: {e}")
            return []
        finally:
            conn.close()

    def get_filter_options(self, cat1=None, cat2=None):
        """
        cat1, cat2 are tuples: (name, is_folder)
        Returns: opts1, opts2, opts3 where each is list of (name, is_folder)
        """
        if not os.path.exists(self.db_path):
            return [], [], []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT path, filename FROM ocr_texts")
            rows = cursor.fetchall()
            
            all_logical_paths = []
            for path, filename in rows:
                p_parts = logical_path_parts(path)
                all_logical_paths.append(p_parts + [filename])

            def get_level_options(prefix_names):
                options = {} # name -> is_folder
                for lp in all_logical_paths:
                    if len(lp) > len(prefix_names) and lp[:len(prefix_names)] == prefix_names:
                        name = lp[len(prefix_names)]
                        is_folder = len(lp) > len(prefix_names) + 1
                        if name not in options:
                            options[name] = is_folder
                        elif is_folder:
                            options[name] = True
                return sorted(options.items())

            prefix1 = []
            opts1 = get_level_options(prefix1)
            
            opts2 = []
            if cat1 and cat1[1]: # cat1 is a folder
                prefix2 = [cat1[0]]
                opts2 = get_level_options(prefix2)
            
            opts3 = []
            if cat1 and cat1[1] and cat2 and cat2[1]: # cat1 and cat2 are folders
                prefix3 = [cat1[0], cat2[0]]
                opts3 = get_level_options(prefix3)
                
            return opts1, opts2, opts3
        finally:
            conn.close()

    def sync_fts_for_record(self, cursor, doc_id, filename, page, path, text):
        fts_sync_single_record(cursor, self.extractor, doc_id, filename, page, path, text)

    def update_record(self, doc_id, text, emb_manager=None):
        conn = self.ensure_db()
        cursor = conn.cursor()
        try:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = cursor.execute(
                "SELECT filename, page, path FROM ocr_texts WHERE id = ?", (doc_id,)
            ).fetchone()
            if not row:
                return False

            filename, page, path = row
            cursor.execute(
                "UPDATE ocr_texts SET text = ?, updated_at = ?, fts_synced = 0 WHERE id = ?",
                (text, now_str, doc_id)
            )
            self.sync_fts_for_record(cursor, doc_id, filename, page, path, text)
            
            if emb_manager and emb_manager.is_loaded():
                embedding_sync_single_record(cursor, emb_manager, doc_id, text)

            conn.commit()
            return True
        except Exception as e:
            if "no such table" in str(e).lower():
                return False
            print(f"Update error: {e}")
            return False
        finally:
            conn.close()

# ===========================================================================
# PDF Viewer Component
# ===========================================================================

class PDFViewer(ctk.CTkFrame):
    def __init__(self, master, on_page_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_page_change = on_page_change
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.toolbar = ctk.CTkFrame(self)
        self.toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        self.btn_prev = ctk.CTkButton(self.toolbar, text="前ページ", width=72, command=self.prev_page)
        self.btn_prev.pack(side="left", padx=2)
        self.page_label = ctk.CTkLabel(self.toolbar, text="ページ: - / -")
        self.page_label.pack(side="left", padx=10)
        self.btn_next = ctk.CTkButton(self.toolbar, text="次ページ", width=72, command=self.next_page)
        self.btn_next.pack(side="left", padx=2)

        self.btn_zoom_out = ctk.CTkButton(self.toolbar, text="―", width=50, command=self.zoom_out)
        self.btn_zoom_out.pack(side="left", padx=(20, 2))
        self.btn_zoom_in = ctk.CTkButton(self.toolbar, text="＋", width=50, command=self.zoom_in)
        self.btn_zoom_in.pack(side="left", padx=2)

        self.btn_rot_l = ctk.CTkButton(self.toolbar, text="左回転", width=60, command=lambda: self.rotate(-90))
        self.btn_rot_l.pack(side="left", padx=(20, 2))
        self.btn_rot_r = ctk.CTkButton(self.toolbar, text="右回転", width=60, command=lambda: self.rotate(90))
        self.btn_rot_r.pack(side="left", padx=2)
        
        self.btn_orig = ctk.CTkButton(self.toolbar, text="元の位置", width=60, command=self.go_orig_page)
        self.btn_orig.pack(side="left", padx=(20, 2))
        
        self.btn_ext = ctk.CTkButton(self.toolbar, text="外部アプリで開く", width=100, command=self.open_external, fg_color="DarkGoldenrod", hover_color="#8B6914")
        self.btn_ext.pack(side="left", padx=2)

        self.canvas = tk.Canvas(self, bg="gray40", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.v_scroll = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=1, column=1, sticky="ns")
        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.h_scroll.grid(row=2, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.doc = None
        self.current_page = 0
        self.zoom = 1.0
        self.rotation = 0
        self.tk_img = None
        self.img_id = None

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

    def load_pdf(self, path, page_idx=0):
        self.pdf_path = path
        self.original_page_idx = page_idx
        if self.doc:
            self.doc.close()
        try:
            path_obj = Path(path)
            ext = path_obj.suffix.lower()
            if ext in {".jpg", ".jpeg", ".png"}:
                # Convert image to PDF on the fly
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pdf_bytes = io.BytesIO()
                img.save(pdf_bytes, format="PDF")
                self.doc = pdfium.PdfDocument(pdf_bytes.getvalue())
            else:
                self.doc = pdfium.PdfDocument(path)
            
            self.current_page = page_idx
            self.rotation = 0
            self.zoom = 1.0
            self.show_page(fit_to_page=True)
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルを読み込めませんでした: {e}")

    def show_page(self, fit_to_page=False):
        if not self.doc: return
        if self.current_page >= len(self.doc): self.current_page = len(self.doc) - 1
        if self.current_page < 0: self.current_page = 0

        page = self.doc[self.current_page]
        if fit_to_page:
            self.canvas.update_idletasks()
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w > 1 and h > 1:
                pw, ph = page.get_size()
                if self.rotation % 180 != 0: pw, ph = ph, pw
                self.zoom = min((w * 0.9) / pw, (h * 0.9) / ph)

        img = page.render(scale=self.zoom, rotation=self.rotation).to_pil()
        self.tk_img = ImageTk.PhotoImage(img)

        if self.img_id: self.canvas.delete(self.img_id)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = max(0, (cw - img.width) // 2), max(0, (ch - img.height) // 2)
        self.img_id = self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.page_label.configure(text=f"ページ: {self.current_page + 1} / {len(self.doc)}")

    def go_orig_page(self):
        if hasattr(self, "original_page_idx"):
            self._set_page(self.original_page_idx)

    def open_external(self):
        if hasattr(self, "pdf_path") and self.pdf_path:
            try:
                page_num = self.current_page + 1
                opened = False
                
                # PDFかつページ番号がある場合はページ指定を試みる
                path_obj = Path(self.pdf_path)
                if path_obj.suffix.lower() == ".pdf":
                    opened = self._open_pdf_at_page(path_obj, page_num)
                
                if not opened:
                    os.startfile(self.pdf_path)
            except Exception as e:
                messagebox.showerror("エラー", f"ファイルを開けませんでした: {e}")

    def _open_pdf_at_page(self, pdf_path: Path, page_num: int) -> bool:
        """
        PDFをページ指定で開く。成功した場合Trueを返す。
        """
        path_str = str(pdf_path)
        
        # --- Adobe Acrobat / Reader ---
        # /A "page=N": ページ指定, /n: 新規ウィンドウ
        acrobat_candidates = [
            r"C:\Program Files\Adobe\Acrobat DC\Acrobat\Acrobat.exe",
            r"C:\Program Files (x86)\Adobe\Acrobat DC\Acrobat\Acrobat.exe",
            r"C:\Program Files\Adobe\Acrobat Reader DC\Reader\AcroRd32.exe",
            r"C:\Program Files (x86)\Adobe\Acrobat Reader DC\Reader\AcroRd32.exe",
            r"C:\Program Files\Adobe\Acrobat Reader\Reader\AcroRd32.exe",
            r"C:\Program Files (x86)\Adobe\Acrobat Reader\Reader\AcroRd32.exe",
        ]
        for acrobat in acrobat_candidates:
            if os.path.exists(acrobat):
                subprocess.Popen([acrobat, "/A", f"page={page_num}", path_str])
                return True

        # --- SumatraPDF ---
        sumatra_candidates = [
            os.path.expandvars(r"%ProgramFiles%\SumatraPDF\SumatraPDF.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\SumatraPDF\SumatraPDF.exe"),
            os.path.expandvars(r"%LocalAppData%\SumatraPDF\SumatraPDF.exe"),
            r"C:\Program Files\SumatraPDF\SumatraPDF.exe",
            r"C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe",
        ]
        for sumatra in sumatra_candidates:
            if os.path.exists(sumatra):
                subprocess.Popen([sumatra, "-page", str(page_num), path_str])
                return True

        # --- Browsers (Chrome / Edge / Firefox) ---
        # ブラウザは URL フラグメント #page=N を使用
        pdf_url = pdf_path.as_uri() + f"#page={page_num}"
        browser_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ]
        for browser in browser_candidates:
            if os.path.exists(browser):
                subprocess.Popen([browser, pdf_url])
                return True

        return False

    def _set_page(self, page_idx):
        if not self.doc: return
        if page_idx < 0: page_idx = 0
        if page_idx >= len(self.doc): page_idx = len(self.doc) - 1
        
        if self.current_page != page_idx:
            self.current_page = page_idx
            self.show_page()
            if self.on_page_change:
                self.on_page_change(self.current_page)

    def prev_page(self):
        if self.current_page > 0: self._set_page(self.current_page - 1)
    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1: self._set_page(self.current_page + 1)
    def zoom_in(self): self.zoom *= 1.2; self.show_page()
    def zoom_out(self): self.zoom /= 1.2; self.show_page()
    def rotate(self, angle): self.rotation = (self.rotation + angle) % 360; self.show_page()
    def on_press(self, e): self.canvas.scan_mark(e.x, e.y)
    def on_drag(self, e): self.canvas.scan_dragto(e.x, e.y, gain=1)
    def on_mousewheel(self, e):
        if e.state & 0x0004:
            if e.delta > 0: self.zoom_in()
            else: self.zoom_out()
        else:
            if e.delta > 0: self.prev_page()
            else: self.next_page()

# ===========================================================================
# Main Application
# ===========================================================================

class SearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"HGNN-search v{VERSION}")
        self._set_window_icon()
        self.geometry("1280x768")

        self.config = ensure_config(CONFIG_PATH)
        self.db_path = resolve_rag_db_path(self.config)
        self.base_kb_path = resolve_rag_base_path(self.config)

        self.index_mgr = IndexManager(str(self.db_path))
        self.emb_manager = EmbeddingManager(self.config.get("embedding_model", "onnx-community/harrier-oss-v1-270m-ONNX"))
        self.current_doc_id = None
        self.current_file_path = None
        self.edit_mode = False
        self.search_history = []

        self._build_ui()
        self._handle_args()

        # Embedding モデルをバックグラウンドでロードし、完了後に未処理のベクトル生成をキャッチアップ
        def load_and_catchup():
            if self.emb_manager.is_available():
                self.emb_manager.load_model()
                background_embedding_catchup(str(self.db_path), self.emb_manager)

        threading.Thread(target=load_and_catchup, daemon=True).start()

    def _set_window_icon(self):
        try:
            icon_path = Path(__file__).parent / "assets" / "icon_search.png"
            if icon_path.exists():
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                self.after(200, lambda: self.iconphoto(False, photo))
                self._icon_photo = photo
        except Exception:
            pass

    def _count_hits(self, text, query, mode):
        if not text or not query: return 0
        if mode == "like":
            tokens = query.split()
        else:
            tokens = self.index_mgr.extractor.extract_nouns(query)
            if not tokens:
                tokens = list(query.replace(" ", ""))
        text_lower = text.lower()
        count = 0
        for token in tokens:
            token_lower = token.lower()
            if not token_lower: continue
            count += text_lower.count(token_lower)
        return count

    def _handle_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", type=str)
        parser.add_argument("--page", type=int)
        parser.add_argument("--query", type=str)
        parser.add_argument("--cat1", type=str)
        parser.add_argument("--cat1_folder", type=str)
        parser.add_argument("--cat2", type=str)
        parser.add_argument("--cat2_folder", type=str)
        parser.add_argument("--cat3", type=str)
        parser.add_argument("--cat3_folder", type=str)
        args, _ = parser.parse_known_args()
        
        if args.cat1:
            self.filter_cat1_var.set(args.cat1)
            is_folder = args.cat1_folder.lower() == "true" if args.cat1_folder else True
            self._filter_data["opts1"] = [(args.cat1, is_folder)]
            self._on_cat1_change(args.cat1)
            
            if args.cat2:
                self.filter_cat2_var.set(args.cat2)
                is_folder = args.cat2_folder.lower() == "true" if args.cat2_folder else True
                self._filter_data["opts2"] = [(args.cat2, is_folder)]
                self._on_cat2_change(args.cat2)
                
                if args.cat3:
                    self.filter_cat3_var.set(args.cat3)
                    is_folder = args.cat3_folder.lower() == "true" if args.cat3_folder else False
                    self._filter_data["opts3"] = [(args.cat3, is_folder)]

        if args.file:
            p = Path(args.file)
            if p.exists():
                page = args.page if args.page is not None else 0
                # Convert 1-indexed DB page to 0-indexed for pypdfium2
                page_idx = max(0, page - 1)
                self.current_file_path = p
                self.viewer.load_pdf(str(p), page_idx=page_idx)
                # Also try to find it in DB to show text and list
                self._select_by_file_page(p, page, query=args.query)
        elif args.query:
            self.search_entry.set(args.query)
            self.after(100, self.do_search)

    def _select_by_file_page(self, path, page, query=None):
        # Search for this exact file/page in DB
        conn = self.index_mgr.ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            sql = "SELECT * FROM ocr_texts WHERE filename = ? AND page = ? LIMIT 1"
            row = cursor.execute(sql, (path.name, page)).fetchone()
            if row:
                res = dict(row)
                hit_count = "-"
                if query:
                    hit_count = self._count_hits(res.get("text", ""), query, self.search_mode_var.get())

                if self.search_mode_var.get() == "index":
                    display_stat = "-"
                else:
                    display_stat = hit_count

                # Populate treeview
                for item in self.tree.get_children(): self.tree.delete(item)
                self.results_data = {}
                raw_date = res.get("date", "-") or "-"
                display_date = raw_date[:10] if raw_date and raw_date != "-" else "-"
                item_id = self.tree.insert("", "end", values=(
                    display_stat, res.get("filename", ""), res.get("page", "-"), res.get("path", ""), display_date
                ))
                self.results_data[item_id] = res
                self.tree.selection_set(item_id)

                self.current_doc_id = res.get("id")
                self.text_preview.configure(state="normal")
                self.text_preview.delete("1.0", "end")
                self.text_preview.insert("1.0", res.get("text", ""))
                
                if query:
                    self.search_entry.set(query)
                    self.highlight_text(query)
                
                self.text_preview.configure(state="disabled")
        finally:
            conn.close()

    def _build_ui(self):
        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#2b2b2b", sashwidth=6)
        self.main_pane.pack(fill="both", expand=True)

        self.left_container = ctk.CTkFrame(self.main_pane)
        self.main_pane.add(self.left_container, width=500)

        # --- Search Bar ---
        self.search_frame = ctk.CTkFrame(self.left_container)
        self.search_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.search_entry = ctk.CTkComboBox(self.search_frame, values=[])
        self.search_entry.set("")
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self.do_search())

        self.search_button = ctk.CTkButton(self.search_frame, text="検索", width=80, command=self.do_search)
        self.search_button.pack(side="left", padx=(0, 10))

        self.search_mode_var = tk.StringVar(value="index")
        self.mode_switch = ctk.CTkSwitch(self.search_frame, text="テキスト検索",
                                         command=self.toggle_search_mode,
                                         onvalue="like", offvalue="index",
                                         variable=self.search_mode_var)
        self.mode_switch.pack(side="left")

        # --- Filters ---
        self.filter_frame = ctk.CTkFrame(self.left_container, fg_color="gray90")
        self.filter_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self._all_label = "すべて"
        self._filter_data = {"opts1": [], "opts2": [], "opts3": []}

        # Row 0: Labels
        ctk.CTkLabel(self.filter_frame, text="大分類", font=("Meiryo", 10)).grid(row=0, column=0, padx=5, sticky="w")
        ctk.CTkLabel(self.filter_frame, text="中分類", font=("Meiryo", 10)).grid(row=0, column=1, padx=5, sticky="w")
        ctk.CTkLabel(self.filter_frame, text="小分類", font=("Meiryo", 10)).grid(row=0, column=2, padx=5, sticky="w")

        # Row 1: Menus
        self.filter_cat1_var = tk.StringVar(value=self._all_label)
        self.filter_cat1 = ctk.CTkOptionMenu(self.filter_frame, variable=self.filter_cat1_var, values=[self._all_label],
                                             width=120, font=("Meiryo", 11), command=self._on_cat1_change)
        self.filter_cat1.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="ew")

        self.filter_cat2_var = tk.StringVar(value=self._all_label)
        self.filter_cat2 = ctk.CTkOptionMenu(self.filter_frame, variable=self.filter_cat2_var, values=[self._all_label],
                                             width=120, font=("Meiryo", 11), command=self._on_cat2_change)
        self.filter_cat2.grid(row=1, column=1, padx=5, pady=(0, 5), sticky="ew")

        self.filter_cat3_var = tk.StringVar(value=self._all_label)
        self.filter_cat3 = ctk.CTkOptionMenu(self.filter_frame, variable=self.filter_cat3_var, values=[self._all_label],
                                             width=120, font=("Meiryo", 11), command=self._on_cat3_change)
        self.filter_cat3.grid(row=1, column=2, padx=5, pady=(0, 5), sticky="ew")

        self.clear_filter_btn = ctk.CTkButton(self.filter_frame, text="×", width=30, command=self._clear_filters, fg_color="gray")
        self.clear_filter_btn.grid(row=1, column=3, padx=5, pady=(0, 5))

        self.filter_frame.grid_columnconfigure((0,1,2), weight=1)

        # Initialize filters
        self._load_filters()

        # --- Left Vertical PanedWindow ---
        self.left_pane = tk.PanedWindow(self.left_container, orient=tk.VERTICAL, bg="#333333", sashwidth=4)
        self.left_pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.tree_frame = ctk.CTkFrame(self.left_pane)
        self.left_pane.add(self.tree_frame, height=400)

        columns = ("hits", "filename", "page", "path", "date")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings")
        self.tree.heading("hits", text="ヒット")
        self.tree.heading("filename", text="ファイル名")
        self.tree.heading("page", text="頁")
        self.tree.heading("path", text="パス")
        self.tree.heading("date", text="日付")
        self.tree.column("hits", width=50, anchor="center")
        self.tree.column("filename", width=150)
        self.tree.column("page", width=40, anchor="center")
        self.tree.column("path", width=120)
        self.tree.column("date", width=80, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree_scroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_result_select)

        self.text_frame = ctk.CTkFrame(self.left_pane)
        self.left_pane.add(self.text_frame)

        self.text_toolbar = ctk.CTkFrame(self.text_frame, height=40)
        self.text_toolbar.pack(fill="x", side="top")

        self.edit_btn = ctk.CTkButton(self.text_toolbar, text="編集開始", width=100, command=self.toggle_edit_mode)
        self.edit_btn.pack(side="left", padx=5, pady=5)

        self.save_btn = ctk.CTkButton(self.text_toolbar, text="保存・再索引", width=100, command=self.save_text, state="disabled", fg_color="SeaGreen")
        self.save_btn.pack(side="left", padx=5, pady=5)

        self.text_preview = ctk.CTkTextbox(self.text_frame, font=("Meiryo", 12), wrap="word")
        self.text_preview.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_preview.configure(state="disabled")

        self.right_container = ctk.CTkFrame(self.main_pane)
        self.main_pane.add(self.right_container)

        self.viewer = PDFViewer(self.right_container, on_page_change=self._on_pdf_page_changed)
        self.viewer.pack(fill="both", expand=True, padx=5, pady=5)

    def _on_pdf_page_changed(self, new_page_idx):
        if not self.current_file_path: return
        db_page = new_page_idx + 1
        conn = self.index_mgr.ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            sql = "SELECT * FROM ocr_texts WHERE filename = ? AND page = ? LIMIT 1"
            row = cursor.execute(sql, (self.current_file_path.name, db_page)).fetchone()
            if row:
                res = dict(row)
                self.current_doc_id = res.get("id")
                self.text_preview.configure(state="normal")
                self.text_preview.delete("1.0", "end")
                self.text_preview.insert("1.0", res.get("text", ""))
                
                query = self.search_entry.get().strip()
                if query:
                    self.highlight_text(query)
                
                self.text_preview.configure(state="disabled")
            else:
                self.text_preview.configure(state="normal")
                self.text_preview.delete("1.0", "end")
                self.text_preview.insert("1.0", f"(P.{db_page} のテキストデータは見つかりませんでした)")
                self.text_preview.configure(state="disabled")
        finally:
            conn.close()

    def toggle_search_mode(self):
        self._render_results()

    def toggle_edit_mode(self):
        if not self.current_doc_id:
            return
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.text_preview.configure(state="normal")
            self.edit_btn.configure(text="編集キャンセル", fg_color="gray")
            self.save_btn.configure(state="normal")
        else:
            self.text_preview.configure(state="disabled")
            self.edit_btn.configure(text="編集開始", fg_color=["#3B8ED0", "#1F6AA5"])
            self.save_btn.configure(state="disabled")
            # Re-select to restore original text if cancelled
            self.on_result_select(None)

    def save_text(self):
        if not self.current_doc_id:
            return
        new_text = self.text_preview.get("1.0", "end-1c")
        
        # バックグラウンドロードがまだ完了しておらず、かつ利用可能な場合はバックグラウンドで開始
        if not self.emb_manager.is_loaded() and self.emb_manager.is_available():
            def load_fallback():
                if self.emb_manager.load_model():
                    background_embedding_catchup(str(self.db_path), self.emb_manager)
            threading.Thread(target=load_fallback, daemon=True).start()
            
        if self.index_mgr.update_record(self.current_doc_id, new_text, emb_manager=self.emb_manager):
            messagebox.showinfo("成功", "テキストを更新し、検索インデックスを再構築しました。（ベクトル情報はバックグラウンドで同期されます）")
            self.edit_mode = False
            self.text_preview.configure(state="disabled")
            self.edit_btn.configure(text="編集開始", fg_color=["#3B8ED0", "#1F6AA5"])
            self.save_btn.configure(state="disabled")
            # Update local data
            selected = self.tree.selection()
            if selected:
                item_id = selected[0]
                self.results_data[item_id]["text"] = new_text
        else:
            messagebox.showerror("エラー", "更新に失敗しました。")

    def do_search(self):
        query = self.search_entry.get().strip()
        if not query: return

        # Update history
        if query in self.search_history:
            self.search_history.remove(query)
        self.search_history.insert(0, query)
        self.search_history = self.search_history[:5]
        self.search_entry.configure(values=self.search_history)

        # Disable controls during search
        self.search_button.configure(state="disabled", text="検索中...")
        self.search_entry.configure(state="disabled")

        mode = self.search_mode_var.get()
        cat1, cat2, cat3 = self._get_active_filters()

        def _worker():
            results = self.index_mgr.search(query, mode=mode, cat1=cat1, cat2=cat2, cat3=cat3)
            self.after(0, lambda: self._on_search_done(results))

        threading.Thread(target=_worker, daemon=True).start()

    def _load_filters(self):
        opts1, opts2, opts3 = self.index_mgr.get_filter_options()
        self._apply_filter_options(opts1, opts2, opts3)

    def _apply_filter_options(self, opts1, opts2, opts3):
        self._filter_data["opts1"] = opts1
        self._filter_data["opts2"] = opts2
        self._filter_data["opts3"] = opts3

        def setup_menu(menu, var, opts):
            names = [self._all_label] + [o[0] for o in opts]
            menu.configure(values=names)
            if var.get() not in names:
                var.set(self._all_label)
            
            if menu != self.filter_cat1:
                if len(names) <= 1:
                    menu.configure(state="disabled")
                else:
                    menu.configure(state="normal")

        setup_menu(self.filter_cat1, self.filter_cat1_var, opts1)
        setup_menu(self.filter_cat2, self.filter_cat2_var, opts2)
        setup_menu(self.filter_cat3, self.filter_cat3_var, opts3)

    def _on_cat1_change(self, value):
        sel1 = next((o for o in self._filter_data["opts1"] if o[0] == value), None)
        opts1, opts2, opts3 = self.index_mgr.get_filter_options(cat1=sel1)
        self._apply_filter_options(opts1, opts2, opts3)

    def _on_cat2_change(self, value):
        sel1 = next((o for o in self._filter_data["opts1"] if o[0] == self.filter_cat1_var.get()), None)
        sel2 = next((o for o in self._filter_data["opts2"] if o[0] == value), None)
        opts1, opts2, opts3 = self.index_mgr.get_filter_options(cat1=sel1, cat2=sel2)
        self._apply_filter_options(opts1, opts2, opts3)

    def _on_cat3_change(self, value):
        pass

    def _clear_filters(self):
        self.filter_cat1_var.set(self._all_label)
        self.filter_cat2_var.set(self._all_label)
        self.filter_cat3_var.set(self._all_label)
        self._load_filters()

    def _get_active_filters(self):
        def get_sel(var, opts):
            val = var.get()
            if val == self._all_label:
                return None
            return next((o for o in opts if o[0] == val), None)

        cat1 = get_sel(self.filter_cat1_var, self._filter_data["opts1"])
        cat2 = get_sel(self.filter_cat2_var, self._filter_data["opts2"])
        cat3 = get_sel(self.filter_cat3_var, self._filter_data["opts3"])
        return cat1, cat2, cat3

    def _on_search_done(self, results):
        # Re-enable controls
        self.search_button.configure(state="normal", text="検索")
        self.search_entry.configure(state="normal")

        query = self.search_entry.get().strip()
        mode = self.search_mode_var.get()

        for res in results:
            res['hit_count'] = self._count_hits(res.get('text', ''), query, mode)

        self.last_results = results
        self._render_results()

    def _render_results(self):
        if not hasattr(self, "last_results"): return
        results = self.last_results
        search_mode = self.search_mode_var.get()

        def sort_key(x):
            try:
                p = int(x.get('page') or 0)
            except ValueError:
                p = 0
            if search_mode == "index":
                # FTS5 bm25 is negative, lower is better. Sort ascending to get most relevant first.
                return (x.get('score', 0.0), x.get('filename') or '', p)
            else:
                return (-x.get('hit_count', 0), x.get('filename') or '', p)
            
        results.sort(key=sort_key)

        # Determine dynamic scaling factor for BM25 score if diluted
        bm25_scale = 1.0
        if search_mode == "index":
            max_abs_score = max([abs(r.get("score", 0.0)) for r in results] or [0.0])
            if 0.0 < max_abs_score < 0.001:
                bm25_scale = 1000000.0

        if search_mode == "index":
            self.tree.heading("hits", text="スコア")
            self.tree.column("hits", width=60)
        else:
            self.tree.heading("hits", text="ヒット")
            self.tree.column("hits", width=50)

        for item in self.tree.get_children(): self.tree.delete(item)
        self.results_data = {}
        for res in results:
            raw_date = res.get("date", "-") or "-"
            display_date = raw_date[:10] if raw_date and raw_date != "-" else "-"
            
            if search_mode == "index":
                # Show positive representation of BM25 score (FTS5 returns negative)
                score_val = -res.get('score', 0.0) * bm25_scale
                display_stat = f"{score_val:.2f}"
            else:
                display_stat = res['hit_count']

            item_id = self.tree.insert("", "end", values=(
                display_stat, res.get("filename", ""), res.get("page", "-"), res.get("path", ""), display_date
            ))
            self.results_data[item_id] = res

        if not results:
            messagebox.showinfo("検索結果", "該当するドキュメントは見つかりませんでした。")
        else:
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.see(first_item)
            self.on_result_select(None)

    def on_result_select(self, event):
        selected = self.tree.selection()
        if not selected: return
        item_id = selected[0]
        res = self.results_data.get(item_id)
        if not res: return
        self.current_doc_id = res.get("id")
        self.text_preview.configure(state="normal")
        self.text_preview.delete("1.0", "end")
        self.text_preview.insert("1.0", res.get("text", ""))
        self.highlight_text(self.search_entry.get().strip())
        if not self.edit_mode:
            self.text_preview.configure(state="disabled")
        
        filename, path_str, page_num = res.get("filename"), res.get("path", ""), res.get("page", 0)
        full_path = self.base_kb_path / path_str / filename
        if full_path.exists():
            self.current_file_path = full_path
            try: p_idx = max(0, int(page_num) - 1)
            except: p_idx = 0
            self.viewer.load_pdf(str(full_path), page_idx=p_idx)

    def highlight_text(self, query):
        if not query: return
        self.text_preview.tag_remove("search", "1.0", "end")
        self.text_preview.tag_config("search", background="yellow", foreground="black")
        
        # Get tokens for highlighting
        if self.search_mode_var.get() == "like":
            tokens = query.split()
        else:
            tokens = self.index_mgr.extractor.extract_nouns(query)
            if not tokens:
                tokens = list(query.replace(" ", ""))
        
        for token in tokens:
            token = token.lower()
            if not token: continue
            start_pos = "1.0"
            while True:
                start_pos = self.text_preview.search(token, start_pos, stopindex="end", nocase=True)
                if not start_pos: break
                end_pos = f"{start_pos}+{len(token)}c"
                self.text_preview.tag_add("search", start_pos, end_pos)
                start_pos = end_pos

if __name__ == "__main__":
    app = SearchApp()
    app.mainloop()
