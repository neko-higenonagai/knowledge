VERSION = "0.0.1"

import os
import sqlite3
import datetime
import shutil
import sys
import subprocess
import tempfile
import threading
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
import queue
import json
import io

try:
    import numpy as np
    import ocr
    import xml.etree.ElementTree as ET
    from ndl_parser import convert_to_xml_string3
    from reading_order.xy_cut.eval import eval_xml
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

import customtkinter as ctk
import pypdfium2 as pdfium
from PIL import Image, ImageTk

from config_ai_common import ensure_config, resolve_rag_base_path, resolve_rag_db_path, save_config
from rag_ft_common import SudachiNounExtractor, fts_sync_single_record

# サポートする拡張子
SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

# UI Colors
COLOR_HAS_DATA = ("#1f538d", "#97c1e7") # SteelBlue
COLOR_IN_PROGRESS = "orange"
COLOR_OUT_OF_RANGE = "#CD5C5C"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config_ai.txt"
CONFIG = ensure_config(CONFIG_PATH)

BASE_DIR = Path(__file__).parent.parent
DB_PATH = resolve_rag_db_path(CONFIG)
KB_DIR = resolve_rag_base_path(CONFIG)

PYTHON_EXE = BASE_DIR / "python" / "python.exe"

if not PYTHON_EXE.exists():
    PYTHON_EXE = sys.executable


# ===========================================================================
# OCR Logic Layer
# ===========================================================================

class OCRManager:
    def __init__(self, python_exe=None):
        self.detector = None
        self.recognizer = None
        self.recognizer30 = None
        self.recognizer50 = None
        self.python_exe = python_exe or sys.executable
        self._lock = threading.Lock()

    def is_loaded(self):
        return self.detector is not None

    def _get_model_paths(self):
        try:
            import ocr as _ocr
            base_dir = Path(_ocr.__file__).parent
        except Exception:
            base_dir = Path(sys.executable).parent / "Lib" / "site-packages"
        return {
            "det_weights": str(base_dir / "model" / "deim-s-1024x1024.onnx"),
            "det_classes": str(base_dir / "config" / "ndl.yaml"),
            "rec_weights": str(base_dir / "model" / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"),
            "rec_weights30": str(base_dir / "model" / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"),
            "rec_weights50": str(base_dir / "model" / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"),
            "rec_classes": str(base_dir / "config" / "NDLmoji.yaml"),
        }

    def ensure_models(self):
        if self.detector is not None:
            return True
        if not OCR_AVAILABLE:
            return False
        with self._lock:
            if self.detector is not None:
                return True
            try:
                paths = self._get_model_paths()
                args = argparse.Namespace(
                    det_weights=paths["det_weights"],
                    det_classes=paths["det_classes"],
                    det_score_threshold=0.2,
                    det_conf_threshold=0.25,
                    det_iou_threshold=0.2,
                    rec_weights=paths["rec_weights"],
                    rec_weights30=paths["rec_weights30"],
                    rec_weights50=paths["rec_weights50"],
                    rec_classes=paths["rec_classes"],
                    device="cpu"
                )
                self.detector = ocr.get_detector(args=args)
                self.recognizer = ocr.get_recognizer(args=args)
                self.recognizer30 = ocr.get_recognizer(args=args, weights_path=args.rec_weights30)
                self.recognizer50 = ocr.get_recognizer(args=args, weights_path=args.rec_weights50)
                return True
            except Exception as e:
                print(f"OCR Model Load Error: {e}")
                return False

    def run_ocr(self, pil_img):
        """PIL Image からテキストを抽出する。モデル未ロード時は subprocess にフォールバック。"""
        if not self.ensure_models():
            return ""
        try:
            npimg = np.array(pil_img.convert("RGB"))
            img_h, img_w = npimg.shape[:2]
            imgname = "temp_page"
            with tempfile.TemporaryDirectory() as tmp_dir:
                detections, classeslist = ocr.process_detector(
                    detector=self.detector, inputname=imgname, npimage=npimg,
                    outputpath=tmp_dir, issaveimg=False
                )
                resultobj = [{}, {i: [] for i in range(17)}]
                resultobj[0][0] = []
                for det in detections:
                    xmin, ymin, xmax, ymax = det["box"]
                    if det["class_index"] == 0:
                        resultobj[0][0].append([xmin, ymin, xmax, ymax])
                    resultobj[1][det["class_index"]].append(
                        [xmin, ymin, xmax, ymax, det["confidence"], det["pred_char_count"]]
                    )
                xmlstr = f"<OCRDATASET>{convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj)}</OCRDATASET>"
                root = ET.fromstring(xmlstr)
                eval_xml(root, logger=None)
                lines = []
                for lineobj in root.findall(".//LINE"):
                    x, y, w, h = int(lineobj.get("X")), int(lineobj.get("Y")), int(lineobj.get("WIDTH")), int(lineobj.get("HEIGHT"))
                    lineimg = npimg[y:y+h, x:x+w, :]
                    if lineimg.size == 0:
                        continue
                    char_count = int(lineobj.get("CHAR_COUNT", "0"))
                    if char_count <= 30:
                        rec = self.recognizer30
                    elif char_count <= 50:
                        rec = self.recognizer50
                    else:
                        rec = self.recognizer
                    result = rec.read(lineimg)
                    if result:
                        lines.append(result)
                return "\n".join(lines)
        except Exception as e:
            print(f"OCR Inference Error: {e}")
            return ""


# ===========================================================================
# Logic Layer (hgnn-ingestor)
# ===========================================================================

class PDFProcessor:
    def __init__(self, db_path, python_exe=None):
        self.db_path = Path(db_path)
        self.python_exe = python_exe or sys.executable
        self.stop_requested = False
        self.extractor = SudachiNounExtractor()
        self.ocr_mgr = OCRManager(self.python_exe)

    def ensure_db(self, force_new=False):
        db_dir = self.db_path.parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True)

        if force_new and self.db_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_dir / f"knowledge_backup_{timestamp}.db"
            print(f"Backing up existing DB to {backup_path}")
            shutil.move(str(self.db_path), str(backup_path))

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

        # 既存DBへのカラム追加（マイグレーション）
        # 重複カラム・既存オブジェクトは sqlite3.OperationalError を無視する
        for migration_sql in [
            "ALTER TABLE ocr_texts ADD COLUMN updated_at TEXT DEFAULT ''",
            "ALTER TABLE ocr_texts ADD COLUMN fts_synced INTEGER DEFAULT 0",
            """
            CREATE TABLE IF NOT EXISTS rag_files (
                path TEXT,
                filename TEXT,
                mtime REAL,
                size INTEGER,
                UNIQUE(path, filename)
            )
            """,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS ocr_texts_fts USING fts5(
                filename UNINDEXED,
                page UNINDEXED,
                path UNINDEXED,
                category1 UNINDEXED,
                category2 UNINDEXED,
                text,
                tokenize = 'unicode61'
            )
            """
        ]:
            try:
                cursor.execute(migration_sql)
            except sqlite3.OperationalError:
                pass
            except sqlite3.Error as exc:
                print(f"Unexpected DB migration error: {exc!s}")

        conn.commit()
        return conn

    def sync_fts_for_record(self, cursor, doc_id, filename, page, path, text):
        """1件のレコードをFTSインデックスに同期する。"""
        fts_sync_single_record(cursor, self.extractor, doc_id, filename, page, path, text)

    def get_processed_files(self, conn):
        """DBに既に存在する(path, filename)のセットを返す。"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT path, filename FROM ocr_texts")
            return {(row[0], row[1]) for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return set()


    def run_ocr(self, image_input):
        """PIL Image またはファイルパスからテキストを抽出する。"""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        else:
            img = image_input
        return self.ocr_mgr.run_ocr(img)

    def check_existing_record(self, conn, cat_path, filename, page):
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM ocr_texts WHERE path=? AND filename=? AND page=?",
                (cat_path, filename, page),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    def update_record(self, conn, cat_path, filename, page, text):
        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO ocr_texts (date, path, filename, page, text, updated_at, fts_synced)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(path, filename, page) DO UPDATE SET
                text=excluded.text,
                date=excluded.date,
                updated_at=excluded.updated_at,
                fts_synced=0
            """,
            (now_str, cat_path, filename, page, text, now_str),
        )
        # Get ID for FTS sync
        row = cursor.execute(
            "SELECT id FROM ocr_texts WHERE path=? AND filename=? AND page=?",
            (cat_path, filename, page)
        ).fetchone()

        if row:
            doc_id = row[0]
            self.sync_fts_for_record(cursor, doc_id, filename, page, cat_path, text)
        conn.commit()

    def delete_record(self, conn, cat_path, filename, page):
        """指定した1ページ分のレコードをDBとFTSから削除する。"""
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT id FROM ocr_texts WHERE path=? AND filename=? AND page=?",
            (cat_path, filename, page)
        ).fetchone()

        if row:
            doc_id = row[0]
            cursor.execute("DELETE FROM ocr_texts_fts WHERE rowid = ?", (doc_id,))
            cursor.execute("DELETE FROM ocr_texts WHERE id = ?", (doc_id,))
            conn.commit()
            return True
        return False

    def get_cat_path(self, rel_path, base_name):
        """
        Target: Root, Tier 1, Tier 2, etc.
        Input rel_path must start with base_name.
        Example: kb/T1/T2/file.pdf -> "T1/T2"
        """
        parts = rel_path.parts
        if not parts:
            return None
        
        # Case-insensitive check for base_name
        if parts[0].lower() != base_name.lower():
            # デバッグログ: なぜスキップされたかを表示
            print(f"Debug: Skip {rel_path} (Base name mismatch: {parts[0]} != {base_name})")
            return None
            
        # Depth limit: 2 levels from base folder (total parts <= 4: base_name, t1, t2, filename)
        if len(parts) > 4:
            # print(f"Debug: Skip {rel_path} (Too deep: {len(parts)} parts)")
            return None
        if len(parts) < 2:
            return None
            
        cat_parts = parts[1:-1]
        return "/".join(cat_parts)


    def is_image_file(self, filepath):
        """画像ファイル（JPEG/PNG）かどうかを判定する。"""
        return Path(filepath).suffix.lower() in {".jpg", ".jpeg", ".png"}

    def process_file(self, file_path, rel_path, base_name, conn, status_callback=None, overwrite_callback=None):
        """PDFまたは画像ファイルを処理する統合エントリポイント。"""
        if self.stop_requested:
            return False, False

        cat_path = self.get_cat_path(rel_path, base_name)
        if cat_path is None:
            if status_callback:
                status_callback(f"Skipping: {file_path.name} (Out of range)")
            return True, True

        filename = file_path.name
        mtime = os.path.getmtime(file_path)
        size = os.path.getsize(file_path)

        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT mtime, size FROM rag_files WHERE path = ? AND filename = ?",
            (cat_path, filename)
        ).fetchone()

        if row and row[0] == mtime and row[1] == size:
            if status_callback:
                status_callback(f"  Skipped (No changes): {filename}")
            
            # Migration/Consistency check: Ensure index exists for this file
            fts_count = cursor.execute(
                "SELECT COUNT(*) FROM ocr_texts_fts WHERE rowid IN (SELECT id FROM ocr_texts WHERE path=? AND filename=?)",
                (cat_path, filename)
            ).fetchone()[0]
            
            if fts_count == 0:
                if status_callback:
                    status_callback(f"  Indexing existing records: {filename}")
                rows = cursor.execute(
                    "SELECT id, page, text FROM ocr_texts WHERE path=? AND filename=?",
                    (cat_path, filename)
                ).fetchall()
                for doc_id, page, text in rows:
                    self.sync_fts_for_record(cursor, doc_id, filename, page, cat_path, text)
                conn.commit()

            # Ensure fts_synced is not -1 for deletion tracking
            cursor.execute(
                "UPDATE ocr_texts SET fts_synced = 1 WHERE path = ? AND filename = ? AND fts_synced = -1",
                (cat_path, filename)
            )
            conn.commit()
            return True, True

        if self.is_image_file(file_path):
            success = self.process_image(file_path, rel_path, base_name, conn, status_callback, overwrite_callback)
        else:
            success = self.process_pdf(file_path, rel_path, base_name, conn, status_callback, overwrite_callback)

        if success:
            cursor.execute(
                "INSERT INTO rag_files (path, filename, mtime, size) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path, filename) DO UPDATE SET mtime=excluded.mtime, size=excluded.size",
                (cat_path, filename, mtime, size)
            )
            conn.commit()
        return success, False

    def process_image(self, img_path, rel_path, base_name, conn, status_callback=None, overwrite_callback=None):
        """画像ファイル（JPEG/PNG）を1ページとしてOCR処理する。"""
        if self.stop_requested:
            return False

        cat_path = self.get_cat_path(rel_path, base_name)
        if cat_path is None:
            if status_callback:
                status_callback(f"Skipping: {img_path.name} (Out of range)")
            return True

        if status_callback:
            status_callback(f"Processing: {img_path.name}")
        else:
            print(f"Processing: {img_path}")

        cursor = conn.cursor()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = Path(img_path).name
        page_num = 1

        if status_callback:
            status_callback(("page_start", page_num))

        existing_text = self.check_existing_record(conn, cat_path, filename, page_num)
        if existing_text is not None:
            if overwrite_callback:
                if not overwrite_callback(filename, page_num):
                    if status_callback:
                        status_callback(f"  Page {page_num}: Skipped (Exists)")
                    # 削除候補から外す
                    cursor.execute("UPDATE ocr_texts SET fts_synced = 1 WHERE path=? AND filename=? AND page=? AND fts_synced = -1", (cat_path, filename, page_num))
                    conn.commit()
                    return True
            else:
                print(f"  Page {page_num}: Skipped (Exists)")
                # 削除候補から外す
                cursor.execute("UPDATE ocr_texts SET fts_synced = 1 WHERE path=? AND filename=? AND page=? AND fts_synced = -1", (cat_path, filename, page_num))
                conn.commit()
                return True

        if status_callback:
            status_callback(f"  Page {page_num}: Image file, performing OCR...")
        else:
            print(f"  Page {page_num}: Image file, performing OCR...")

        text = self.run_ocr(str(img_path))

        cursor.execute(
            """
            INSERT INTO ocr_texts (date, path, filename, page, text, updated_at, fts_synced)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(path, filename, page) DO UPDATE SET
                text=excluded.text,
                date=excluded.date,
                updated_at=excluded.updated_at,
                fts_synced=0
            """,
            (now_str, cat_path, filename, page_num, text, now_str),
        )
        # Get ID for FTS sync
        doc_id = cursor.lastrowid
        if not doc_id: # ON CONFLICT DO UPDATE case
            doc_id = cursor.execute(
                "SELECT id FROM ocr_texts WHERE path=? AND filename=? AND page=?",
                (cat_path, filename, page_num)
            ).fetchone()[0]
        
        self.sync_fts_for_record(cursor, doc_id, filename, page_num, cat_path, text)

        if status_callback:
            status_callback(("text_update", page_num, text))

        conn.commit()
        return True

    def process_pdf(self, pdf_path, rel_path, base_name, conn, status_callback=None, overwrite_callback=None):
        if self.stop_requested:
            return False

        cat_path = self.get_cat_path(rel_path, base_name)
        if cat_path is None:
            if status_callback:
                status_callback(f"Skipping: {pdf_path.name} (Out of range)")
            return True

        filename = pdf_path.name

        if status_callback:
            status_callback(f"Processing: {pdf_path.name}")
        else:
            print(f"Processing: {pdf_path}")


        try:
            doc = pdfium.PdfDocument(pdf_path)
        except Exception as e:
            if status_callback:
                status_callback(f"  Error opening PDF: {e}")
            return False

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.cursor()

        for i in range(len(doc)):
            if self.stop_requested:
                doc.close()
                return False

            page_num = i + 1

            if status_callback:
                status_callback(("page_start", page_num))

            existing_text = self.check_existing_record(conn, cat_path, filename, page_num)
            if existing_text is not None:
                if overwrite_callback:
                    if not overwrite_callback(filename, page_num):
                        if status_callback:
                            status_callback(f"  Page {page_num}: Skipped (Exists)")
                        # 削除候補から外す
                        cursor.execute("UPDATE ocr_texts SET fts_synced = 1 WHERE path=? AND filename=? AND page=? AND fts_synced = -1", (cat_path, filename, page_num))
                        continue
                else:
                    print(f"  Page {page_num}: Skipped (Exists)")
                    # 削除候補から外す
                    cursor.execute("UPDATE ocr_texts SET fts_synced = 1 WHERE path=? AND filename=? AND page=? AND fts_synced = -1", (cat_path, filename, page_num))
                    continue

            page = doc[i]
            text = page.get_textpage().get_text_bounded().strip()

            if not text:
                if status_callback:
                    status_callback(f"  Page {page_num}: No text found, performing OCR...")
                else:
                    print(f"  Page {page_num}: No text found, performing OCR...")

                img = page.render(scale=2.0).to_pil()
                text = self.run_ocr(img)
            else:
                if status_callback:
                    status_callback(f"  Page {page_num}: Text extracted directly.")
                else:
                    print(f"  Page {page_num}: Text extracted directly.")

            cursor.execute(
                """
                INSERT INTO ocr_texts (date, path, filename, page, text, updated_at, fts_synced)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(path, filename, page) DO UPDATE SET
                    text=excluded.text,
                    date=excluded.date,
                    updated_at=excluded.updated_at,
                    fts_synced=0
                """,
                (now_str, cat_path, filename, page_num, text, now_str),
            )
            # Get ID for FTS sync
            doc_id = cursor.lastrowid
            if not doc_id: # ON CONFLICT DO UPDATE case
                doc_id = cursor.execute(
                    "SELECT id FROM ocr_texts WHERE path=? AND filename=? AND page=?",
                    (cat_path, filename, page_num)
                ).fetchone()[0]
            
            self.sync_fts_for_record(cursor, doc_id, filename, page_num, cat_path, text)

            if status_callback:
                status_callback(("text_update", page_num, text))


        # Cleanup orphaned pages if the new PDF has fewer pages
        current_page_count = len(doc)
        cursor.execute(
            "DELETE FROM ocr_texts_fts WHERE rowid IN (SELECT id FROM ocr_texts WHERE path=? AND filename=? AND page > ?)",
            (cat_path, filename, current_page_count)
        )
        cursor.execute(
            "DELETE FROM ocr_texts WHERE path=? AND filename=? AND page > ?",
            (cat_path, filename, current_page_count)
        )

        conn.commit()
        doc.close()
        return True


# ===========================================================================
# GUI Layer (hgnn-ingestor)
# ===========================================================================

class PDFViewer(ctk.CTkFrame):
    def __init__(self, master, on_page_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_page_change = on_page_change

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Toolbar
        self.toolbar = ctk.CTkFrame(self)
        self.toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        self.btn_prev = ctk.CTkButton(self.toolbar, text="前ページ", width=72, command=self.prev_page)
        self.btn_prev.pack(side="left", padx=2)
        self.btn_next = ctk.CTkButton(self.toolbar, text="次ページ", width=72, command=self.next_page)
        self.btn_next.pack(side="left", padx=2)

        self.page_label = ctk.CTkLabel(self.toolbar, text="ページ: - / -")
        self.page_label.pack(side="left", padx=10)

        self.btn_zoom_out = ctk.CTkButton(self.toolbar, text="- 縮小", width=80, command=self.zoom_out)
        self.btn_zoom_out.pack(side="left", padx=(20, 2))
        self.btn_zoom_in = ctk.CTkButton(self.toolbar, text="+ 拡大", width=80, command=self.zoom_in)
        self.btn_zoom_in.pack(side="left", padx=2)

        self.btn_open_ext = ctk.CTkButton(
            self.toolbar, text="外部アプリで開く", width=120,
            fg_color="DarkGoldenrod", command=self.open_external,
        )
        self.btn_open_ext.pack(side="right", padx=5)

        # Canvas
        self.canvas = tk.Canvas(self, bg="gray", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.v_scroll = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=1, column=1, sticky="ns")
        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.h_scroll.grid(row=2, column=0, sticky="ew")

        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.doc = None
        self.doc_path = None
        self.current_page = 0
        self.zoom = 1.0
        self.img_id = None
        self.tk_img = None

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

    def load_doc(self, file_path, show=True):
        try:
            self.doc_path = Path(file_path)
            ext = self.doc_path.suffix.lower()
            if ext in {".jpg", ".jpeg", ".png"}:
                # 画像ファイルをPDFドキュメントに変換して表示
                img = Image.open(file_path)
                img_bytes = io.BytesIO()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_bytes, format="PDF")
                self.doc = pdfium.PdfDocument(img_bytes.getvalue())
            else:
                self.doc = pdfium.PdfDocument(file_path)
            self.current_page = 0
            self.zoom = 1.0
            if show:
                self.show_page(fit_to_page=True)
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルを開けませんでした: {e}")

    def show_page(self, fit_to_page=False):
        if not self.doc:
            return

        page = self.doc[self.current_page]

        if fit_to_page:
            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                pw, ph = page.get_size()
                zoom_w = (canvas_w * 0.95) / pw
                zoom_h = (canvas_h * 0.95) / ph
                self.zoom = min(zoom_w, zoom_h)

        img = page.render(scale=self.zoom).to_pil()
        self.tk_img = ImageTk.PhotoImage(img)

        if self.img_id:
            self.canvas.delete(self.img_id)

        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        x = max(0, (canvas_w - img.width) // 2)
        y = max(0, (canvas_h - img.height) // 2)

        self.img_id = self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.page_label.configure(text=f"ページ: {self.current_page + 1} / {len(self.doc)}")

        if self.on_page_change:
            self.on_page_change(self.current_page + 1)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.show_page()

    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.show_page()

    def zoom_in(self):
        self.zoom *= 1.2
        self.show_page()

    def zoom_out(self):
        self.zoom /= 1.2
        self.show_page()

    def on_press(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mousewheel(self, event):
        if event.state & 0x0004:  # Ctrl
            if event.delta > 0:
                self.zoom *= 1.1
            else:
                self.zoom /= 1.1
            self.show_page()
        else:
            if event.delta > 0:
                self.prev_page()
            else:
                self.next_page()

    def set_page(self, page_num):
        if self.doc and 0 <= page_num < len(self.doc):
            self.current_page = page_num
            self.show_page(fit_to_page=True)

    def clear(self):
        if self.img_id:
            self.canvas.delete(self.img_id)
            self.img_id = None
        self.doc = None
        self.doc_path = None
        self.tk_img = None
        self.page_label.configure(text="ページ: - / -")

    def open_external(self):
        if not self.doc_path:
            return
        page_num = self.current_page + 1
        path_obj = Path(self.doc_path)
        
        opened = False
        if path_obj.suffix.lower() == ".pdf":
            opened = self._open_pdf_at_page(path_obj, page_num)
        
        if not opened:
            os.startfile(str(self.doc_path))

    def _open_pdf_at_page(self, pdf_path: Path, page_num: int) -> bool:
        """PDFをページ指定で開く。成功した場合Trueを返す。"""
        path_str = str(pdf_path)
        
        # --- Adobe Acrobat / Reader ---
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

        # --- Browsers ---
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


class IngestApp(ctk.CTk):
    def __init__(self, target_file=None, target_page=1):
        super().__init__()
        self.title(f"HGNN-ingestor v{VERSION}")
        self._set_window_icon()
        self.geometry("1280x768")
        
        self.target_file = Path(target_file) if target_file else None
        self.target_page = target_page


        self.processor = PDFProcessor(DB_PATH, PYTHON_EXE)
        self.base_dir = KB_DIR
        self.is_processing = False
        self.is_edit_mode = False
        self.is_first_page_of_batch = False
        self.overwrite_mode = "ask"
        self.queue = queue.Queue()

        self._build_ui()
        self.after(100, self.check_queue)

        if self.base_dir.exists():
            self.after(500, self._initial_load_sequence)

    def _initial_load_sequence(self):
        self._add_initial_dir(self.base_dir)
        if self.target_file:
            self.after(200, self._jump_to_target)

    def _jump_to_target(self):
        if not self.target_file:
            return
        
        # 解決済みの絶対パスまたはファイル名で検索
        target_abs = self.target_file.absolute()
        
        for item in self.file_vars:
            path, rel_path, lbl, var, frame, has_data, is_out = item
            if path.absolute() == target_abs or path.name == self.target_file.name:
                self.select_pdf(path, rel_path)
                self.viewer.set_page(self.target_page - 1)
                self._highlight_item(frame)
                
                # スクロール位置の調整（簡易的）
                # ctk.CTkScrollableFrame に直接スクロールするメソッドがないため、
                # pack_forget/pack を駆使して位置を調整するのは難しいため、
                # ハイライト表示のみにとどめる。
                break

    # ------------------------------------------------------------------
    # Initial load
    # ------------------------------------------------------------------

    def _set_window_icon(self):
        try:
            icon_path = Path(__file__).parent / "assets" / "icon_ingestor.png"
            if icon_path.exists():
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                self.after(200, lambda: self.iconphoto(False, photo))
                self._icon_photo = photo
        except Exception:
            pass

    def _add_initial_dir(self, path):
        self._clear_list()
        base = path
        skip_count = 0
        
        conn = self.processor.ensure_db()
        processed = self.processor.get_processed_files(conn)
        conn.close()

        for root, dirs, files in os.walk(path):
            root_path = Path(root)
            # 探索深度制限 (path から見て 3階層まで)
            try:
                rel_from_root = root_path.relative_to(path)
                if len(rel_from_root.parts) >= 3:
                    dirs[:] = []
            except:
                pass

            for f in files:
                if Path(f).suffix.lower() in SUPPORTED_EXTS:
                    full_path = root_path / f
                    is_out_of_range = True
                    has_data = False
                    display_name = str(full_path)
                    
                    try:
                        rel_path = full_path.relative_to(self.base_dir.parent)
                        display_name = str(rel_path)
                        cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
                        if cat_path is not None:
                            is_out_of_range = False
                            has_data = (cat_path, full_path.name) in processed
                    except:
                        pass
                    
                    self._add_to_list(full_path, display_name, has_data=has_data, is_out_of_range=is_out_of_range)
        self._apply_filter()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top bar
        self.top_bar = ctk.CTkFrame(self)
        self.top_bar.pack(fill="x", padx=10, pady=(5, 0))

        ctk.CTkLabel(self.top_bar, text="ベースフォルダ:", font=("Meiryo", 12)).pack(side="left", padx=5)
        self.lbl_base_dir = ctk.CTkLabel(self.top_bar, text=str(self.base_dir), font=("Meiryo", 11), text_color="black")
        self.lbl_base_dir.pack(side="left", padx=5)
        ctk.CTkButton(self.top_bar, text="変更", width=60, command=self.change_base_dir).pack(side="left", padx=5)
        ctk.CTkButton(self.top_bar, text="DB新規作成", width=100, command=self.create_new_db, fg_color="#A9A9A9").pack(side="left", padx=5)

        # Main paned window
        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#f0f0f0", sashwidth=4)
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Left panel
        self.left_panel = tk.PanedWindow(self.paned, orient=tk.VERTICAL, bg="#f0f0f0", sashwidth=4)

        self.left_top = ctk.CTkFrame(self.left_panel)
        self.left_top.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(self.left_top, text="ファイル選択", command=self.add_files).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(self.left_top, text="フォルダ選択", command=self.add_dir).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.btn_start = ctk.CTkButton(self.left_top, text="取り込み開始", command=self.start_processing, width=120, fg_color="SeaGreen")
        self.btn_start.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        self.btn_stop = ctk.CTkButton(self.left_top, text="中断", command=self.stop_processing, width=120, fg_color="#CD5C5C", state="disabled")
        self.btn_stop.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        self.edit_mode_var = tk.BooleanVar(value=False)
        self.chk_edit_mode = ctk.CTkCheckBox(self.left_top, text="修正モード", variable=self.edit_mode_var, command=self.toggle_edit_mode)
        self.chk_edit_mode.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # File list
        self.list_frame = ctk.CTkFrame(self.left_panel)
        self.list_frame.grid_rowconfigure(1, weight=1)
        self.list_frame.grid_columnconfigure(0, weight=1)

        list_header = ctk.CTkFrame(self.list_frame, fg_color="transparent")
        list_header.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkLabel(list_header, text="ファイル一覧", font=("Meiryo", 12, "bold")).pack(side="left")

        self.lbl_list_status = ctk.CTkLabel(list_header, text="", font=("Meiryo", 11), text_color="#CD5C5C")
        self.lbl_list_status.pack(side="left", padx=10)

        self.list_filter_var = tk.StringVar(value="すべて")
        self.list_filter = ctk.CTkOptionMenu(
            list_header,
            variable=self.list_filter_var,
            values=["すべて", "登録済み", "未登録", "範囲外"],
            width=100,
            font=("Meiryo", 11),
            command=self._apply_filter
        )
        self.list_filter.pack(side="left", padx=5)

        self.select_all_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(list_header, text="全選択", variable=self.select_all_var, command=self.toggle_all).pack(side="right")

        self.scrollable_list = ctk.CTkScrollableFrame(self.list_frame)
        self.scrollable_list.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)

        # OCR text area
        self.ocr_frame = ctk.CTkFrame(self.left_panel)
        self.ocr_frame.grid_rowconfigure(1, weight=1)
        self.ocr_frame.grid_columnconfigure(0, weight=1)

        # ヘッダー行：「テキスト表示」ラベル ＋ スピナー（右側）
        ocr_header = ctk.CTkFrame(self.ocr_frame, fg_color="transparent")
        ocr_header.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        self.lbl_ocr_status = ctk.CTkLabel(ocr_header, text="テキスト表示", font=("Meiryo", 12, "bold"))
        self.lbl_ocr_status.pack(side="left")

        # スピナー部品（ラベル＋プログレスバー）を横並びで右側に配置
        self.ocr_spinner_frame = ctk.CTkFrame(ocr_header, fg_color="transparent")
        self.ocr_spinner_frame.pack(side="left", padx=(10, 0))
        self.ocr_spinner_label = ctk.CTkLabel(
            self.ocr_spinner_frame,
            text="このページのOCR処理中...",
            font=("Meiryo", 11),
            text_color=("gray40", "gray70"),
        )
        self.ocr_spinner_label.pack(side="left", padx=(0, 6))
        self.ocr_progress = ctk.CTkProgressBar(self.ocr_spinner_frame, mode="indeterminate", width=100, height=8)
        self.ocr_progress.pack(side="left")
        self.ocr_spinner_frame.pack_forget()  # 初期は非表示
        self._spinner_visible = False

        self.ocr_text = ctk.CTkTextbox(self.ocr_frame, font=("Meiryo", 12), state="disabled", wrap="word")
        self.ocr_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.btn_edit_frame = ctk.CTkFrame(self.ocr_frame, fg_color="transparent")
        self.btn_edit_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.btn_edit_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_update_db = ctk.CTkButton(self.btn_edit_frame, text="更新 (修正反映)", command=self.update_current_record, fg_color="SteelBlue")
        self.btn_update_db.grid(row=0, column=0, padx=(0, 2), sticky="ew")

        self.btn_delete_page = ctk.CTkButton(self.btn_edit_frame, text="ページ削除", command=self.delete_current_record, fg_color="#8B0000")
        self.btn_delete_page.grid(row=0, column=1, padx=(2, 0), sticky="ew")

        self.btn_edit_frame.grid_remove()

        self.left_panel.add(self.left_top, height=120)
        self.left_panel.add(self.list_frame, height=350)
        self.left_panel.add(self.ocr_frame)
        self.paned.add(self.left_panel, width=450)

        # Right panel (PDF viewer)
        self.viewer = PDFViewer(self.paned, on_page_change=self.on_viewer_page_change)
        self.paned.add(self.viewer)

        self.file_vars = []
        self.current_pdf_info = None
        self.selected_item_frame = None

    # ------------------------------------------------------------------
    # File list helpers
    # ------------------------------------------------------------------

    def change_base_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.base_dir)
        if new_dir:
            self.base_dir = Path(new_dir)
            self.lbl_base_dir.configure(text=str(self.base_dir))
            save_config(CONFIG_PATH, {"rag_base_path": str(self.base_dir).replace("\\", "/")})


    def _clear_list(self):
        for widget in self.scrollable_list.winfo_children():
            widget.destroy()
        self.file_vars = []
        self.selected_item_frame = None
        self.lbl_list_status.configure(text="")

    def add_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[
                ("対応ファイル", "*.pdf *.jpg *.jpeg *.png"),
                ("PDF files", "*.pdf"),
                ("画像ファイル", "*.jpg *.jpeg *.png"),
            ],
            initialdir=self.base_dir,
        )
        if files:
            self._clear_list()
            skip_count = 0
            
            conn = self.processor.ensure_db()
            processed = self.processor.get_processed_files(conn)
            conn.close()

            for f in files:
                path = Path(f)
                is_out_of_range = True
                has_data = False
                display_name = str(path)
                try:
                    rel_path = path.relative_to(self.base_dir.parent)
                    display_name = str(rel_path)
                    cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
                    if cat_path is not None:
                        is_out_of_range = False
                        has_data = (cat_path, path.name) in processed
                except ValueError:
                    pass
                self._add_to_list(path, display_name, has_data=has_data, is_out_of_range=is_out_of_range)
            self._apply_filter()

    def add_dir(self):
        dir_path_str = filedialog.askdirectory(initialdir=self.base_dir)
        if dir_path_str:
            dir_path = Path(dir_path_str)
            self._clear_list()
            
            conn = self.processor.ensure_db()
            processed = self.processor.get_processed_files(conn)
            conn.close()

            # os.walk の深度を制限するために dirs を操作する
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)
                # dir_path からの相対深度を計算
                try:
                    rel_from_root = root_path.relative_to(dir_path)
                    # 3階層下 (partsの長さが3) までで探索を止める
                    if len(rel_from_root.parts) >= 3:
                        dirs[:] = []
                except:
                    pass

                for f in files:
                    if Path(f).suffix.lower() in SUPPORTED_EXTS:
                        full_path = root_path / f
                        is_out_of_range = True
                        has_data = False
                        display_name = str(full_path)
                        
                        try:
                            # base_dir の親からの相対パスを取得 (kb/filename...)
                            rel_path = full_path.relative_to(self.base_dir.parent)
                            display_name = str(rel_path)
                            
                            # get_cat_path で 2階層以内か判定
                            cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
                            if cat_path is not None:
                                is_out_of_range = False
                                has_data = (cat_path, full_path.name) in processed
                        except ValueError:
                            # base_dir 外の場合
                            pass
                            
                        self._add_to_list(full_path, display_name, has_data=has_data, is_out_of_range=is_out_of_range)
            self._apply_filter()

    def _add_to_list(self, path, display_name, has_data=False, is_out_of_range=False):
        if is_out_of_range:
            self.lbl_list_status.configure(text="範囲外があります。")

        frame = ctk.CTkFrame(self.scrollable_list, fg_color="transparent")
        var = tk.BooleanVar(value=(not is_out_of_range)) # 範囲外はデフォルトOFF
        cb = ctk.CTkCheckBox(frame, text="", variable=var, width=20, state=("disabled" if is_out_of_range else "normal"))
        cb.pack(side="left", padx=(0, 5))
        
        color = COLOR_OUT_OF_RANGE if is_out_of_range else (COLOR_HAS_DATA if has_data else None)
        font = ("Meiryo", 11, "overstrike") if is_out_of_range else ("Meiryo", 11)
        
        lbl = ctk.CTkLabel(frame, text=display_name, anchor="w", cursor="hand2", font=font, text_color=color)
        lbl.pack(side="left", fill="x", expand=True)

        def on_click(e, f=frame, p=path, rp=Path(display_name)):
            self.select_pdf(p, rp)
            self._highlight_item(f)

        lbl.bind("<Button-1>", on_click)
        self.file_vars.append([path, Path(display_name), lbl, var, frame, has_data, is_out_of_range])

    def _apply_filter(self, choice=None):
        choice = choice or self.list_filter_var.get()
        # 一旦すべて隠してから、条件に合うものだけ再配置
        for item in self.file_vars:
            item[4].pack_forget()
        
        for item in self.file_vars:
            path, rel_path, lbl, var, frame, has_data, is_out_of_range = item
            if choice == "すべて" and not is_out_of_range:
                frame.pack(fill="x", padx=5, pady=1)
            elif choice == "登録済み" and has_data and not is_out_of_range:
                frame.pack(fill="x", padx=5, pady=1)
            elif choice == "未登録" and not has_data and not is_out_of_range:
                frame.pack(fill="x", padx=5, pady=1)
            elif choice == "範囲外" and is_out_of_range:
                frame.pack(fill="x", padx=5, pady=1)

    def _highlight_item(self, frame):
        if self.selected_item_frame:
            self.selected_item_frame.configure(fg_color="transparent")
        self.selected_item_frame = frame
        self.selected_item_frame.configure(fg_color="#3B8ED0")

    def toggle_all(self):
        val = self.select_all_var.get()
        for item in self.file_vars:
            item[3].set(val)

    # ------------------------------------------------------------------
    # PDF selection & text display
    # ------------------------------------------------------------------

    def select_pdf(self, path, rel_path):
        self.current_pdf_info = (path, rel_path)
        self.viewer.load_doc(path)
        self.load_existing_text()

    def toggle_edit_mode(self):
        self.is_edit_mode = self.edit_mode_var.get()
        if self.is_edit_mode:
            self.ocr_text.configure(state="normal")
            self.btn_edit_frame.grid()
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="disabled")
            self.lbl_ocr_status.configure(text="テキスト表示 (修正可能)")
        else:
            self.ocr_text.configure(state="disabled")
            self.btn_edit_frame.grid_remove()
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.lbl_ocr_status.configure(text="テキスト表示")
        self.load_existing_text()

    def load_existing_text(self):
        if not self.current_pdf_info or not self.viewer.doc:
            return
        path, rel_path = self.current_pdf_info
        page_num = self.viewer.current_page + 1
        cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
        filename = path.name
        conn = self.processor.ensure_db()
        text = self.processor.check_existing_record(conn, cat_path, filename, page_num)
        conn.close()
        self.ocr_text.configure(state="normal")
        self.ocr_text.delete("1.0", "end")
        
        display_text = ""
        if text:
            display_text = text
        else:
            if not self.is_edit_mode:
                display_text = "(DBに記録なし)"
        
        self.ocr_text.insert("1.0", display_text)
        
        # データがないときはページ削除はグレーアウト
        if text:
            self.btn_delete_page.configure(state="normal")
        else:
            self.btn_delete_page.configure(state="disabled")

        if not self.is_edit_mode:
            self.ocr_text.configure(state="disabled")

    def update_current_record(self):
        if not self.current_pdf_info:
            return
        path, rel_path = self.current_pdf_info
        page_num = self.viewer.current_page + 1
        new_text = self.ocr_text.get("1.0", "end-1c").strip()
        
        if not new_text or new_text == "(DBに記録なし)":
            messagebox.showwarning("警告", "有効なテキストを入力してください。")
            return

        cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
        filename = path.name
        conn = sqlite3.connect(self.processor.db_path)
        self.processor.update_record(conn, cat_path, filename, page_num, new_text)
        conn.close()
        messagebox.showinfo("完了", f"{filename} P.{page_num} を更新しました。")

    def delete_current_record(self):
        if not self.current_pdf_info:
            return
        path, rel_path = self.current_pdf_info
        page_num = self.viewer.current_page + 1
        filename = path.name
        
        if not messagebox.askyesno("削除確認", f"{filename} P.{page_num} をDBから削除しますか？\n(ファイル自体は削除されません)"):
            return
            
        cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
        conn = sqlite3.connect(self.processor.db_path)
        success = self.processor.delete_record(conn, cat_path, filename, page_num)
        conn.close()
        
        if success:
            messagebox.showinfo("完了", "ページデータをDBから削除しました。")
            self.load_existing_text()
        else:
            messagebox.showwarning("警告", "削除するレコードが見つかりませんでした。")

    def on_viewer_page_change(self, page_num):
        if not self.is_processing:
            self.load_existing_text()

    def create_new_db(self):
        if self.is_processing:
            messagebox.showwarning("警告", "処理中はDBを新規作成できません。")
            return
        initial_dir = self.processor.db_path.parent
        if not initial_dir.exists():
            initial_dir = BASE_DIR / "db"

        new_db_path = filedialog.asksaveasfilename(
            title="新規データベースの保存先を選択",
            initialdir=initial_dir,
            initialfile="knowledge.db",
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
        )
        if not new_db_path:
            return

        if messagebox.askyesno("DB新規作成", f"新しいデータベースを作成しますか？\n場所: {new_db_path}"):
            try:
                self.processor.db_path = Path(new_db_path)
                conn = self.processor.ensure_db(force_new=True)
                conn.close()
                
                # パスを相対パス（可能な場合）または絶対パスで保存
                try:
                    rel_path = os.path.relpath(new_db_path, os.path.dirname(__file__))
                    # ..\db\knowledge.db などを ../db/knowledge.db に変換
                    save_path = rel_path.replace("\\", "/")
                except ValueError:
                    save_path = new_db_path.replace("\\", "/")
                
                save_config(CONFIG_PATH, {"rag_db_path": save_path})
                
                # ファイル一覧のステータスをリセット
                default_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
                for item in self.file_vars:
                    item[5] = False # has_data
                    if not item[6]: # not is_out_of_range
                        item[2].configure(text_color=default_text_color) # デフォルト色に戻す
                self._apply_filter()

                messagebox.showinfo("完了", "DBを新規作成し、設定を保存しました。")
            except Exception as e:
                messagebox.showerror("エラー", f"DB作成に失敗しました: {e}")


    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def start_processing(self):
        if self.is_edit_mode:
            return
        selected = [item for item in self.file_vars if item[3].get()]
        if not selected:
            messagebox.showwarning("警告", "処理するファイルを選択してください。")
            return
        self.is_processing = True
        self.is_first_page_of_batch = True
        self.overwrite_mode = "ask"
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.processor.stop_requested = False

        # Clear UI initially
        self.viewer.clear()
        self.ocr_text.configure(state="normal")
        self.ocr_text.delete("1.0", "end")
        self.ocr_text.configure(state="disabled")

        self._hide_spinner()

        if not self.processor.ocr_mgr.is_loaded() and OCR_AVAILABLE:
            self._prepare_ocr_and_start(selected)
        else:
            threading.Thread(target=self.run_ingestion, args=(selected,), daemon=True).start()

    def _prepare_ocr_and_start(self, selected):
        """OCRモデルをロードしてから取り込みを開始する（初回のみ）。"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("OCR準備中")
        dialog.geometry("420x120")
        dialog.attributes("-topmost", True)
        dialog.resizable(False, False)
        px = self.winfo_rootx() + (self.winfo_width() - 420) // 2
        py = self.winfo_rooty() + (self.winfo_height() - 120) // 2
        dialog.geometry(f"+{px}+{py}")
        ctk.CTkLabel(dialog, text="NDLOCR-Lite を読み込み中です...\n初回のみ時間がかかります。",
                     font=("Meiryo", 13)).pack(expand=True, padx=20, pady=20)
        
        # 強制的に描画を更新
        dialog.update_idletasks()
        dialog.update()

        def load_task():
            self.processor.ocr_mgr.ensure_models()
            self.after(100, finalize)

        def finalize():
            if dialog.winfo_exists():
                dialog.destroy()
            threading.Thread(target=self.run_ingestion, args=(selected,), daemon=True).start()

        # ウィンドウが確実に表示されてからスレッドを開始
        self.after(200, lambda: threading.Thread(target=load_task, daemon=True).start())

    def stop_processing(self):
        self.processor.stop_requested = True
        self.btn_stop.configure(state="disabled")

    def run_ingestion(self, selected_items):
        conn = self.processor.ensure_db()
        try:
            # --- 削除追跡: 処理対象ファイルに対応するレコードを「削除候補」(-1) にリセット ---
            # 選択ファイルの (path, filename, page) 集合を収集するため、まずファイル側の
            # path/filename を割り出してから対応レコードを -1 にする
            cursor = conn.cursor()
            for item in selected_items:
                path, rel_path, _lbl, _var, _frame, _has_data, _is_out = item
                cat_path = self.processor.get_cat_path(rel_path, self.base_dir.name)
                if cat_path is None:
                    continue
                filename = path.name
                cursor.execute(
                    "UPDATE ocr_texts SET fts_synced = -1 WHERE path = ? AND filename = ?",
                    (cat_path, filename),
                )
            conn.commit()

            any_skipped = False
            for item in selected_items:
                path, rel_path, lbl, var, frame, has_data, is_out_of_range = item
                if self.processor.stop_requested:
                    break
                self.queue.put(("highlight", (lbl, frame)))
                self.queue.put(("load_pdf_silent", (path, rel_path)))
                
                result = self.processor.process_file(
                    path, rel_path, self.base_dir.name, conn,
                    status_callback=lambda msg: self.queue.put(("status", msg)),
                    overwrite_callback=lambda f, p, path=path, rel_path=rel_path: self.wait_for_overwrite_choice(f, p, path, rel_path),
                )
                if isinstance(result, tuple):
                    success, is_skipped = result
                else:
                    success, is_skipped = result, False
                    
                if is_skipped:
                    any_skipped = True

                if success:
                    item[5] = True # has_data を更新
                    self.queue.put(("done", lbl))
                    self.queue.put(("deselect", var))
                else:
                    self.queue.put(("interrupted", lbl))
                    break

            # --- 削除追跡: -1 のまま残ったレコードはファイルが消えた扱いで削除 ---
            # 停止中断時は実行しない（途中まましか処理していないため誤削除を防ぐ）
            if not self.processor.stop_requested:
                cursor.execute("DELETE FROM ocr_texts_fts WHERE rowid IN (SELECT id FROM ocr_texts WHERE fts_synced = -1)")
                cursor.execute("DELETE FROM ocr_texts WHERE fts_synced = -1")
                conn.commit()
        finally:
            conn.close()
            self.queue.put(("finished", any_skipped))

    def wait_for_overwrite_choice(self, filename, page, path, rel_path):
        if self.overwrite_mode == "always":
            return True
        if self.overwrite_mode == "skip_all":
            return False
        self.queue.put(("load_pdf_and_page", (path, rel_path, page)))
        res_queue = queue.Queue()
        self.queue.put(("ask_overwrite", (filename, page, res_queue)))
        return res_queue.get()

    # ------------------------------------------------------------------
    # Overwrite dialog
    # ------------------------------------------------------------------

    def _show_overwrite_dialog(self, fname, page, res_q):
        """重複確認ダイアログ。ウィンドウサイズをコンテンツに合わせて自動調整する。"""
        dlg = ctk.CTkToplevel(self)
        dlg.title("重複確認")
        dlg.attributes("-topmost", True)
        dlg.resizable(False, False)

        def on_choice(choice):
            dlg.grab_release()
            dlg.destroy()
            if choice == "always":
                self.overwrite_mode = "always"
                res_q.put(True)
            elif choice == "yes":
                res_q.put(True)
            elif choice == "skip_all":
                self.overwrite_mode = "skip_all"
                res_q.put(False)
            else:
                res_q.put(False)

        main_f = ctk.CTkFrame(dlg, fg_color="transparent")
        main_f.pack(expand=True, fill="both", padx=30, pady=30)

        msg = f"ファイル: {fname}\n{page} ページ目\n\n既にレコードが存在します。上書きしますか？"
        ctk.CTkLabel(main_f, text=msg, font=("Meiryo", 15), justify="center").pack(pady=(0, 30))

        btn_f = ctk.CTkFrame(main_f, fg_color="transparent")
        btn_f.pack()

        btn_cfg = dict(width=180, height=55, font=("Meiryo", 14))
        ctk.CTkButton(btn_f, text="上書き",          **btn_cfg, command=lambda: on_choice("yes")     ).grid(row=0, column=0, padx=15, pady=10)
        ctk.CTkButton(btn_f, text="以降すべて上書き", **btn_cfg, command=lambda: on_choice("always")  ).grid(row=0, column=1, padx=15, pady=10)
        ctk.CTkButton(btn_f, text="スキップ",         **btn_cfg, command=lambda: on_choice("no")      ).grid(row=1, column=0, padx=15, pady=10)
        ctk.CTkButton(btn_f, text="以降すべてスキップ", **btn_cfg, command=lambda: on_choice("skip_all")).grid(row=1, column=1, padx=15, pady=10)

        # --- サイズを内容に合わせて自動計算してから中央配置 ---
        dlg.update_idletasks()
        req_w = dlg.winfo_reqwidth()
        req_h = dlg.winfo_reqheight()
        w = max(req_w, 520)
        h = max(req_h, 320)
        screen_w = dlg.winfo_screenwidth()
        screen_h = dlg.winfo_screenheight()
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.update()  # ウィンドウを完全に描画してから grab_set
        dlg.grab_set()

    # ------------------------------------------------------------------
    # Spinner helpers
    # ------------------------------------------------------------------

    def _show_spinner(self):
        if self._spinner_visible:
            return
        self._spinner_visible = True
        self.ocr_spinner_frame.pack(side="left", padx=(10, 0))
        self.ocr_progress.start()

    def _hide_spinner(self):
        if not self._spinner_visible:
            return
        self._spinner_visible = False
        self.ocr_progress.stop()
        self.ocr_spinner_frame.pack_forget()

    # ------------------------------------------------------------------
    # Queue processor (runs on UI thread via after())
    # ------------------------------------------------------------------

    def check_queue(self):
        try:
            while True:
                try:
                    msg_type, data = self.queue.get_nowait()
                except queue.Empty:
                    break

                if msg_type == "status":
                    if isinstance(data, tuple) and data[0] == "page_start":
                        page_num = data[1]
                        if self.is_first_page_of_batch:
                            self.ocr_text.configure(state="normal")
                            self.ocr_text.delete("1.0", "end")
                            self.ocr_text.insert("1.0", "作業中...")
                            self.ocr_text.configure(state="disabled")
                            self.viewer.set_page(page_num - 1)
                            self.is_first_page_of_batch = False
                            self.ocr_spinner_label.configure(text="このページのOCR処理中...")
                        else:
                            # 2ページ目以降の開始時は表示を維持（ブランクにしない）
                            self.ocr_spinner_label.configure(text="次のページのOCR処理中...")
                    elif isinstance(data, tuple) and data[0] == "text_update":
                        _, page_num, text = data
                        self._hide_spinner()
                        self.ocr_text.configure(state="normal")
                        self.ocr_text.delete("1.0", "end")
                        self.ocr_text.insert("end", text)
                        if not self.is_edit_mode:
                            self.ocr_text.configure(state="disabled")
                        self.viewer.set_page(page_num - 1)
                    elif isinstance(data, str):
                        if "performing OCR" in data:
                            self._show_spinner()
                        # UI上部のステータスラベルへの表示は無効化

                elif msg_type == "ask_overwrite":
                    fname, page, res_q = data
                    self.update_idletasks()
                    self._show_overwrite_dialog(fname, page, res_q)

                elif msg_type == "highlight":
                    lbl, frame = data
                    lbl.configure(text_color=COLOR_IN_PROGRESS)
                    self._highlight_item(frame)

                elif msg_type == "done":
                    data.configure(text_color=COLOR_HAS_DATA)

                elif msg_type == "deselect":
                    data.set(False)

                elif msg_type == "load_pdf_silent":
                    path, rel_path = data
                    self.current_pdf_info = (path, rel_path)
                    self.viewer.load_doc(path, show=False)

                elif msg_type == "load_pdf_and_page":
                    path, rel_path, page = data
                    self.select_pdf(path, rel_path)
                    self.viewer.set_page(page - 1)

                elif msg_type == "finished":
                    any_skipped = data
                    self.is_processing = False
                    self._hide_spinner()
                    if not self.is_edit_mode:
                        self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    if not self.is_edit_mode:
                        self.ocr_text.configure(state="disabled")
                    self._apply_filter() # フィルタ表示を更新（登録済みへの反映など）
                    
                    if any_skipped:
                        messagebox.showinfo("完了", "処理が終了しました。\n（変更が無いためスキップされたファイルが含まれています）")
                    else:
                        messagebox.showinfo("完了", "処理が終了しました。")

        except Exception as e:
            print(f"Exception in check_queue: {e}")

        self.after(100, self.check_queue)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF Ingestor for Knowledge RAG")
    parser.add_argument("--new", action="store_true", help="Create a new DB (backups old one)")
    parser.add_argument("--file", type=str, help="Open a specific file in GUI")
    parser.add_argument("--page", type=int, default=1, help="Page number to open (1-indexed)")
    args = parser.parse_args()

    # Ingestion mode (CLI only)
    if args.new:
        if not KB_DIR.exists():
            KB_DIR.mkdir()
            print(f"Created {KB_DIR} folder. Please place your PDFs there.")
            return

        processor = PDFProcessor(DB_PATH, PYTHON_EXE)
        conn = processor.ensure_db(force_new=True)
        cursor = conn.cursor()
        try:
            seen_files = set()
            for root, dirs, files in os.walk(KB_DIR):
                for file in files:
                    if Path(file).suffix.lower() in SUPPORTED_EXTS:
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(KB_DIR.parent)
                        cat_path = processor.get_cat_path(rel_path, KB_DIR.name)
                        if cat_path:
                            seen_files.add((cat_path, file))
                        processor.process_file(full_path, rel_path, KB_DIR.name, conn)

            db_files = set(cursor.execute("SELECT DISTINCT path, filename FROM ocr_texts").fetchall())
            to_delete = db_files - seen_files
            if to_delete:
                print(f"Cleaning up {len(to_delete)} deleted files...")
                for path, filename in to_delete:
                    cursor.execute("DELETE FROM ocr_texts_fts WHERE rowid IN (SELECT id FROM ocr_texts WHERE path=? AND filename=?)", (path, filename))
                    cursor.execute("DELETE FROM ocr_texts WHERE path = ? AND filename = ?", (path, filename))
                    cursor.execute("DELETE FROM rag_files WHERE path = ? AND filename = ?", (path, filename))
                conn.commit()
                print(f"Deleted {len(to_delete)} files from DB.")
        finally:
            conn.close()
        print("Ingestion complete.")
        return

    # GUI Mode (Default)
    app = IngestApp(target_file=args.file, target_page=args.page)
    app.mainloop()


if __name__ == "__main__":
    main()
