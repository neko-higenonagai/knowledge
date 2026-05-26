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

VERSION = "0.0.2"

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk
import pypdfium2 as pdfium
from PIL import Image, ImageTk
from pathlib import Path
import io
from collections import deque

from config_ai_common import ensure_config, resolve_rag_base_path, save_config

# サポートする拡張子
SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

# ===========================================================================
# Configuration & Constants
# ===========================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "config_ai.txt"
KB_DIR = resolve_rag_base_path(ensure_config(CONFIG_PATH))

UNDO_DEPTH = 5

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# ===========================================================================
# Dialogs
# ===========================================================================

class DestinationDialog(ctk.CTkToplevel):
    def __init__(self, master, max_pages, title="移動先を指定"):
        super().__init__(master)
        self.title(title)
        
        # 画面中央に表示
        self.update_idletasks()
        w, h = 300, 250
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        
        self.result = None
        self.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self, text=title, font=("Meiryo", 14, "bold")).pack(pady=10)
        
        self.choice = tk.StringVar(value="after")
        
        ctk.CTkRadioButton(self, text="先頭", variable=self.choice, value="head").pack(anchor="w", padx=40, pady=5)
        ctk.CTkRadioButton(self, text="末尾", variable=self.choice, value="tail").pack(anchor="w", padx=40, pady=5)
        
        after_frame = ctk.CTkFrame(self, fg_color="transparent")
        after_frame.pack(anchor="w", padx=40, pady=5)
        ctk.CTkRadioButton(after_frame, text="指定ページの後：", variable=self.choice, value="after").pack(side="left")
        self.page_entry = ctk.CTkEntry(after_frame, width=50)
        self.page_entry.insert(0, str(max_pages))
        self.page_entry.pack(side="left", padx=5)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text="OK", width=80, command=self.on_ok).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="キャンセル", width=80, command=self.on_cancel).pack(side="left", padx=5)
        
        self.grab_set()

    def on_ok(self):
        choice = self.choice.get()
        if choice == "head":
            self.result = 0
        elif choice == "tail":
            self.result = -1 # Special value for end
        else:
            try:
                p = int(self.page_entry.get())
                self.result = p  # Insert after page p (0-indexed logic: insert at p)
            except ValueError:
                messagebox.showerror("エラー", "有効なページ番号を入力してください")
                return
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

class ProcessingDialog(ctk.CTkToplevel):
    def __init__(self, master, message="処理中..."):
        super().__init__(master)
        self.overrideredirect(True)  # タイトルバーを非表示にする
        
        # デザインの設定
        self.configure(fg_color=("#F0F0F0", "#1E1E1E"))
        
        # 親ウィンドウの中央に配置
        master.update_idletasks()
        parent_x = master.winfo_x()
        parent_y = master.winfo_y()
        parent_w = master.winfo_width()
        parent_h = master.winfo_height()
        
        w, h = 320, 120
        x = parent_x + (parent_w - w) // 2
        y = parent_y + (parent_h - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        
        # 枠線とコンテナ
        border_frame = ctk.CTkFrame(self, fg_color="transparent", border_width=2, border_color="#3B8ED0")
        border_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        # メッセージ表示
        self.label = ctk.CTkLabel(border_frame, text=message, font=("Meiryo", 16, "bold"))
        self.label.pack(pady=(30, 5))
        
        self.sub_label = ctk.CTkLabel(border_frame, text="しばらくお待ちください...", font=("Meiryo", 11), text_color="gray")
        self.sub_label.pack(pady=(0, 20))
        
        self.grab_set()  # モーダル化（他の操作を一時ブロック）
        self.update()

# ===========================================================================
# PDF Management Logic
# ===========================================================================

class PDFManager:
    def __init__(self):
        self.doc = None
        self.path = None
        self.initial_bytes = None
        self.last_added_filename = ""
        self._history_stack = deque(maxlen=UNDO_DEPTH)

    def load(self, path):
        if self.doc:
            self.doc.close()
            self.doc = None
        self.path = Path(path)
        ext = self.path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            # 画像ファイルをPDFドキュメントに変換
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pdf_bytes = io.BytesIO()
            img.save(pdf_bytes, format="PDF")
            self.initial_bytes = pdf_bytes.getvalue()
        else:
            with open(path, "rb") as f:
                self.initial_bytes = f.read()
        self._history_stack.clear()
        self.last_added_filename = ""
        self.doc = pdfium.PdfDocument(self.initial_bytes)

    def _save_history(self):
        out = io.BytesIO()
        self.doc.save(out)
        self._history_stack.append((out.getvalue(), self.last_added_filename))

    def rotate(self, page_indices, angle):
        if not self.doc: return
        self._save_history()
        for idx in page_indices:
            page = self.doc[idx]
            new_rot = int((page.get_rotation() + angle) % 360)
            page.set_rotation(new_rot)

    def is_page_cropped(self, idx):
        if not self.doc: return False
        page = self.doc[idx]
        media = page.get_mediabox()
        crop = page.get_cropbox()
        # 微小な差は無視
        return any(abs(m - c) > 1 for m, c in zip(media, crop))

    def apply_crop(self, page_indices, rect):
        """rect: (left, bottom, right, top) in PDF points (unrotated)"""
        if not self.doc: return
        self._save_history()
        for idx in page_indices:
            page = self.doc[idx]
            page.set_cropbox(*rect)

    def reset_crop(self, page_indices):
        if not self.doc: return
        self._save_history()
        for idx in page_indices:
            page = self.doc[idx]
            media = page.get_mediabox()
            page.set_cropbox(*media)

    def delete(self, page_indices):
        if not self.doc: return
        self._save_history()
        for idx in sorted(page_indices, reverse=True):
            self.doc.del_page(idx)

    def move_page(self, old_idx, new_idx):
        if not self.doc: return
        if old_idx == new_idx: return
        self._save_history()
        
        temp_doc = pdfium.PdfDocument.new()
        temp_doc.import_pages(self.doc, [old_idx])
        self.doc.del_page(old_idx)
        
        # dst番目への挿入処理 (PyMuPDFのmove_page仕様: 元のリストのdst番目の前に挿入)
        # del_pageしたことで、old_idxより後ろのインデックスは1つ前にずれているため補正する
        if new_idx == -1:
            new_idx = len(self.doc)
        elif old_idx < new_idx:
            new_idx -= 1
        
        self.doc.import_pages(temp_doc, [0], index=new_idx)

    def move_multiple_pages(self, indices, target_idx):
        """複数のページを一括移動"""
        if not self.doc: return
        self._save_history()
        
        # 挿入位置が末尾の場合
        if target_idx == -1:
            target_idx = len(self.doc)
            
        # 手順: 
        # 1. 指定されたページの内容を一時的に保持
        # 2. 逆順で元の場所から削除
        # 3. 指定位置に挿入
        
        # pypdfium2 の import_pages を使用
        temp_doc = pdfium.PdfDocument.new()
        temp_doc.import_pages(self.doc, sorted(indices))
        
        for idx in sorted(indices, reverse=True):
            self.doc.del_page(idx)
            if idx < target_idx:
                target_idx -= 1
        
        self.doc.import_pages(temp_doc, index=target_idx)

    def copy_pages(self, indices, target_idx):
        """複数のページを一括コピー"""
        if not self.doc: return
        self._save_history()
        
        if target_idx == -1:
            target_idx = len(self.doc)
            
        temp_doc = pdfium.PdfDocument.new()
        temp_doc.import_pages(self.doc, sorted(indices))
        
        self.doc.import_pages(temp_doc, index=target_idx)

    def insert(self, other_file_path, position_idx):
        if not self.doc: return
        self._save_history()
        self.last_added_filename = Path(other_file_path).name
        other_path = Path(other_file_path)
        ext = other_path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            # 画像ファイルをPDFに変換してから挿入
            img = Image.open(other_file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pdf_bytes = io.BytesIO()
            img.save(pdf_bytes, format="PDF")
            other_doc = pdfium.PdfDocument(pdf_bytes.getvalue())
        else:
            other_doc = pdfium.PdfDocument(other_file_path)
        
        if position_idx == -1:
            position_idx = len(self.doc)
            
        self.doc.import_pages(other_doc, index=position_idx)

    def undo(self):
        if not self._history_stack:
            return False
        snap_bytes, snap_filename = self._history_stack.pop()
        if self.doc:
            self.doc.close()
        self.doc = pdfium.PdfDocument(snap_bytes)
        self.last_added_filename = snap_filename
        return True

    def reset(self):
        if self.initial_bytes:
            if self.doc:
                self.doc.close()
            self.doc = pdfium.PdfDocument(self.initial_bytes)
            self._history_stack.clear()
            self.last_added_filename = ""
            return True
        return False

    def save(self, output_path):
        if self.doc:
            self.doc.save(output_path)
            return True
        return False

    def close(self):
        if self.doc: self.doc.close()

# ===========================================================================
# GUI Components
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

        self.btn_zoom_out = ctk.CTkButton(self.toolbar, text="- 縮小", width=80, command=self.zoom_out)
        self.btn_zoom_out.pack(side="left", padx=(20, 2))
        self.btn_zoom_in = ctk.CTkButton(self.toolbar, text="+ 拡大", width=80, command=self.zoom_in)
        self.btn_zoom_in.pack(side="left", padx=2)

        self.btn_open_ext = ctk.CTkButton(
            self.toolbar, text="外部アプリで開く", width=120,
            fg_color="DarkGoldenrod", command=self.open_external,
        )
        self.btn_open_ext.pack(side="right", padx=5)

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
        self.tk_img = None
        self.img_id = None

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        self.mode = "view" # "view" or "crop"
        self.crop_callback = None
        self.rect_id = None
        self.start_x = 0
        self.start_y = 0

    def set_mode(self, mode, callback=None):
        self.mode = mode
        self.crop_callback = callback
        if mode == "crop":
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")

    def set_doc(self, doc, current_page=0):
        self.doc = doc
        self.current_page = min(current_page, len(doc)-1) if doc and len(doc)>0 else 0
        self.show_page(fit_to_page=True)

    def show_page(self, fit_to_page=False):
        if not self.doc or len(self.doc) == 0:
            self.canvas.delete("all")
            self.page_label.configure(text="ページ: - / -")
            return

        if self.current_page >= len(self.doc): self.current_page = len(self.doc) - 1
        if self.current_page < 0: self.current_page = 0

        page = self.doc[self.current_page]
        if fit_to_page:
            self.canvas.update_idletasks()
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w > 1 and h > 1:
                pw, ph = page.get_size()
                self.zoom = min((w*0.9)/pw, (h*0.9)/ph)

        img = page.render(scale=self.zoom).to_pil()
        self.tk_img = ImageTk.PhotoImage(img)

        if self.img_id: self.canvas.delete(self.img_id)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = max(0, (cw - img.width)//2), max(0, (ch - img.height)//2)
        self.img_id = self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.page_label.configure(text=f"ページ: {self.current_page + 1} / {len(self.doc)}")

        if self.on_page_change:
            self.on_page_change(self.current_page)

    def prev_page(self):
        if self.current_page > 0: self.current_page -= 1; self.show_page()
    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1: self.current_page += 1; self.show_page()
    def zoom_in(self): self.zoom *= 1.2; self.show_page()
    def zoom_out(self): self.zoom /= 1.2; self.show_page()
    def on_press(self, e):
        if self.mode == "crop":
            self.start_x = self.canvas.canvasx(e.x)
            self.start_y = self.canvas.canvasy(e.y)
            if self.rect_id: self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)
            return
        self.canvas.scan_mark(e.x, e.y)
        self.canvas.config(cursor="fleur")
        
    def on_drag(self, e):
        if self.mode == "crop":
            cur_x = self.canvas.canvasx(e.x)
            cur_y = self.canvas.canvasy(e.y)
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
            return
        self.canvas.scan_dragto(e.x, e.y, gain=1)
        
    def on_release(self, e):
        if self.mode == "crop":
            end_x = self.canvas.canvasx(e.x)
            end_y = self.canvas.canvasy(e.y)
            self.finish_crop_selection(self.start_x, self.start_y, end_x, end_y)
            return
        self.canvas.config(cursor="")

    def finish_crop_selection(self, x1, y1, x2, y2):
        if not self.doc or self.current_page >= len(self.doc): return
        
        # キャンバス上の画像位置を取得
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        page = self.doc[self.current_page]
        bitmap = page.render(scale=self.zoom)
        img_w, img_h = bitmap.width, bitmap.height
        off_x, off_y = max(0, (cw - img_w)//2), max(0, (ch - img_h)//2)
        
        # 相対座標 (pixel)
        rx1, ry1 = (min(x1, x2) - off_x), (min(y1, y2) - off_y)
        rx2, ry2 = (max(x1, x2) - off_x), (max(y1, y2) - off_y)
        
        # PDFポイントに変換 (render時のscale=self.zoomを考慮)
        u_x1 = rx1 / self.zoom
        u_x2 = rx2 / self.zoom
        u_y1 = ry1 / self.zoom
        u_y2 = ry2 / self.zoom
        
        # 現在のunrotatedなcropboxを取得
        c_left, c_bottom, c_right, c_top = page.get_cropbox()
        rot = page.get_rotation()
        
        # 始点 (u_x1, u_y1) のマッピング
        if rot == 0:
            X1 = c_left + u_x1
            Y1 = c_top - u_y1
        elif rot == 90:
            X1 = c_left + u_y1
            Y1 = c_bottom + u_x1
        elif rot == 180:
            X1 = c_right - u_x1
            Y1 = c_bottom + u_y1
        elif rot == 270:
            X1 = c_right - u_y1
            Y1 = c_top - u_x1
        else:
            X1 = c_left + u_x1
            Y1 = c_top - u_y1

        # 終点 (u_x2, u_y2) のマッピング
        if rot == 0:
            X2 = c_left + u_x2
            Y2 = c_top - u_y2
        elif rot == 90:
            X2 = c_left + u_y2
            Y2 = c_bottom + u_x2
        elif rot == 180:
            X2 = c_right - u_x2
            Y2 = c_bottom + u_y2
        elif rot == 270:
            X2 = c_right - u_y2
            Y2 = c_top - u_x2
        else:
            X2 = c_left + u_x2
            Y2 = c_top - u_y2

        # 範囲を現在のcropbox境界内に収める
        x_min = min(c_left, c_right)
        x_max = max(c_left, c_right)
        y_min = min(c_bottom, c_top)
        y_max = max(c_bottom, c_top)
        
        X1 = max(x_min, min(x_max, X1))
        X2 = max(x_min, min(x_max, X2))
        Y1 = max(y_min, min(y_max, Y1))
        Y2 = max(y_min, min(y_max, Y2))

        # 新しいcropboxの算出
        pdf_l = min(X1, X2)
        pdf_b = min(Y1, Y2)
        pdf_r = max(X1, X2)
        pdf_t = max(Y1, Y2)
        
        if self.crop_callback:
            self.crop_callback((pdf_l, pdf_b, pdf_r, pdf_t))
        
        self.set_mode("view")
        if self.rect_id: self.canvas.delete(self.rect_id); self.rect_id = None
    def on_mousewheel(self, e):
        if e.state & 0x0004:
            if e.delta > 0: self.zoom_in()
            else: self.zoom_out()
        else:
            if e.delta > 0: self.prev_page()
            else: self.next_page()

    def open_external(self):
        # winfo_toplevel() (PDFEditorApp) の manager からパスを取得
        toplevel = self.winfo_toplevel()
        if not hasattr(toplevel, "manager") or not toplevel.manager.path:
            return
        
        pdf_path = toplevel.manager.path
        page_num = self.current_page + 1
        
        opened = False
        if pdf_path.suffix.lower() == ".pdf":
            opened = self._open_pdf_at_page(pdf_path, page_num)
        
        if not opened:
            os.startfile(str(pdf_path))

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

class ThumbnailPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, on_select_page=None, on_move_page=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_select_page = on_select_page
        self.on_move_page = on_move_page
        self.thumbnails = []  # (frame, checkbox, var)
        self.doc = None
        self.active_idx = -1

        # ドラッグ状態
        self._drag_src = None   # ドラッグ開始ページインデックス
        self._drag_slot = None  # 現在のドロップ先スロット(0=先頭, n=末尾)
        self._scroll_job = None
        self._scroll_dir = 0    # -1: up, 1: down, 0: none
        self._last_y_root = 0
        
        # ドラッグ遅延用
        self._drag_pending_idx = None
        self._drag_start_job = None
        self._click_x = 0
        self._click_y = 0

        # 挿入線（tk.Frame を使用。CTkFrame は place の width 指定不可）
        self._marker = tk.Frame(self, height=3, bg="#3B8ED0")

    # ------------------------------------------------------------------
    # データ読み込み・再描画
    # ------------------------------------------------------------------

    def load_doc(self, doc):
        self.doc = doc
        self.refresh()

    def refresh(self):
        # insertion_marker 以外の子ウィジェットをすべて破棄
        for w in self.winfo_children():
            if w is not self._marker:
                w.destroy()
        self.thumbnails = []
        if not self.doc:
            return

        for i in range(len(self.doc)):
            # ハイライト色の決定 (薄い青色に変更)
            is_active = (i == self.active_idx)
            fg_color = ("#D1E8FF", "#2E5077") if is_active else "transparent"
            
            frame = ctk.CTkFrame(self, fg_color=fg_color)
            frame.pack(pady=10, padx=10, fill="x")

            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(frame, text=f"P.{i+1}", variable=var,
                                 font=("Meiryo", 10), width=20)
            cb.pack(side="top", anchor="nw", padx=2)

            page = self.doc[i]
            img = page.render(scale=0.12).to_pil()
            img_tk = ImageTk.PhotoImage(img)

            lbl = tk.Label(frame, image=img_tk, bg="gray30")
            lbl.image = img_tk
            lbl.pack(pady=2)

            # frame・lbl両方にバインドして広いドラッグ領域を確保
            for w in (frame, lbl):
                w.bind("<Button-1>",        lambda e, idx=i: self._on_press(e, idx))
                w.bind("<B1-Motion>",       self._drag_motion)
                w.bind("<ButtonRelease-1>", self._drag_end)

            # チェックボックスはドラッグ開始させない（frameへの伝播をブロック）
            cb.bind("<Button-1>",        lambda e: "break")
            cb.bind("<B1-Motion>",       lambda e: "break")
            cb.bind("<ButtonRelease-1>", lambda e: "break")

            self.thumbnails.append((frame, cb, var))

    # ------------------------------------------------------------------
    # ドラッグ処理
    # ------------------------------------------------------------------

    def _on_press(self, event, idx):
        self._drag_pending_idx = idx
        self._click_x = event.x_root
        self._click_y = event.y_root
        self.on_select_page(idx)
        # 400ms長押しでドラッグ開始
        self._drag_start_job = self.after(400, lambda: self._commit_drag_start(idx))

    def _commit_drag_start(self, idx):
        if self._drag_pending_idx == idx:
            self._drag_src = idx
            self._drag_slot = None
            self.thumbnails[idx][0].configure(fg_color="#333333")
            try:
                self.master.config(cursor="fleur")
            except Exception:
                pass
            self._drag_pending_idx = None
            self._drag_start_job = None

    def _drag_motion(self, event):
        """ドラッグ中：挿入線をマウス位置に追従させる"""
        if self._drag_src is None:
            if self._drag_pending_idx is not None:
                dx = abs(event.x_root - self._click_x)
                dy = abs(event.y_root - self._click_y)
                # ドラッグ確定前に大きく動かした場合は長押しをキャンセル
                if dx > 10 or dy > 10:
                    if self._drag_start_job:
                        self.after_cancel(self._drag_start_job)
                        self._drag_start_job = None
                    self._drag_pending_idx = None
            return

        # オートスクロール
        self._last_y_root = event.y_root
        y_rel = event.y_root - self.winfo_rooty()
        
        # オートスクロール方向の決定 (上端/下端 50px)
        canvas = self._parent_canvas
        y_rel = event.y_root - canvas.winfo_rooty()
        canvas_h = canvas.winfo_height()
        
        if y_rel < 50:
            self._scroll_dir = -1
        elif y_rel > canvas_h - 50:
            self._scroll_dir = 1
        else:
            self._scroll_dir = 0

        if self._scroll_dir != 0 and self._scroll_job is None:
            self._auto_scroll()

        slot = self._y_to_slot(event.y_root)
        self._drag_slot = slot
        self._show_marker(slot)

    def _drag_end(self, event):
        """ドラッグ終了：スロットを確定してページを移動"""
        if self._drag_start_job:
            self.after_cancel(self._drag_start_job)
            self._drag_start_job = None
        self._drag_pending_idx = None

        if self._drag_src is None:
            return

        # オートスクロール停止
        self._scroll_dir = 0
        if self._scroll_job:
            self.after_cancel(self._scroll_job)
            self._scroll_job = None

        src = self._drag_src
        slot = self._drag_slot

        # 状態を先にリセット（コールバック中の再入を防ぐ）
        self._drag_src = None
        self._drag_slot = None
        self._marker.place_forget()
        try:
            self.master.config(cursor="")
        except Exception:
            pass

        if slot is None:
            self.refresh()
            return

        n = len(self.thumbnails)

        # 移動なし判定
        if slot == src or slot == src + 1:
            self.refresh()
            return

        # スロット → PyMuPDF move_page(src, dst) の dst 変換
        # move_page内部: pop(src)後にinsert(dst, page)
        # → dst = pop後リストへの挿入インデックス
        # 末尾(slot==n): pop後リストサイズはn-1なので末尾はn-1
        # 先頭(slot==0): dst=0
        # src < slot: popでインデックスが1つ前にずれるため slot-1
        # src > slot: 補正不要で slot
        # PyMuPDF move_page(src, to) の仕様: 「toの前に挿入」
        # スロットs = 「s番ページの前」に挿入したい
        #
        # src > slot（前方向の移動）:
        #   popしてもslot以降のインデックスは変わらない → to = slot
        #
        # src < slot（後方向の移動）:
        #   pop(src)するとslot以降が1つ前にずれる
        #   「元のslot番の前」= pop後の「slot-1番の前」→ to = slot - 1
        #
        # slot=0（先頭）: to = 0（0番の前 = 先頭）
        # slot=n（末尾）: to = -1（PyMuPDF末尾指定）

        if slot == n:
            dst = -1
        elif src < slot:
            dst = slot          # 後ろ方向: srcが抜けてずれる分+1
        else:
            dst = slot          # 前方向: 補正不要

        self.on_move_page(src, dst)

    def _auto_scroll(self):
        """マウスが上下端にある間、自動でスクロールを継続する"""
        if self._scroll_dir == 0:
            self._scroll_job = None
            return

        vt, vb = self._parent_canvas.yview()
        
        # 境界チェックとスクロール実行
        can_scroll = False
        if self._scroll_dir == -1 and vt > 0:
            self._parent_canvas.yview_scroll(-8, "units")
            can_scroll = True
        elif self._scroll_dir == 1 and vb < 1:
            self._parent_canvas.yview_scroll(8, "units")
            can_scroll = True

        if not can_scroll:
            self._scroll_job = None
            return

        # 挿入マーカーの表示を更新（スクロールによりサムネイル位置が動くため）
        self.update_idletasks()
        slot = self._y_to_slot(self._last_y_root)
        self._drag_slot = slot
        self._show_marker(slot)

        # 次のスクロールをスケジュール
        self._scroll_job = self.after(50, self._auto_scroll)

    # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------

    def _y_to_slot(self, y_root):
        """
        画面上の y_root 座標をスロット番号(0〜len)に変換する。
        スロット i は「i番目のサムネイルの直前」を意味する。
        スロット len は「全サムネイルの直後（末尾）」。
        各サムネイルの中点より上 → そのサムネイル의 直前スロット(i)
        各サムネイルの中点より下 → そのサムネイル의 直後スロット(i+1)
        """
        if not self.thumbnails:
            return 0

        for i, (frame, _, _) in enumerate(self.thumbnails):
            fy = frame.winfo_rooty()
            fh = frame.winfo_height()
            if fh == 0:
                continue
            mid = fy + fh / 2
            if y_root <= mid:
                return i        # 中点より上 → スロットi（このサムネイルの直前）

        # 最後のサムネイルの中点より下 → 末尾スロット
        return len(self.thumbnails)

    def _show_marker(self, slot):
        """挿入線をスロット位置に表示する"""
        n = len(self.thumbnails)
        if n == 0:
            self._marker.place_forget()
            return

        slot = max(0, min(slot, n))

        if slot == 0:
            # 先頭サムネイルの上
            ref_frame = self.thumbnails[0][0]
            y = ref_frame.winfo_y() - 6
        elif slot == n:
            # 末尾サムネイルの下
            ref_frame = self.thumbnails[-1][0]
            y = ref_frame.winfo_y() + ref_frame.winfo_height() + 3
        else:
            # slot-1 の下と slot の上の中間
            f_above = self.thumbnails[slot - 1][0]
            f_below = self.thumbnails[slot][0]
            y_above = f_above.winfo_y() + f_above.winfo_height()
            y_below = f_below.winfo_y()
            y = (y_above + y_below) // 2

        ref_frame = self.thumbnails[0][0]
        x = ref_frame.winfo_x()
        w = ref_frame.winfo_width()
        self._marker.place(x=x, y=y, width=w)
        self._marker.lift()

    def get_selected_indices(self):
        return [i for i, (f, c, v) in enumerate(self.thumbnails) if v.get()]

    def set_all(self, value):
        for f, c, v in self.thumbnails:
            v.set(value)

    def invert_selection(self):
        """すべてのサムネイルの選択状態を反転する"""
        for f, c, v in self.thumbnails:
            v.set(not v.get())

    def set_checkbox_state(self, state):
        """チェックボックスの有効/無効を一括設定 (state='normal' or 'disabled')"""
        for f, c, v in self.thumbnails:
            c.configure(state=state)

    def highlight_page(self, idx):
        """指定したインデックスのサムネイルを強調表示し、必要ならスクロールする"""
        if not self.thumbnails or not (0 <= idx < len(self.thumbnails)):
            self.active_idx = -1
            return

        # 以前のハイライトをクリア
        if 0 <= self.active_idx < len(self.thumbnails):
            self.thumbnails[self.active_idx][0].configure(fg_color="transparent")
        
        # 新しいハイライト (薄い青色)
        self.active_idx = idx
        frame = self.thumbnails[idx][0]
        frame.configure(fg_color=("#D1E8FF", "#2E5077"))

        # 自動スクロール
        self.update_idletasks()
        canvas = self._parent_canvas
        
        # スクロール領域の取得
        region = canvas.cget("scrollregion").split()
        if not region: return
        total_h = float(region[3])
        
        if total_h > 0:
            frame_y = frame.winfo_y()
            frame_h = frame.winfo_height()
            canvas_h = canvas.winfo_height()
            
            # 現在の表示範囲
            vt, vb = canvas.yview()
            current_top = vt * total_h
            current_bottom = vb * total_h
            
            if frame_y < current_top or (frame_y + frame_h) > current_bottom:
                target_y = frame_y - (canvas_h / 2) + (frame_h / 2)
                new_v = max(0, min(1.0, target_y / total_h))
                canvas.yview_moveto(new_v)

# ===========================================================================
# Main Application
# ===========================================================================

class PDFEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"HGNN-editor v{VERSION}")
        self._set_window_icon()
        self.geometry("1280x768")

        self.manager = PDFManager()
        self.config = ensure_config(CONFIG_PATH)
        self.last_dir = self.config.get("last_editor_dir") or str(KB_DIR)
        if not os.path.isdir(self.last_dir):
            self.last_dir = str(KB_DIR)

        self.skip_selection_mode = False
        self.skip_selection_first_idx = None
        self._build_ui()
        self.bind("<Escape>", self.on_escape)

    def run_with_processing(self, func, *args, message="処理中...", **kwargs):
        dialog = ProcessingDialog(self, message=message)
        self.update()
        try:
            res = func(*args, **kwargs)
            return res
        finally:
            dialog.grab_release()
            dialog.destroy()
            self.update()

    def _set_window_icon(self):
        try:
            icon_path = Path(__file__).parent / "assets" / "icon_editor.png"
            if icon_path.exists():
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                self.after(200, lambda: self.iconphoto(False, photo))
                self._icon_photo = photo
        except Exception:
            pass

    def _build_ui(self):
        self.top_bar = ctk.CTkFrame(self)
        self.top_bar.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(self.top_bar, text="PDFを開く", command=self.open_file).pack(side="left", padx=5)
        self.lbl_filename = ctk.CTkLabel(self.top_bar, text="ファイルが選択されていません", font=("Meiryo", 12))
        self.lbl_filename.pack(side="left", padx=10)

        ctk.CTkButton(self.top_bar, text="保存", fg_color="SeaGreen", command=self.save_file).pack(side="right", padx=5)
        ctk.CTkButton(self.top_bar, text="アンドゥ", fg_color="gray50", command=self.undo).pack(side="right", padx=5)
        ctk.CTkButton(self.top_bar, text="リセット", fg_color="#CD5C5C", command=self.reset).pack(side="right", padx=5)

        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#2b2b2b", sashwidth=6)
        self.paned.pack(fill="both", expand=True, padx=10, pady=5)

        self.left_container = ctk.CTkFrame(self.paned, width=220)
        self.paned.add(self.left_container, stretch="never")

        # Skip selection message area
        self.skip_msg_frame = ctk.CTkFrame(self.left_container, fg_color="#CD5C5C")
        self.skip_msg_label = ctk.CTkLabel(self.skip_msg_frame, text="", text_color="white", font=("Meiryo", 11, "bold"))
        self.skip_msg_label.pack(pady=5)
        # Hidden by default

        self.sel_ctrl = ctk.CTkFrame(self.left_container)
        self.sel_ctrl.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(self.sel_ctrl, text="全選択", width=64, command=lambda: self.thumb_panel.set_all(True)).pack(side="left", padx=2)
        ctk.CTkButton(self.sel_ctrl, text="解除", width=64, command=lambda: self.thumb_panel.set_all(False)).pack(side="left", padx=2)
        ctk.CTkButton(self.sel_ctrl, text="反転", width=64, command=lambda: self.thumb_panel.invert_selection()).pack(side="left", padx=2)

        self.thumb_panel = ThumbnailPanel(self.left_container, on_select_page=self.jump_to_page, on_move_page=self.move_single_op)
        self.thumb_panel.pack(fill="both", expand=True, padx=5, pady=5)

        self.viewer = PDFViewer(self.paned, on_page_change=self.thumb_panel.highlight_page)
        self.paned.add(self.viewer, stretch="always")

        self.right_panel = ctk.CTkFrame(self.paned, width=220)
        self.paned.add(self.right_panel, stretch="never")

        # 1. Page Operations
        ctk.CTkLabel(self.right_panel, text="ページ操作", font=("Meiryo", 14, "bold")).pack(pady=(10, 5))
        page_ops_frame = ctk.CTkFrame(self.right_panel)
        page_ops_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(page_ops_frame, text="選択を移動", command=self.move_multi_dialog).pack(pady=5, fill="x", padx=10)
        ctk.CTkButton(page_ops_frame, text="選択をコピー", command=self.copy_multi_dialog).pack(pady=5, fill="x", padx=10)
        ctk.CTkButton(page_ops_frame, text="選択を削除", fg_color="#CD5C5C", command=self.delete_op).pack(pady=5, fill="x", padx=10)

        # 2. Image Operations (Merged Selection and Rotation)
        ctk.CTkLabel(self.right_panel, text="画像操作", font=("Meiryo", 14, "bold")).pack(pady=(20, 5))
        img_ops_frame = ctk.CTkFrame(self.right_panel)
        img_ops_frame.pack(fill="x", padx=10, pady=5)
        
        self.auto_clear_var = tk.BooleanVar(value=True)
        self.auto_clear_switch = ctk.CTkSwitch(img_ops_frame, text="自動解除 あり", variable=self.auto_clear_var, command=self._on_auto_clear_toggle)
        self.auto_clear_switch.pack(pady=5)
        
        self.skip_sel_btn = ctk.CTkButton(img_ops_frame, text="スキップ選択", command=self.skip_selection_start)
        self.skip_sel_btn.pack(pady=(10, 0), fill="x", padx=10)
        ctk.CTkLabel(img_ops_frame, text="マウスで指定したページを\n一つ置きに選択します。", font=("Meiryo", 10), text_color="gray").pack(pady=(0, 10))

        # Rotation & Crop buttons
        ctk.CTkButton(img_ops_frame, text="⟲ 左回転 (90°)", command=lambda: self.rotate_op(-90)).pack(pady=2, fill="x", padx=10)
        ctk.CTkButton(img_ops_frame, text="⟳ 右回転 (90°)", command=lambda: self.rotate_op(90)).pack(pady=2, fill="x", padx=10)
        ctk.CTkButton(img_ops_frame, text="反転 (180°)", command=lambda: self.rotate_op(180)).pack(pady=2, fill="x", padx=10)
        ctk.CTkButton(img_ops_frame, text="切り抜き (Crop)", fg_color="DarkSlateGray", command=self.crop_op).pack(pady=(10, 2), fill="x", padx=10)

        # 3. Insert PDF
        ctk.CTkLabel(self.right_panel, text="PDFを挿入", font=("Meiryo", 14, "bold")).pack(pady=(20, 10))
        insert_ops_frame = ctk.CTkFrame(self.right_panel)
        insert_ops_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(insert_ops_frame, text="先頭に追加", command=lambda: self.insert_op(0)).pack(pady=2, fill="x", padx=10)
        ctk.CTkButton(insert_ops_frame, text="現在位置に挿入", command=lambda: self.insert_op(self.viewer.current_page)).pack(pady=2, fill="x", padx=10)
        ctk.CTkButton(insert_ops_frame, text="最後に追加", command=lambda: self.insert_op(len(self.manager.doc) if self.manager.doc else 0)).pack(pady=2, fill="x", padx=10)


    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("対応ファイル", "*.pdf *.jpg *.jpeg *.png"),
                ("PDF files", "*.pdf"),
                ("画像ファイル", "*.jpg *.jpeg *.png"),
            ],
            initialdir=self.last_dir,
        )
        if path:
            self._update_last_dir(path)
            def _heavy():
                self.manager.load(path)
                self.refresh_ui(keep_page=False)
            self.run_with_processing(_heavy, message="ファイルを読み込み中...")

    def _update_last_dir(self, file_path):
        new_dir = str(Path(file_path).parent)
        self.last_dir = new_dir
        save_config(CONFIG_PATH, {"last_editor_dir": new_dir})

    def refresh_ui(self, keep_page=True, keep_selection=False):
        # 必要なら現在の選択状態を退避
        selected = []
        if keep_selection:
            selected = self.thumb_panel.get_selected_indices()

        current = self.viewer.current_page if keep_page else 0
        self.thumb_panel.load_doc(self.manager.doc)
        self.viewer.set_doc(self.manager.doc, current_page=current)
        
        # 選択状態を復元
        if keep_selection:
            for idx in selected:
                if 0 <= idx < len(self.thumb_panel.thumbnails):
                    self.thumb_panel.thumbnails[idx][2].set(True)

        main_name = self.manager.path.name if (self.manager and self.manager.path) else "ファイルが選択されていません"
        display_text = main_name
        if self.manager.last_added_filename:
            display_text += f" ({self.manager.last_added_filename})"
        self.lbl_filename.configure(text=display_text)

    def jump_to_page(self, idx):
        if self.skip_selection_mode:
            if self.skip_selection_first_idx is None:
                self.skip_selection_first_idx = idx
                self.thumb_panel.highlight_page(idx)
                self._set_skip_msg("【選択】 終了ページをクリック...")
                return
            else:
                start = min(self.skip_selection_first_idx, idx)
                end = max(self.skip_selection_first_idx, idx)
                # 一つ置きに選択
                for i in range(start, end + 1, 2):
                    if 0 <= i < len(self.thumb_panel.thumbnails):
                        self.thumb_panel.thumbnails[i][2].set(True)
                self.skip_selection_mode = False
                self.skip_selection_first_idx = None
                self.config(cursor="")
                self._set_skip_msg(None)
                self.thumb_panel.set_checkbox_state("normal")
                return
        self.viewer.current_page = idx
        self.viewer.show_page()

    def _on_auto_clear_toggle(self):
        if self.auto_clear_var.get():
            self.auto_clear_switch.configure(text="自動解除 あり")
        else:
            self.auto_clear_switch.configure(text="自動解除 なし")

    def _set_skip_msg(self, text):
        if text:
            self.skip_msg_label.configure(text=text)
            self.skip_msg_frame.pack(fill="x", padx=5, pady=(5, 0), before=self.sel_ctrl)
        else:
            self.skip_msg_frame.pack_forget()

    def skip_selection_start(self):
        self.skip_selection_mode = True
        self.skip_selection_first_idx = None
        self.config(cursor="cross")
        self._set_skip_msg("【選択】 開始ページをクリック...")
        self.thumb_panel.set_checkbox_state("disabled")

    def _clear_selection_if_needed(self):
        if self.auto_clear_var.get():
            self.thumb_panel.set_all(False)

    def _always_clear_selection(self):
        self.thumb_panel.set_all(False)

    def on_escape(self, event=None):
        """Escキーで操作をキャンセル"""
        if self.skip_selection_mode:
            self.skip_selection_mode = False
            self.skip_selection_first_idx = None
            self._set_skip_msg(None)
            self.thumb_panel.set_checkbox_state("normal")
            self.config(cursor="")
            return

        if self.viewer.mode == "crop":
            self.viewer.set_mode("view")
            self._set_skip_msg(None)
            self.config(cursor="")
            return

    def move_single_op(self, src, dst):
        def _heavy():
            self.manager.move_page(src, dst)
            if dst == -1:
                self.viewer.current_page = len(self.manager.doc) - 1
            elif src < dst:
                self.viewer.current_page = dst - 1
            else:
                self.viewer.current_page = dst
            self.refresh_ui()
            self._always_clear_selection()
        self.run_with_processing(_heavy, message="ページを移動中...")

    def move_multi_dialog(self):
        indices = self.thumb_panel.get_selected_indices()
        if not indices:
            messagebox.showinfo("情報", "ページを選択してください")
            return
        
        dlg = DestinationDialog(self, len(self.manager.doc))
        self.wait_window(dlg)
        if dlg.result is not None:
            def _heavy():
                self.manager.move_multiple_pages(indices, dlg.result)
                self.refresh_ui()
                self._always_clear_selection()
            self.run_with_processing(_heavy, message="ページを移動中...")

    def copy_multi_dialog(self):
        indices = self.thumb_panel.get_selected_indices()
        if not indices:
            messagebox.showinfo("情報", "ページを選択してください")
            return
        
        dlg = DestinationDialog(self, len(self.manager.doc), title="コピー先を指定")
        self.wait_window(dlg)
        if dlg.result is not None:
            def _heavy():
                self.manager.copy_pages(indices, dlg.result)
                self.refresh_ui()
                self._always_clear_selection()
            self.run_with_processing(_heavy, message="ページをコピー中...")

    def rotate_op(self, angle):
        indices = self.thumb_panel.get_selected_indices()
        if not indices: indices = [self.viewer.current_page]
        def _heavy():
            self.manager.rotate(indices, angle)
            keep = not self.auto_clear_var.get()
            self.refresh_ui(keep_selection=keep)
        self.run_with_processing(_heavy, message="回転処理中...")

    def crop_op(self):
        idx = self.viewer.current_page
        if self.manager.is_page_cropped(idx):
            # ダイアログを表示
            dlg = ctk.CTkToplevel(self)
            dlg.title("切り抜き設定")
            dlg.geometry("300x180")
            dlg.attributes("-topmost", True)
            
            ctk.CTkLabel(dlg, text="このページは既に切り抜かれています", font=("Meiryo", 11, "bold")).pack(pady=10)
            
            def on_restore():
                def _heavy():
                    self.manager.reset_crop([idx])
                    self.refresh_ui()
                self.run_with_processing(_heavy, message="切り抜きを復元中...")
                dlg.destroy()
                
            def on_reselect():
                dlg.destroy()
                self._start_crop_selection()

            ctk.CTkButton(dlg, text="元に戻す", command=on_restore).pack(pady=5, fill="x", padx=20)
            ctk.CTkButton(dlg, text="範囲をし直す", command=on_reselect).pack(pady=5, fill="x", padx=20)
            ctk.CTkButton(dlg, text="キャンセル", fg_color="gray", command=dlg.destroy).pack(pady=5, fill="x", padx=20)
        else:
            self._start_crop_selection()

    def _start_crop_selection(self):
        self._set_skip_msg("【切抜】 範囲をマウスで指定...")
        self.viewer.set_mode("crop", callback=self._on_crop_selected)

    def _on_crop_selected(self, rect):
        indices = self.thumb_panel.get_selected_indices()
        if not indices: indices = [self.viewer.current_page]
        
        def _heavy():
            self.manager.apply_crop(indices, rect)
            self._set_skip_msg(None)
            self.refresh_ui()
            self._clear_selection_if_needed()
        self.run_with_processing(_heavy, message="切り抜き処理中...")

    def delete_op(self):
        indices = self.thumb_panel.get_selected_indices()
        if not indices: indices = [self.viewer.current_page]
        if messagebox.askyesno("確認", f"{len(indices)} ページを削除しますか？"):
            def _heavy():
                self.manager.delete(indices)
                self.refresh_ui()
                self._always_clear_selection()
            self.run_with_processing(_heavy, message="ページを削除中...")

    def insert_op(self, pos):
        path = filedialog.askopenfilename(
            filetypes=[
                ("対応ファイル", "*.pdf *.jpg *.jpeg *.png"),
                ("PDF files", "*.pdf"),
                ("画像ファイル", "*.jpg *.jpeg *.png"),
            ],
            initialdir=self.last_dir,
        )
        if path:
            self._update_last_dir(path)
            def _heavy():
                self.manager.insert(path, pos)
                self.refresh_ui()
                self._always_clear_selection()
            self.run_with_processing(_heavy, message="ページを挿入中...")

    def undo(self):
        def _heavy():
            if self.manager.undo(): self.refresh_ui()
            else: messagebox.showinfo("情報", "アンドゥできる操作はありません")
        self.run_with_processing(_heavy, message="元に戻しています...")

    def reset(self):
        if messagebox.askyesno("確認", "最初に読み込んだ状態に戻しますか？"):
            def _heavy():
                if self.manager.reset(): self.refresh_ui(keep_page=False)
            self.run_with_processing(_heavy, message="初期状態にリセット中...")

    def save_file(self):
        if not self.manager.doc: return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", 
            filetypes=[("PDF files", "*.pdf")],
            initialdir=self.last_dir
        )
        if path:
            self._update_last_dir(path)
            self.manager.save(path)
            messagebox.showinfo("完了", "保存しました")

if __name__ == "__main__":
    app = PDFEditorApp()
    app.mainloop()
