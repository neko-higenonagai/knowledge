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

import contextlib
import datetime
import hashlib
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk

# Suppress llama-cpp-python's low-level console output
os.environ["LLAMA_LOG_VERBOSITY"] = "0"
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk
from llama_cpp import Llama

from config_ai_common import (
    DEFAULT_SYSTEM_PROMPT,
    ensure_config,
    find_newest_gguf,
    resolve_rag_base_if_set,
    resolve_rag_db_path,
    save_config,
)
from embedding_manager import EmbeddingManager
from rag_ft_common import SudachiNounExtractor, fts_sync_single_record, embedding_sync_single_record, logical_path_parts, background_embedding_catchup

# UI settings
SHOW_FILE_OPEN_INFO = False


def parse_int_or_default(value, default):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_optional_int(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        number = int(text)
        return number if number > 0 else None
    except Exception:
        return None


def _find_word_boundary(text, pos, direction):
    """pos から direction (+1 or -1) 方向に単語境界を探す。
    日本語の句読点・括弧・空白を境界とみなす。"""
    length = len(text)
    delimiters = " \t、。，．・「」『』【】（）()[]{}…"
    if direction > 0:
        i = pos
        while i < length and text[i] not in delimiters:
            i += 1
        return min(i, length)
    else:
        i = pos
        while i > 0 and text[i - 1] not in delimiters:
            i -= 1
        return max(i, 0)


def extract_centered_text(text, search_tokens, max_chars):
    """Extract a snippet centered around search token hits.
    Behavior:
    - If max_chars is None or the text is shorter than max_chars, return the full text.
    - Find all hit positions of the search tokens (case‑insensitive).
    - If no hits, return the first max_chars characters (with ellipsis if truncated).
    - For hits, expand window to include all hits.
    - If hit span < max_chars, expand to max_chars and center.
    - If hit span >= max_chars, add max_chars // 4 context on each side.
    - Finally, cut the window at word boundaries and add leading/trailing ellipsis as needed.
    """
    def _clean(t):
        return t.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    if max_chars is None or len(text) <= max_chars:
        return _clean(text)

    if not search_tokens:
        end = _find_word_boundary(text, min(max_chars, len(text)), +1)
        return text[:end] + ("..." if end < len(text) else "")

    # Find all hit positions
    text_lower = text.lower()
    hit_positions = []
    for token in search_tokens:
        token_lower = token.lower()
        if not token_lower:
            continue
        start_pos = 0
        while True:
            idx = text_lower.find(token_lower, start_pos)
            if idx == -1:
                break
            hit_positions.append((idx, idx + len(token)))
            start_pos = idx + len(token)

    if not hit_positions:
        end = _find_word_boundary(text, min(max_chars, len(text)), +1)
        return text[:end] + ("..." if end < len(text) else "")

    # Sort hit positions
    hit_positions.sort()
    first_start = hit_positions[0][0]
    last_end = hit_positions[-1][1]
    span = last_end - first_start

    if span >= max_chars:
        # Hits span exceeds max_chars: use fixed margin on each side
        margin = max_chars // 4
        new_start = max(first_start - margin, 0)
        new_end = min(last_end + margin, len(text))
    else:
        # Hits span within max_chars: expand window to max_chars and center it
        extra = max_chars - span
        margin = extra // 2
        new_start = max(first_start - margin, 0)
        new_end = min(last_end + margin, len(text))
        
        # Adjust if we hit bounds
        if new_start == 0:
            new_end = min(max_chars, len(text))
        elif new_end == len(text):
            new_start = max(len(text) - max_chars, 0)

    # Align to word boundaries
    start = _find_word_boundary(text, new_start, -1) if new_start > 0 else 0
    end = _find_word_boundary(text, new_end, +1)

    snippet = _clean(text[start:end])
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + snippet + suffix


class IndexManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.is_ready = False
        self.lock = threading.Lock()
        self.extractor = SudachiNounExtractor()

    # 初期化の結果種別
    INIT_READY = "ready"          # インデックス最新
    INIT_NEED_FIRST = "first"     # rag_meta 未存在 → 初回構築が必要
    INIT_NEED_UPDATE = "update"   # レコード追加／変更 → 差分更新が必要
    INIT_NO_DB = "no_db"          # DB ファイル自体がない

    def check_init_status(self):
        """Determine initialization status using lightweight checks."""
        if not os.path.exists(self.db_path):
            return self.INIT_NO_DB

        if not self._fts_exists():
            return self.INIT_NEED_FIRST

        # Check for unsynced records
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            row = cursor.execute("SELECT 1 FROM ocr_texts WHERE fts_synced = 0 LIMIT 1").fetchone()
            if row:
                return self.INIT_NEED_UPDATE
        finally:
            conn.close()

        return self.INIT_READY

    def initialize(self, status_callback=None):
        with self.lock:
            # DB ファイルの親ディレクトリを作成
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            self._ensure_support_tables()
            
            # FTS がない場合は初期状態とするが、初期化自体は成功させる
            if not self._fts_exists():
                self.is_ready = False # 検索はできない
                return True, "データベースを準備しました。インデックスは未構築です。"

            self.is_ready = True
            return True, "インデックスの準備が完了しました。"


    def search(self, source_text, limit=5, cat1=None, cat2=None, cat3=None, forced_tokens=None, ranking_mode="hit_count", emb_manager=None):
        if not self.is_ready:
            return [], "", False

        is_simple = False
        if forced_tokens:
            search_tokens = forced_tokens[:30]
            is_simple = True
        else:
            source = (source_text or "").strip()
            if not source:
                return [], "", False
            tokens = self.extractor.extract_nouns(source, noun_only=True)
            if not tokens:
                search_tokens = [source]
            else:
                search_tokens = tokens[:30]
        
        query_text = " ".join(search_tokens)
        
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

        candidate_limit = max(int(limit) * 10, 30)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            results = []

            if is_simple:
                search_sql = [
                    "SELECT id AS doc_id, filename, page, path, text, embedding, 0.0 AS score",
                    "FROM ocr_texts",
                    "WHERE 1=1",
                ]
                search_params = []
                for token in search_tokens:
                    search_sql.append("AND text LIKE ?")
                    search_params.append(f"%{token}%")
                
                search_sql.append(filter_sql)
                search_params.extend(filter_params)

                search_sql.append("ORDER BY id DESC LIMIT ?")
                search_params.append(candidate_limit)
                
                for row in cursor.execute("\n".join(search_sql), search_params).fetchall():
                    item = dict(row)
                    parts = logical_path_parts(item.get("path"))
                    item["category1"] = parts[0] if len(parts) > 0 else ""
                    item["category2"] = parts[1] if len(parts) > 1 else ""
                    results.append(item)
            else:
                # 1. キーワード (FTS5) 検索候補の取得
                fts_candidates = []
                match_query = " OR ".join(self._quote_token(token) for token in search_tokens)
                if match_query:
                    search_sql = [
                        "SELECT t.id AS doc_id, f.filename, f.page, f.path, f.category1, f.category2, t.text, t.embedding, f.score",
                        "FROM (",
                        "    SELECT rowid, filename, page, path, category1, category2, bm25(ocr_texts_fts, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) AS score",
                        "    FROM ocr_texts_fts",
                        "    WHERE text MATCH ?",
                        ") AS f",
                        "JOIN ocr_texts AS t ON t.id = f.rowid",
                        "WHERE 1=1",
                    ]
                    search_params = [match_query]
                    search_sql.append(filter_sql.replace("path", "f.path").replace("filename", "f.filename"))
                    search_params.extend(filter_params)
                    search_sql.append("ORDER BY score ASC LIMIT ?")
                    search_params.append(candidate_limit)
                    
                    for row in cursor.execute("\n".join(search_sql), search_params).fetchall():
                        fts_candidates.append(dict(row))

                # 2. ベクトル (Embedding) セマンティック検索候補 of the candidate
                vector_candidates = []
                query_vec = None
                if emb_manager is not None and emb_manager.is_loaded():
                    query_vec = emb_manager.encode_np(source_text, is_query=True)
                    
                if query_vec is not None:
                    vec_sql = [
                        "SELECT id AS doc_id, filename, page, path, text, embedding",
                        "FROM ocr_texts",
                        "WHERE embedding IS NOT NULL"
                    ]
                    vec_params = []
                    vec_sql.append(filter_sql)
                    vec_params.extend(filter_params)
                    
                    all_rows = cursor.execute("\n".join(vec_sql), vec_params).fetchall()
                    
                    scored = []
                    for row in all_rows:
                        item = dict(row)
                        doc_vec = emb_manager.decode_blob(item.get("embedding"))
                        if doc_vec is not None:
                            sim = float(emb_manager.cosine_similarity(query_vec, doc_vec))
                            scored.append((item, sim))
                    
                    # 類似度の降順でソートし、上位件数を取得
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for item, sim in scored[:candidate_limit]:
                        parts = logical_path_parts(item.get("path"))
                        item["category1"] = parts[0] if len(parts) > 0 else ""
                        item["category2"] = parts[1] if len(parts) > 1 else ""
                        item["score"] = 0.0
                        item["vector_score"] = sim
                        vector_candidates.append(item)

                # 3. FTS と ベクトル検索結果を RRF で融合
                if ranking_mode == "rrf" and query_vec is not None:
                    merged_dict = {}
                    
                    # 順位付けマッチ
                    fts_rank = {c["doc_id"]: i + 1 for i, c in enumerate(fts_candidates)}
                    vector_rank = {c["doc_id"]: i + 1 for i, c in enumerate(vector_candidates)}
                    
                    # 二つの検索結果の和集合
                    all_c_dicts = {}
                    for c in fts_candidates:
                        all_c_dicts[c["doc_id"]] = c
                    for c in vector_candidates:
                        all_c_dicts[c["doc_id"]] = c
                    
                    k = 60
                    for doc_id, c in all_c_dicts.items():
                        r_fts = fts_rank.get(doc_id, len(fts_candidates) + 10)
                        r_vec = vector_rank.get(doc_id, len(vector_candidates) + 10)
                        
                        text = c["text"] or ""
                        hit_count = sum(text.lower().count(token.lower()) for token in search_tokens)
                        
                        rrf = 1.0 / (k + r_fts) + 1.0 / (k + r_vec)
                        c["rrf_score"] = rrf
                        c["hit_count"] = hit_count
                        merged_dict[doc_id] = c
                        
                    results = sorted(merged_dict.values(), key=lambda x: x["rrf_score"], reverse=True)
                else:
                    results = fts_candidates

            processed = []
            for item in results:
                text = item["text"] or ""
                # 表示用スコアの決定 (RRFスコア -> ベクトルスコア -> BM25スコア)
                score_val = 0.0
                if "rrf_score" in item:
                    score_val = item["rrf_score"]
                elif "vector_score" in item:
                    score_val = item["vector_score"]
                elif item.get("score") is not None:
                    score_val = item["score"]

                processed.append(
                    {
                        "doc_id": str(item["doc_id"]),
                        "filename": item["filename"] or "Unknown",
                        "page": item["page"] if item["page"] is not None else "-",
                        "path": item["path"] or "",
                        "category1": item.get("category1", ""),
                        "category2": item.get("category2", ""),
                        "text": text,
                        "hit_count": sum(text.lower().count(token.lower()) for token in search_tokens),
                        "matched_token_count": sum(1 for token in search_tokens if token.lower() in text.lower()),
                        "score": score_val,
                        "rrf_score": item.get("rrf_score"),
                        "vector_score": item.get("vector_score"),
                        "embedding_blob": item.get("embedding"),
                    }
                )
            
            # ランキングモードに応じてソート
            if ranking_mode == "bm25" and not is_simple:
                # BM25 スコア昇順（FTS5 の bm25() は低いほど関連度が高い）
                processed.sort(key=lambda x: x["score"])
            elif ranking_mode == "rrf" and emb_manager is not None and emb_manager.is_loaded() and not is_simple:
                pass  # Skip _rrf_rerank because results are already sorted by RRF
            else:
                # デフォルト: ヒット数降順
                processed.sort(key=lambda x: x["hit_count"], reverse=True)
            
            return processed[: int(limit)], query_text, is_simple
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

    def _rrf_rerank(self, candidates, query_text, emb_manager, k=60):
        """RRF (Reciprocal Rank Fusion) でヒット数・BM25・ベクトル類似度を統合リランキング。"""
        if not candidates:
            return candidates

        # 1. ヒット数ランキング（降順）
        by_hit = sorted(candidates, key=lambda x: x["hit_count"], reverse=True)
        hit_rank = {id(c): i + 1 for i, c in enumerate(by_hit)}

        # 2. BM25 ランキング（昇順＝低いほど良い）
        by_bm25 = sorted(candidates, key=lambda x: x["score"])
        bm25_rank = {id(c): i + 1 for i, c in enumerate(by_bm25)}

        # 3. ベクトル類似度ランキング
        query_vec = emb_manager.encode_np(query_text, is_query=True)
        vec_rank = {}
        if query_vec is not None:
            scored = []
            for c in candidates:
                doc_vec = emb_manager.decode_blob(c.get("embedding_blob"))
                if doc_vec is not None:
                    sim = emb_manager.cosine_similarity(query_vec, doc_vec)
                else:
                    sim = -1.0  # embedding がない場合は最低順位
                scored.append((c, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            vec_rank = {id(c): i + 1 for i, (c, _) in enumerate(scored)}
        else:
            # クエリの embedding が取れない場合はベクトル順位を無視
            vec_rank = {id(c): len(candidates) for c in candidates}

        # 4. RRF スコア計算
        for c in candidates:
            cid = id(c)
            rrf = (
                1.0 / (k + hit_rank.get(cid, len(candidates)))
                + 1.0 / (k + bm25_rank.get(cid, len(candidates)))
                + 1.0 / (k + vec_rank.get(cid, len(candidates)))
            )
            c["rrf_score"] = rrf

        candidates.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return candidates

    def _ensure_support_tables(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # 基底テーブル
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
            # 管理用テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_files (
                    path TEXT,
                    filename TEXT,
                    mtime REAL,
                    size INTEGER,
                    UNIQUE(path, filename)
                )
            """)
            # メタデータ
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            # FTS5 仮想テーブル (unicode61 tokenizer 使用)
            # 注: すでに存在する場合はエラーにならないよう CREATE VIRTUAL TABLE IF NOT EXISTS を使用
            # embedding カラムの追加（マイグレーション）
            try:
                cursor.execute("ALTER TABLE ocr_texts ADD COLUMN embedding BLOB DEFAULT NULL")
            except sqlite3.OperationalError:
                pass

            try:
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
            except sqlite3.OperationalError:
                # FTS5 がサポートされていない環境などのエラーは無視するか適切に処理
                pass

            conn.commit()
        finally:
            conn.close()

    def _fts_exists(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ocr_texts_fts'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _calculate_source_digest(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            digest = hashlib.sha256()
            for row in cursor.execute(
                "SELECT id, filename, page, path, text FROM ocr_texts ORDER BY id"
            ):
                digest.update(str(row[0]).encode("utf-8"))
                digest.update((row[1] or "").encode("utf-8"))
                digest.update(str(row[2] if row[2] is not None else "").encode("utf-8"))
                digest.update((row[3] or "").encode("utf-8"))
                digest.update((row[4] or "").encode("utf-8"))
            return digest.hexdigest()
        finally:
            conn.close()

    def _load_meta_value(self, key):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            row = cursor.execute("SELECT value FROM rag_meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _save_meta_value(self, key, value):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO rag_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _quote_token(token):
        return '"' + token.replace('"', '""') + '"'

    def sync_fts_for_record(self, cursor, doc_id, filename, page, path, text):
        """1件のレコードをFTSインデックスに同期する。"""
        fts_sync_single_record(cursor, self.extractor, doc_id, filename, page, path, text)

    def update_record(self, doc_id, text, emb_manager=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Get metadata for FTS update
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
            embedding_sync_single_record(cursor, emb_manager, doc_id, text)
            conn.commit()
            return True
        finally:
            conn.close()



class TextEditDialog(ctk.CTkToplevel):
    def __init__(self, parent, title, initial_text, on_save):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x600")
        self.on_save = on_save
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.textbox = ctk.CTkTextbox(self, font=("Meiryo", 12), wrap="word")
        self.textbox.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.textbox.insert("1.0", initial_text)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=1, column=0, pady=(0, 10))
        
        ctk.CTkButton(btn_frame, text="保存", command=self._save, width=120).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="キャンセル", command=self.destroy, width=120, fg_color="gray").pack(side="left", padx=5)
        
        self.after(200, self.lift)
        self.focus()

    def _save(self):
        text = self.textbox.get("1.0", "end-1c")
        self.on_save(text)
        self.destroy()


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, config, on_save_callback):
        super().__init__(parent)
        self.title("設定")
        self.geometry("640x600")
        self.after(200, self.lift)
        self.focus()
        self.config = config
        self.on_save_callback = on_save_callback

        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text="モデルパス:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.model_path_var = tk.StringVar(value=config.get("model_path", ""))
        ctk.CTkEntry(self, textvariable=self.model_path_var).grid(
            row=0, column=1, columnspan=2, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(self, text="参照", width=70, command=self.browse_model).grid(
            row=0, column=3, padx=10, pady=10
        )

        ctk.CTkLabel(self, text="システムプロンプト:").grid(row=1, column=0, padx=10, pady=10, sticky="ne")
        self.system_prompt_text = ctk.CTkTextbox(self, height=100)
        self.system_prompt_text.grid(row=1, column=1, columnspan=3, padx=10, pady=10, sticky="ew")
        self.system_prompt_text.insert("1.0", config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

        ctk.CTkLabel(self, text="知識パス:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.db_path_var = tk.StringVar(value=config.get("rag_db_path", "../db/knowledge.db"))
        ctk.CTkEntry(self, textvariable=self.db_path_var).grid(
            row=2, column=1, columnspan=2, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(self, text="参照", width=70, command=self.browse_db_path).grid(
            row=2, column=3, padx=10, pady=10
        )

        ctk.CTkLabel(self, text="PDFベース:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.base_path_var = tk.StringVar(value=config.get("rag_base_path", ""))
        ctk.CTkEntry(self, textvariable=self.base_path_var).grid(
            row=3, column=1, columnspan=2, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(self, text="参照", width=70, command=self.browse_base_path).grid(
            row=3, column=3, padx=10, pady=10
        )

        ctk.CTkLabel(self, text="検索件数:").grid(row=4, column=0, padx=10, pady=10, sticky="e")
        frame_r4 = ctk.CTkFrame(self, fg_color="transparent")
        frame_r4.grid(row=4, column=1, columnspan=3, sticky="w")
        self.rag_k_var = tk.StringVar(value=config.get("rag_top_k", "5"))
        ctk.CTkEntry(frame_r4, textvariable=self.rag_k_var, width=100).pack(side="left", padx=(10, 20), pady=10)
        ctk.CTkLabel(frame_r4, text="コンテキスト長:").pack(side="left", padx=(10, 10), pady=10)
        self.n_ctx_var = tk.StringVar(value=config.get("n_ctx", "8192"))
        ctk.CTkEntry(frame_r4, textvariable=self.n_ctx_var, width=120).pack(side="left", padx=(0, 10), pady=10)

        ctk.CTkLabel(self, text="最大文字数:").grid(row=5, column=0, padx=10, pady=10, sticky="e")
        frame_r5 = ctk.CTkFrame(self, fg_color="transparent")
        frame_r5.grid(row=5, column=1, columnspan=3, sticky="w")
        self.rag_max_chars_var = tk.StringVar(value=config.get("rag_max_chars", ""))
        ctk.CTkEntry(frame_r5, textvariable=self.rag_max_chars_var, width=100).pack(side="left", padx=(10, 20), pady=(10, 2))
        ctk.CTkLabel(frame_r5, text="最大トークン数:").pack(side="left", padx=(10, 10), pady=(10, 2))
        self.max_tokens_var = tk.StringVar(value=config.get("max_tokens", "2048"))
        ctk.CTkEntry(frame_r5, textvariable=self.max_tokens_var, width=120).pack(side="left", padx=(0, 10), pady=(10, 2))

        ctk.CTkLabel(
            self,
            text="空欄なら各レコードの全文を使います",
            font=("Meiryo", 13),
            text_color="#333333",
            anchor="w",
            justify="left",
        ).grid(row=6, column=1, columnspan=3, padx=(10, 10), pady=(0, 8), sticky="w")

        ctk.CTkLabel(self, text="ランキングモード:").grid(row=7, column=0, padx=10, pady=10, sticky="e")
        self.ranking_mode_var = tk.StringVar(value=config.get("ranking_mode", "rrf"))
        self.ranking_mode_menu = ctk.CTkOptionMenu(
            self,
            variable=self.ranking_mode_var,
            values=["hit_count", "bm25", "rrf"],
            width=150,
        )
        self.ranking_mode_menu.grid(row=7, column=1, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(self, text="話題変更判定:").grid(row=8, column=0, padx=10, pady=10, sticky="e")
        frame_r8 = ctk.CTkFrame(self, fg_color="transparent")
        frame_r8.grid(row=8, column=1, columnspan=3, sticky="w")
        self.topic_detection_var = tk.StringVar(value=config.get("topic_detection", "off"))
        self.topic_detection_menu = ctk.CTkOptionMenu(
            frame_r8,
            variable=self.topic_detection_var,
            values=["auto", "off"],
            width=100,
        )
        self.topic_detection_menu.pack(side="left", padx=(10, 20), pady=10)
        ctk.CTkLabel(frame_r8, text="閾値:").pack(side="left", padx=(10, 10), pady=10)
        self.topic_threshold_var = tk.StringVar(value=config.get("topic_threshold", "0.65"))
        ctk.CTkEntry(frame_r8, textvariable=self.topic_threshold_var, width=120).pack(side="left", padx=(0, 10), pady=10)

        ctk.CTkLabel(
            self,
            text="自動で話題の切り替えを判断して検索しなおします。",
            font=("Meiryo", 13),
            text_color="#666666",
            anchor="w",
            justify="left",
        ).grid(row=9, column=1, columnspan=3, padx=(10, 10), pady=(0, 8), sticky="w")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=10, column=0, columnspan=4, pady=(20, 20))
        ctk.CTkButton(btn_frame, text="保存", command=self.save_settings, width=120).pack(
            padx=10
        )

    def browse_model(self):
        initial_dir = self.config.get("last_model_dir", "")
        if initial_dir and not os.path.isabs(initial_dir):
            initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), initial_dir))

        filename = filedialog.askopenfilename(
            parent=self,
            initialdir=initial_dir,
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        self.lift()
        self.focus()
        if not filename:
            return

        try:
            relative_path = os.path.relpath(filename, os.path.dirname(__file__))
            if not relative_path.startswith(".."):
                self.model_path_var.set(relative_path.replace("\\", "/"))
            else:
                self.model_path_var.set(filename.replace("\\", "/"))
            self.config["last_model_dir"] = os.path.dirname(filename).replace("\\", "/")
        except ValueError:
            self.model_path_var.set(filename.replace("\\", "/"))
            self.config["last_model_dir"] = os.path.dirname(filename).replace("\\", "/")

    def browse_db_path(self):
        initial_dir = self.db_path_var.get().strip() or self.config.get("rag_db_path", "")
        if initial_dir and not os.path.isabs(initial_dir):
            initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), initial_dir))
        if os.path.isfile(initial_dir):
            initial_dir = os.path.dirname(initial_dir)
        filename = filedialog.askopenfilename(
            parent=self,
            initialdir=initial_dir or None,
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")],
        )
        self.lift()
        self.focus()
        if filename:
            try:
                relative_path = os.path.relpath(filename, os.path.dirname(__file__))
                if not relative_path.startswith(".."):
                    self.db_path_var.set(relative_path.replace("\\", "/"))
                else:
                    self.db_path_var.set(filename.replace("\\", "/"))
            except ValueError:
                self.db_path_var.set(filename.replace("\\", "/"))

    def browse_base_path(self):
        initial_dir = self.base_path_var.get().strip() or self.config.get("rag_base_path", "")
        if initial_dir and not os.path.isabs(initial_dir):
            initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), initial_dir))
        folder = filedialog.askdirectory(parent=self, initialdir=initial_dir or None)
        self.lift()
        self.focus()
        if folder:
            self.base_path_var.set(folder.replace("\\", "/"))

    def save_settings(self):
        new_config = {
            "model_path": self.model_path_var.get(),
            "system_prompt": self.system_prompt_text.get("1.0", "end-1c"),
            "rag_db_path": self.db_path_var.get(),
            "rag_base_path": self.base_path_var.get(),
            "rag_top_k": self.rag_k_var.get(),
            "rag_max_chars": self.rag_max_chars_var.get(),
            "n_ctx": self.n_ctx_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "temperature": self.config.get("temperature", "0.7"),
            "n_threads": self.config.get("n_threads", "4"),
            "chat_format": self.config.get("chat_format", "chatml"),
            "interaction_mode": self.config.get("interaction_mode", "auto"),
            "last_model_dir": self.config.get("last_model_dir", "../models"),
            "last_kb_dir": self.config.get("last_kb_dir", ""),
            "ranking_mode": self.ranking_mode_var.get(),
            "topic_detection": self.topic_detection_var.get(),
            "topic_threshold": self.topic_threshold_var.get(),
            "embedding_model": self.config.get("embedding_model", "onnx-community/harrier-oss-v1-270m-ONNX"),
        }
        self.on_save_callback(new_config)
        self.destroy()

class ChatApp(ctk.CTk):
    def __init__(self, config_path):
        super().__init__()

        self.config_path = config_path
        self.config = ensure_config(config_path)
        self.title(f"HGNN-ragchat v{VERSION}")
        self._set_window_icon()

        self.geometry("1280x768")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.llm = None
        self.messages = []
        self.stop_requested = False
        self.is_thinking = False
        self.kb_visible = False
        self.last_search_query = ""
        self.last_search_total_hits = 0
        self.kb_link_targets = {}
        self.last_populated_kb_text = ""
        self._is_system_updating_kb = False
        self.chat_history = []
        self.search_history = []

        self._kb_index_root: Path | None = None
        self._kb_index_by_name: dict[str, Path] | None = None
        self._kb_index_by_stem: dict[str, Path] | None = None

        self.db_path = str(resolve_rag_db_path(self.config))
        self.rag = IndexManager(self.db_path)
        self.emb_manager = EmbeddingManager(self.config.get("embedding_model", "onnx-community/harrier-oss-v1-270m-ONNX"))

        self._build_header()
        self._build_filters()
        self._build_main_panes()
        self._build_input_area()
        self._init_messages()

        self.append_to_chat("システム", "インデックスとモデルを読み込んでいます...")

        threading.Thread(target=self.initial_load, daemon=True).start()

    def _set_window_icon(self):
        try:
            icon_path = Path(__file__).parent / "assets" / "icon_chat.png"
            if icon_path.exists():
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                self.after(200, lambda: self.iconphoto(False, photo))
                self._icon_photo = photo # Keep reference
        except Exception:
            pass

    def _build_header(self):
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.model_name = os.path.basename(self.config.get("model_path", "None"))
        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text=f"使用モデル: {self.model_name}",
            font=("Meiryo", 12, "bold"),
        )
        self.header_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # 「データベースを使う」スイッチは削除され、常に有効になります。

        self.kb_toggle_button = ctk.CTkButton(
            self.header_frame,
            text="検索結果 表示/非表示",
            width=160,
            command=self.toggle_kb,
            fg_color="SeaGreen",
        )
        self.kb_toggle_button.grid(row=0, column=3, padx=5, pady=5)

        self.reset_button = ctk.CTkButton(
            self.header_frame,
            text="会話をリセット",
            width=110,
            command=self.reset_chat,
            fg_color="#CD5C5C",
            hover_color="#B22222",
        )
        self.reset_button.grid(row=0, column=4, padx=5, pady=5)

        self.settings_button = ctk.CTkButton(
            self.header_frame,
            text="設定",
            width=90,
            command=self.open_settings,
            fg_color="#A9A9A9",
            hover_color="#888888",
        )
        self.settings_button.grid(row=0, column=5, padx=10, pady=5)

    def _build_filters(self):
        self.filter_frame = ctk.CTkFrame(self, fg_color="#e8edf2")
        self.filter_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 2))
        # 10 columns for 3 labels, 3 menus, clear btn
        for index in range(9):
            self.filter_frame.grid_columnconfigure(index, weight=0)
        self.filter_frame.grid_columnconfigure(9, weight=1)

        self._all_label = "すべて"
        self._filter_data = {"opts1": [], "opts2": [], "opts3": []}

        # Major
        self.filter_cat1_label = ctk.CTkLabel(self.filter_frame, text="大分類:", font=("Meiryo", 11))
        self.filter_cat1_label.grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
        self.filter_cat1_var = tk.StringVar(value=self._all_label)
        self.filter_cat1 = ctk.CTkOptionMenu(
            self.filter_frame,
            variable=self.filter_cat1_var,
            values=[self._all_label],
            font=("Meiryo", 11),
            width=160,
            command=self._on_cat1_change,
        )
        self.filter_cat1.grid(row=0, column=1, padx=(0, 10), pady=6, sticky="w")

        # Middle
        self.filter_cat2_label = ctk.CTkLabel(self.filter_frame, text="中分類:", font=("Meiryo", 11))
        self.filter_cat2_label.grid(row=0, column=2, padx=(0, 2), pady=6, sticky="e")
        self.filter_cat2_var = tk.StringVar(value=self._all_label)
        self.filter_cat2 = ctk.CTkOptionMenu(
            self.filter_frame,
            variable=self.filter_cat2_var,
            values=[self._all_label],
            font=("Meiryo", 11),
            width=160,
            command=self._on_cat2_change,
        )
        self.filter_cat2.grid(row=0, column=3, padx=(0, 10), pady=6, sticky="w")

        # Minor
        self.filter_cat3_label = ctk.CTkLabel(self.filter_frame, text="小分類:", font=("Meiryo", 11))
        self.filter_cat3_label.grid(row=0, column=4, padx=(0, 2), pady=6, sticky="e")
        self.filter_cat3_var = tk.StringVar(value=self._all_label)
        self.filter_cat3 = ctk.CTkOptionMenu(
            self.filter_frame,
            variable=self.filter_cat3_var,
            values=[self._all_label],
            font=("Meiryo", 11),
            width=160,
            command=self._on_cat3_change,
        )
        self.filter_cat3.grid(row=0, column=5, padx=(0, 10), pady=6, sticky="w")

        # File (Remaining) - keep it as a fallback or filename filter?
        # The user said "Major, Middle, Minor". I'll repurpose the 3rd one for files too.
        # But wait, if I have 3 levels, and a path is 4 levels deep, I might need more.
        # However, the user explicitly asked for 3 levels.
        
        self.filter_clear_btn = ctk.CTkButton(
            self.filter_frame,
            text="クリア",
            width=70,
            font=("Meiryo", 11),
            fg_color="#A9A9A9",
            hover_color="#888888",
            command=self._clear_filters,
        )
        self.filter_clear_btn.grid(row=0, column=6, padx=(0, 10), pady=6)

    def _build_main_panes(self):
        self.paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#f0f0f0", sashwidth=4)
        self.paned_window.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        self.chat_panel = ctk.CTkFrame(self.paned_window, fg_color="white")
        self.chat_panel.grid_rowconfigure(0, weight=1)
        self.chat_panel.grid_columnconfigure(0, weight=1)
        self.chat_display = ctk.CTkTextbox(
            self.chat_panel,
            font=("Meiryo", 14),
            state="disabled",
            wrap="word",
            border_spacing=10,
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.paned_window.add(self.chat_panel, stretch="always")

        self.kb_panel = ctk.CTkFrame(self.paned_window)
        self.kb_panel.grid_rowconfigure(2, weight=1)
        self.kb_panel.grid_columnconfigure(0, weight=1)

        self.kb_header_frame = ctk.CTkFrame(self.kb_panel, fg_color="transparent")
        self.kb_header_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.kb_header_frame.grid_columnconfigure(1, weight=1)

        self.kb_header_label = ctk.CTkLabel(
            self.kb_header_frame,
            text="検索結果",
            font=("Meiryo", 13, "bold"),
        )
        self.kb_header_label.grid(row=0, column=0, sticky="w")

        self.no_research_var = tk.BooleanVar(value=False)
        self.no_research_check = ctk.CTkCheckBox(
            self.kb_header_frame,
            text="再検索しない。編集してこの内容に質問する。",
            variable=self.no_research_var,
            font=("Meiryo", 13),
            checkbox_width=18,
            checkbox_height=18,
            command=self._on_no_research_toggled,
        )
        self.no_research_check.grid(row=0, column=1, padx=(10, 0), sticky="w")

        self.search_status_label = ctk.CTkLabel(
            self.kb_panel,
            text="検索クエリ: - / 検索結果: 0",
            font=("Meiryo", 11),
            anchor="w",
            justify="left",
        )
        self.search_status_label.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")

        self.kb_textbox = ctk.CTkTextbox(self.kb_panel, font=("Meiryo", 12), wrap="word", state="disabled")
        self.kb_textbox.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        self.kb_textbox._textbox.bind("<<Modified>>", self._on_kb_textbox_modified)

        self.kb_button_frame = ctk.CTkFrame(self.kb_panel, fg_color="transparent")
        self.kb_button_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.kb_button_frame.grid_columnconfigure(0, weight=1)

        self.kb_search_var = tk.StringVar()
        self.kb_search_var.trace_add("write", self._on_kb_search_var_changed)
        self.kb_search_entry = ctk.CTkComboBox(
            self.kb_button_frame,
            variable=self.kb_search_var,
            values=[],
        )
        self.kb_search_entry.set("")
        self.kb_search_entry.bind("<KeyRelease>", self._on_kb_search_var_changed)
        self.kb_search_entry.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.kb_search_entry.bind("<Return>", lambda _event: self.run_manual_search())

        self.kb_search_btn = ctk.CTkButton(
            self.kb_button_frame,
            text="再度検索",
            width=80,
            command=self.run_manual_search,
            state="disabled",
        )
        self.kb_search_btn.grid(row=0, column=1, padx=(0, 4))

        self.kb_inline_search_btn = ctk.CTkButton(
            self.kb_button_frame,
            text="枠内検索",
            width=80,
            command=self.run_inline_search,
            fg_color="#4682B4",
            hover_color="#2F5F8F",
            state="disabled",
        )
        self.kb_inline_search_btn.grid(row=0, column=2, padx=(0, 4))

        self.kb_search_clear_btn = ctk.CTkButton(
            self.kb_button_frame,
            text="クリア",
            width=70,
            command=self.clear_kb_search,
            fg_color="#A9A9A9",
            hover_color="#888888",
            state="disabled",
        )
        self.kb_search_clear_btn.grid(row=0, column=3)

    def _build_input_area(self):
        self.input_parent_frame = ctk.CTkFrame(self)
        self.input_parent_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 15))
        self.input_parent_frame.grid_columnconfigure(1, weight=1)

        self.indicator = ctk.CTkLabel(
            self.input_parent_frame,
            text="●",
            font=("Meiryo", 20),
            text_color="#FF0000",
            width=30,
        )

        self.input_entry = ctk.CTkComboBox(
            self.input_parent_frame,
            font=("Meiryo", 14),
            height=40,
            values=[],
        )
        self.input_entry.set("")
        self.input_entry.grid(row=0, column=1, padx=(5, 5), pady=10, sticky="ew")
        self.input_entry.bind("<Return>", lambda _event: self.send_message())

        self.send_button = ctk.CTkButton(
            self.input_parent_frame, text="送信", command=self.send_message, width=80, height=40
        )
        self.send_button.grid(row=0, column=2, padx=(5, 5), pady=10)

        self.stop_button = ctk.CTkButton(
            self.input_parent_frame,
            text="停止",
            command=self.request_stop,
            width=80,
            height=40,
            fg_color="#CD5C5C",
            hover_color="#B22222",
            state="disabled",
        )
        self.stop_button.grid(row=0, column=3, padx=(0, 10), pady=10)

        # ------------------------------------------------------------------ #
        # 注意書きラベル
        # ------------------------------------------------------------------ #
        self.notice_label = ctk.CTkLabel(
            self.input_parent_frame,
            text="AIは間違った回答をする場合があります。検索できないときは単語を「」で囲んでください。回答が的確でないときは会話をリセットまたは再検索しないをチェックしてください。",
            font=("Meiryo", 11),
            text_color="gray50",
        )
        self.notice_label.grid(row=1, column=1, columnspan=3, padx=5, pady=(0, 10), sticky="w")

    def _init_messages(self):
        self.messages = [{"role": "system", "content": self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)}]

    def initial_load(self):
        try:
            status = self.rag.check_init_status()

            if status == IndexManager.INIT_NO_DB:
                self.after(0, lambda: self.append_to_chat("システム", "データベースファイルがありません。新規に作成します。"))
                # そのまま initialize() に進むことで作成される

            if status == IndexManager.INIT_NEED_FIRST:
                self.after(0, lambda: self.append_to_chat(
                    "システム", "検索インデックスが未構築です。PDF読み込みアプリ を使用して PDF を読み込んでください。"
                ))
            
            elif status == IndexManager.INIT_NEED_UPDATE:
                try:
                    conn = sqlite3.connect(self.rag.db_path)
                    cursor = conn.cursor()
                    count_row = cursor.execute("SELECT COUNT(*) FROM ocr_texts WHERE fts_synced = 0").fetchone()
                    unsynced_count = count_row[0] if count_row else 0
                finally:
                    conn.close()
                msg = f"未同期のレコードが検出されました ({unsynced_count}件)。最新の状態にするには PDF読み込みアプリ を実行してください。"
                self.after(0, lambda: self.append_to_chat("システム", msg))

            # インデックスの初期化（フォルダ/テーブル作成含む）
            success, message = self.rag.initialize()
            if not success:
                self.after(0, lambda: self.append_to_chat("エラー", message))
            else:
                if "未構築" in message or status == IndexManager.INIT_NO_DB:
                    pass
                else:
                    self.after(0, lambda: self.append_to_chat("システム", "検索インデックスの準備が完了しました。"))

            self.initialize_model()

            # Embedding モデルのロードと完了後のキャッチアップ
            def load_and_catchup(status_cb):
                if self.emb_manager.load_model(status_cb):
                    background_embedding_catchup(self.db_path, self.emb_manager)

            threading.Thread(
                target=load_and_catchup,
                args=(lambda msg: self.after(0, lambda: self.append_to_chat("システム", msg)),),
                daemon=True
            ).start()
        except Exception as exc:
            self.after(0, lambda exc=exc: self.append_to_chat("エラー", f"初期化エラー: {exc}"))

    def initialize_model(self):
        try:
            model_path = self.config.get("model_path")
            if model_path and not os.path.isabs(model_path):
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))

            if self.llm is not None:
                old_llm = self.llm
                self.llm = None
                del old_llm

            if not model_path or not os.path.exists(model_path) or os.path.isdir(model_path):
                # 自動選択を試みる
                m_dir = self.config.get("last_model_dir", "../models").strip()
                if m_dir:
                    if not os.path.isabs(m_dir):
                        m_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), m_dir))
                    
                    auto_model = find_newest_gguf(m_dir)
                    if auto_model:
                        model_path = str(auto_model)
                        # scripts フォルダからの相対パスに変換を試みる
                        try:
                            rel_model = os.path.relpath(model_path, os.path.dirname(__file__))
                            if not rel_model.startswith(".."):
                                self.config["model_path"] = rel_model.replace("\\", "/")
                            else:
                                self.config["model_path"] = model_path.replace("\\", "/")
                        except ValueError:
                            self.config["model_path"] = model_path.replace("\\", "/")
                        
                        save_config(self.config_path, self.config)
                        self.after(0, lambda: self.append_to_chat("システム", f"モデルが未設定のため、最新のモデルを自動選択しました: {auto_model.name}"))
                    else:
                        self.after(0, lambda: self.append_to_chat("システム", "AIモデルが見つかりません。download_model.batを実行し、GGUF モデルをダウンロードしてから、設定からモデルを選択してください。"))
                        self.after(0, lambda: self.append_to_chat("システム", "モデルがないため、検索のみ実行可能です。"))
                        self.model_name = "None"
                        self.after(0, lambda: self.header_label.configure(text=f"使用モデル: {self.model_name}"))
                        return
                else:
                    self.after(0, lambda: self.append_to_chat("システム", "AIモデルが見つかりません。LM Studio等で GGUF モデルをダウンロードし、設定から選択してください。"))
                    self.after(0, lambda: self.append_to_chat("システム", "モデルがないため、検索のみ実行可能です。"))
                    self.model_name = "None"
                    self.after(0, lambda: self.header_label.configure(text=f"使用モデル: {self.model_name}"))
                    return

            # 設定から gpu_backend を読み取る
            gpu_backend = self.config.get("gpu_backend", "cpu").lower()
            target_layers = 0 if gpu_backend == "cpu" else -1

            gpu_success = False
            original_n_ctx = parse_int_or_default(self.config.get("n_ctx", 8192), 8192)
            loaded_n_ctx = original_n_ctx

            if target_layers != 0:
                # VRAM容量に合わせて段階的にダウングレードしてGPUロードを再試行する
                ctx_candidates = [original_n_ctx]
                if original_n_ctx > 4096:
                    ctx_candidates.append(4096)
                if original_n_ctx > 2048:
                    ctx_candidates.append(2048)
                
                # 重複を除外して順序を維持
                seen = set()
                retry_candidates = [x for x in ctx_candidates if not (x in seen or seen.add(x))]

                for ctx_val in retry_candidates:
                    try:
                        with open(os.devnull, "w") as fnull:
                            with contextlib.redirect_stderr(fnull):
                                self.llm = Llama(
                                    model_path=model_path,
                                    n_ctx=ctx_val,
                                    n_threads=parse_int_or_default(self.config.get("n_threads", 4), 4),
                                    n_gpu_layers=target_layers,
                                    chat_format=self.config.get("chat_format", "chatml"),
                                    verbose=False,
                                )
                        gpu_success = True
                        loaded_n_ctx = ctx_val
                        break
                    except Exception:
                        continue

            if not gpu_success:
                # GPUで全滅したか、最初からCPUロード指定の場合
                with open(os.devnull, "w") as fnull:
                    with contextlib.redirect_stderr(fnull):
                        self.llm = Llama(
                            model_path=model_path,
                            n_ctx=original_n_ctx,
                            n_threads=parse_int_or_default(self.config.get("n_threads", 4), 4),
                            n_gpu_layers=0,
                            chat_format=self.config.get("chat_format", "chatml"),
                            verbose=False,
                        )

            self.model_name = os.path.basename(model_path)
            self.after(0, lambda: self.header_label.configure(text=f"使用モデル: {self.model_name}"))
            if gpu_success:
                if loaded_n_ctx < original_n_ctx:
                    self.after(0, lambda: self.append_to_chat("システム", f"VRAM容量に合わせるため、コンテキスト長を自動的に {original_n_ctx} から {loaded_n_ctx} に縮小してGPUで読み込みました。"))
                else:
                    self.after(0, lambda: self.append_to_chat("システム", "モデルをGPUで読み込みました。"))
            else:
                self.after(0, lambda: self.append_to_chat("システム", "モデルをCPUで読み込みました。"))
        except Exception as exc:
            self.after(0, lambda exc=exc: self.append_to_chat("エラー", f"モデルエラー: {exc}"))
            self.llm = None
            self.model_name = "Error"
            self.after(0, lambda: self.header_label.configure(text=f"使用モデル: {self.model_name}"))

    def _load_filters(self):
        opts1, opts2, opts3 = self.rag.get_filter_options()
        self.after(0, lambda: self._apply_filter_options(opts1, opts2, opts3))

    def _apply_filter_options(self, opts1, opts2, opts3):
        self._filter_data["opts1"] = opts1
        self._filter_data["opts2"] = opts2
        self._filter_data["opts3"] = opts3

        def setup_menu(menu, var, opts, label_widget):
            names = [self._all_label] + [o[0] for o in opts]
            menu.configure(values=names)
            # 現在の値がリストにない場合は「すべて」に戻す
            if var.get() not in names:
                var.set(self._all_label)
            
            # 選択肢が「すべて」のみなら無効化（ただし大分類は常に有効）
            if menu != self.filter_cat1:
                if len(names) <= 1:
                    menu.configure(state="disabled")
                    label_widget.configure(text_color="gray")
                else:
                    menu.configure(state="normal")
                    label_widget.configure(text_color="black")

        setup_menu(self.filter_cat1, self.filter_cat1_var, opts1, self.filter_cat1_label)
        setup_menu(self.filter_cat2, self.filter_cat2_var, opts2, self.filter_cat2_label)
        setup_menu(self.filter_cat3, self.filter_cat3_var, opts3, self.filter_cat3_label)

    def _on_cat1_change(self, value):
        sel1 = next((o for o in self._filter_data["opts1"] if o[0] == value), None)
        opts1, opts2, opts3 = self.rag.get_filter_options(cat1=sel1)
        self._apply_filter_options(opts1, opts2, opts3)

    def _on_cat2_change(self, value):
        sel1 = next((o for o in self._filter_data["opts1"] if o[0] == self.filter_cat1_var.get()), None)
        sel2 = next((o for o in self._filter_data["opts2"] if o[0] == value), None)
        opts1, opts2, opts3 = self.rag.get_filter_options(cat1=sel1, cat2=sel2)
        self._apply_filter_options(opts1, opts2, opts3)

    def _on_cat3_change(self, value):
        # 小分類の変更では下位がないので再描画のみ（必要あれば）
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

    def _on_no_research_toggled(self):
        """チェックON→kb_textbox編集可、チェックOFF→編集不可。"""
        self._kb_textbox_restore_state()
        if getattr(self, "kb_header_label", None):
            if self.no_research_var.get():
                self.kb_header_label.configure(text="検索結果 (編集可)")
            else:
                self.kb_header_label.configure(text="検索結果")

    def _on_kb_textbox_modified(self, event=None):
        if self.kb_textbox._textbox.edit_modified():
            if not getattr(self, "_is_system_updating_kb", False):
                current_text = self._get_editable_context()
                last_text = getattr(self, "last_populated_kb_text", "")
                if current_text != last_text and bool(current_text.strip()):
                    if getattr(self, "kb_header_label", None):
                        if self.no_research_var.get():
                            self.kb_header_label.configure(text="検索結果 (編集可)")
                        else:
                            self.kb_header_label.configure(text="検索結果")
                        self._clear_all_highlights()
                
                # 手動編集時もクリアボタンの状態を更新する
                self._on_kb_search_var_changed()

            self.kb_textbox._textbox.edit_modified(False)

    def toggle_kb(self):
        if self.kb_visible:
            self.paned_window.forget(self.kb_panel)
            self.kb_visible = False
            return

        self.paned_window.add(self.kb_panel, stretch="always")
        self.kb_visible = True
        self.update_idletasks()
        
        current_width = self.winfo_width()
        half_width = current_width // 2
        if half_width < 200:
            half_width = 600
            
        # 直接サッシュを配置し、さらに after で再確認
        try:
            self.paned_window.sash_place(0, half_width, 0)
        except:
            pass
        self.after(100, lambda: self._ensure_kb_sash_position(half_width))

    def _ensure_kb_sash_position(self, target_x):
        if not self.kb_visible: return
        try:
            self.paned_window.sash_place(0, target_x, 0)
        except:
            pass

    def clear_kb_search(self):
        self.kb_search_entry.set("")
        self._kb_textbox_set_state("normal")
        self.kb_textbox.delete("1.0", "end")
        self._kb_textbox_restore_state()
        self._clear_all_highlights()
        self.search_status_label.configure(text="検索クエリ: - / 検索結果: 0")
        self.last_search_query = ""
        self.last_search_total_hits = 0
        self.no_research_var.set(False)
        self._on_no_research_toggled()

    def _on_kb_search_var_changed(self, *_args):
        if not hasattr(self, "kb_search_btn"):
            return
        has_text = bool(self.kb_search_entry.get().strip())
        btn_state = "normal" if has_text else "disabled"
        self.kb_search_btn.configure(state=btn_state)
        self.kb_inline_search_btn.configure(state=btn_state)

        # クリアボタンは、入力があるか、または検索結果がある場合に有効にする
        has_results = bool(self.kb_textbox.get("1.0", "end-1c").strip())
        clear_state = "normal" if (has_text or has_results) else "disabled"
        self.kb_search_clear_btn.configure(state=clear_state)

        if not has_text:
            self._clear_inline_highlights()

    # ------------------------------------------------------------------ #
    #  ハイライト関連ヘルパー
    # ------------------------------------------------------------------ #
    _HIGHLIGHT_TAG = "search_highlight"
    _HIGHLIGHT_INLINE_TAG = "inline_highlight"

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """各種全角・特殊ダブルクォートや「」を半角 " に統一する。"""
        for ch in ("\u201c", "\u201d", "\u201e", "\u301d", "\u301e", "\uff02", "「", "」"):
            text = text.replace(ch, '"')
        return text

    @staticmethod
    def _parse_search_tokens(query: str) -> list[str]:
        """
        クォート対応のトークン分割。半角・全角どちらの "" も認識する。
        　"Bグループ"  → ['Bグループ']
        　"B グループ" → ['B グループ']
        　東京 大阪    → ['東京', '大阪']
        閉じ忘れ等の不正入力はフォールバックで空白分割。
        """
        normalized = ChatApp._normalize_quotes(query)
        try:
            tokens = shlex.split(normalized, posix=True)
        except ValueError:
            tokens = normalized.split()
        return [t for t in tokens if t.strip()]

    @staticmethod
    def _extract_forced_tokens(query: str) -> list[str]:
        """クォートで囲まれたフレーズだけを抽出して返す。"""
        normalized = ChatApp._normalize_quotes(query)
        import re
        return [m.group(1) for m in re.finditer(r'"([^"]+)"', normalized) if m.group(1).strip()]

    @staticmethod
    def _strip_quoted_phrases(query: str) -> str:
        """クォートで囲まれた部分を除去して残りのテキストを返す（形態素解析用）。"""
        normalized = ChatApp._normalize_quotes(query)
        import re
        return re.sub(r'"[^"]*"', '', normalized).strip()

    def _apply_highlights(self, tokens, tag=_HIGHLIGHT_TAG, bg="#FFD700", fg="black"):
        """kb_textbox 内で tokens をすべて検索してタグ付けする。"""
        textbox = self.kb_textbox._textbox
        textbox.tag_remove(tag, "1.0", "end")
        textbox.tag_config(tag, background=bg, foreground=fg)
        if not tokens:
            return
        content = textbox.get("1.0", "end")
        content_lower = content.lower()
        for token in tokens:
            if not token:
                continue
            start_idx = 0
            token_lower = token.lower()
            while True:
                pos = content_lower.find(token_lower, start_idx)
                if pos == -1:
                    break
                line = content[:pos].count("\n") + 1
                col = pos - content[:pos].rfind("\n") - 1
                end_pos = pos + len(token)
                end_line = content[:end_pos].count("\n") + 1
                end_col = end_pos - content[:end_pos].rfind("\n") - 1
                textbox.tag_add(tag, f"{line}.{col}", f"{end_line}.{end_col}")
                start_idx = end_pos

    def _clear_inline_highlights(self):
        textbox = self.kb_textbox._textbox
        textbox.tag_remove(self._HIGHLIGHT_INLINE_TAG, "1.0", "end")

    def _clear_all_highlights(self):
        textbox = self.kb_textbox._textbox
        textbox.tag_remove(self._HIGHLIGHT_TAG, "1.0", "end")
        textbox.tag_remove(self._HIGHLIGHT_INLINE_TAG, "1.0", "end")

    def run_inline_search(self):
        """テキストエリア内を枠内検索してハイライト（インデックス再検索はしない）。"""

        query = self.kb_search_entry.get().strip()
        if not query:
            return
        tokens = self._parse_search_tokens(query)
        self._apply_highlights(tokens, tag=self._HIGHLIGHT_INLINE_TAG, bg="#90EE90", fg="black")
        # 最初のヒットまでスクロール
        textbox = self.kb_textbox._textbox
        first = textbox.tag_ranges(self._HIGHLIGHT_INLINE_TAG)
        if first:
            textbox.see(first[0])

    def reset_chat(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

        if not self.no_research_var.get():
            self._kb_textbox_set_state("normal")
            self.kb_textbox.delete("1.0", "end")
            self._kb_textbox_restore_state()
            self._clear_all_highlights()
            self.search_status_label.configure(text="検索クエリ: - / 検索結果: 0")
            self.last_search_query = ""
            self.last_search_total_hits = 0
            self.kb_search_entry.set("")

        self._init_messages()

    def append_to_chat(self, sender, text, end="\n", color=None):
        self.chat_display.configure(state="normal")
        
        if sender:
            # [あなた] の前に改行を入れる
            prefix = "\n" if sender == "あなた" else ""
            self.chat_display.insert("end", f"{prefix}[{sender}] ")
        
        content_start = self.chat_display.index("end-1c")
        self.chat_display.insert("end", text + end)
        content_end = self.chat_display.index("end-1c")
        
        if color:
            # color_ の後のIDをより一意にする
            tag_name = f"color_{content_start.replace('.', '_')}"
            self.chat_display.tag_add(tag_name, content_start, content_end)
            self.chat_display.tag_config(tag_name, foreground=color)

        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def open_settings(self):
        SettingsWindow(self, self.config, self.apply_settings)

    def apply_settings(self, new_config):
        reload_model = (
            new_config.get("model_path") != self.config.get("model_path")
            or new_config.get("n_ctx") != self.config.get("n_ctx")
        )
        reload_rag = new_config.get("rag_db_path") != self.config.get("rag_db_path")

        self._invalidate_kb_path_index()

        self.config.update(new_config)
        save_config(self.config_path, self.config)

        if reload_rag:
            db_path = str(resolve_rag_db_path(self.config))
            self.rag = IndexManager(db_path)
            threading.Thread(target=self.initial_load, daemon=True).start()
            return

        if reload_model:
            self.append_to_chat("システム", "モデルを再読み込みしています...")
            threading.Thread(target=self.initialize_model, daemon=True).start()
        else:
            self.append_to_chat("システム", "設定を保存しました。")

    def run_manual_search(self):
        if self.is_thinking:
            messagebox.showwarning("考え中", "現在応答を生成中です。\n完了後に操作してください。")
            return
        query = self.kb_search_entry.get().strip()
        if not query:
            self.append_to_chat("システム", "検索欄が空です。")
            return
        # Update history
        if query in self.search_history:
            self.search_history.remove(query)
        self.search_history.insert(0, query)
        self.search_history = self.search_history[:5]
        self.kb_search_entry.configure(values=self.search_history)

        threading.Thread(
            target=self._refresh_rag_results,
            args=(query,),
            kwargs={"announce_source": "検索欄"},
            daemon=True,
        ).start()

    def _disable_ui_while_thinking(self):
        """考え中は停止・終了以外の操作をすべてロックする。"""
        # ヘッダー
        self.kb_toggle_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.settings_button.configure(state="disabled")
        # フィルター
        self.filter_cat1.configure(state="disabled")
        self.filter_cat2.configure(state="disabled")
        self.filter_cat3.configure(state="disabled")
        self.filter_clear_btn.configure(state="disabled")
        # 検索結果パネル
        self.no_research_check.configure(state="disabled")
        self.kb_textbox.configure(state="disabled")
        self.kb_textbox._textbox.configure(state="disabled")
        # CTkEntry の内部 tk.Entry に直接 readonly をセット（キー入力も完全ブロック）
        self.kb_search_entry._entry.configure(state="readonly")
        self.kb_search_btn.configure(state="disabled")
        self.kb_inline_search_btn.configure(state="disabled")
        self.kb_search_clear_btn.configure(state="disabled")
        # 入力エリア（readonly にしてメッセージ表示、送信ボタン無効）
        self.input_entry._entry.configure(state="readonly")
        self.send_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def send_message(self):
        user_text = self.input_entry.get().strip()
        if not user_text:
            return
        if self.is_thinking:
            messagebox.showwarning("考え中", "現在応答を生成中です。\n停止ボタンを押すか、完了をお待ちください。")
            return

        self.input_entry.set("")
        # Note: ComboBox doesn't usually show a placeholder while disabled like this, 
        # but we can set the value.
        self.input_entry.set("考え中...操作はできません...（停止のみ）")
        # kb_textbox を disabled にする前に内容をスナップショット
        self._snapshot_kb_text = self._get_editable_context()
        self._disable_ui_while_thinking()
        self.append_to_chat("あなた", user_text)

        self.is_thinking = True
        self.indicator.grid(row=0, column=0, padx=(10, 0))
        self.animate_indicator()

        self.stop_requested = False
        # Update history
        if user_text in self.chat_history:
            self.chat_history.remove(user_text)
        self.chat_history.insert(0, user_text)
        self.chat_history = self.chat_history[:5]
        self.input_entry.configure(values=self.chat_history)

        threading.Thread(target=self.process_chat, args=(user_text,), daemon=True).start()

    def process_chat(self, user_text):
        try:
            context_text = ""
            source_message = ""

            # kb_textbox は送信時に disabled になるため、事前にキャプチャした値を使う
            current_editable = getattr(self, "_snapshot_kb_text", "")
            last_populated = getattr(self, "last_populated_kb_text", "")
            is_manually_edited = (current_editable != last_populated) and bool(current_editable.strip())

            no_research = getattr(self, "no_research_var", None) and self.no_research_var.get()

            if no_research:
                # チェックON：再検索しない。スナップショットの内容を参照。会話履歴はリセット。
                context_text = current_editable.strip()
                if is_manually_edited:
                    source_message = "再検索せずに編集された検索結果から回答します。"
                else:
                    source_message = "再検索せずに現在の検索結果から回答します。"
                # チェックONの質問は毎回独立。前回の会話履歴を引きずらないようリセット。
                self.messages = [{"role": "system", "content": self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)}]
            else:
                # チェックOFF：is_manually_edited を無視して必ず通常検索
                search_text = self.kb_search_entry.get().strip()
                if search_text:
                    context_text = self._refresh_rag_results(search_text, announce_source="検索欄")
                    if context_text:
                        source_message = "検索欄に入力があるので、検索結果から回答します。"
                else:
                    # 話題判定による再検索スキップ
                    skip_research = False
                    if self.config.get("topic_detection", "off") == "auto" and self.emb_manager.is_loaded():
                        threshold_val = 0.65
                        try:
                            threshold_val = float(self.config.get("topic_threshold", 0.65))
                        except ValueError:
                            pass
                        if not self.emb_manager.should_research(user_text, self.last_search_query, threshold=threshold_val):
                            skip_research = True
                    
                    if skip_research:
                        context_text = current_editable.strip()
                        source_message = "話題が継続しているため、前回の検索結果から回答します。"
                        self.after(0, lambda: self.append_to_chat("システム", "（話題継続により再検索をスキップしました）"))
                    else:
                        context_text = self._refresh_rag_results(user_text, announce_source="質問内容")
                        if context_text:
                            source_message = "検索されたものから回答します。"

            # 検索結果が得られなかった場合のフォールバック
            if not source_message:
                # disabled 中は kb_textbox.get() が空になるためスナップショットを使う
                fallback_text = current_editable if no_research else getattr(self, "_snapshot_kb_text", "")
                fallback_text = fallback_text.strip()
                if fallback_text:
                    source_message = "検索されなかったので、前の検索結果から回答します。"
                    context_text = fallback_text
                    self.after(0, self._clear_all_highlights)
            
            if source_message:
                color = "#FF0000" if ("検索されなかった" in source_message or "検索欄に入力があるので" in source_message or "再検索せずに" in source_message) else None
                self.after(0, lambda msg=source_message, c=color: self.append_to_chat("システム", msg, color=c))

            messages_to_send = self._build_messages_with_rag_system_prompt(context_text)
            messages_to_send.append({"role": "user", "content": user_text})

            self.messages.append({"role": "user", "content": user_text})
            if self.llm:
                self.after(0, lambda: self.append_to_chat("システム", "応答を生成中..."))
                self.generate_response(messages_to_send)
            else:
                self.after(0, lambda: self.append_to_chat("AI", "は答えられません。", color="red"))
                self.after(0, self.stop_status)
                self.after(0, self.reenable_ui)
        except Exception as exc:
            self.after(0, lambda exc=exc: self.append_to_chat("エラー", f"会話エラー: {exc}"))
            self.after(0, self.stop_status)
            self.after(0, self.reenable_ui)

    def _build_messages_with_rag_system_prompt(self, rag_context):
        base_prompt = self.config.get("system_prompt", DEFAULT_SYSTEM_PROMPT).strip()
        if rag_context:
            combined_prompt = (
                f"{base_prompt}\n\n"
                "【重要】\n"
                "以下に提供された [検索内容] の情報のみに基づいて回答してください。\n"
                "検索内容に記載がない情報については、絶対に推測で補わず、「検索内容には記載がありません」と回答してください。\n\n"
                "[検索内容]\n"
                f"{rag_context}"
            )
        else:
            combined_prompt = (
                f"{base_prompt}\n\n"
                "【重要】\n"
                "現在、関連する検索結果が見つかりませんでした。そのため、あなたの一般的な知識に基づいて回答してください。\n"
                "回答の冒頭には必ず「検索結果が見つからなかったため、AIの知識の範囲でお答えします。」と明記してください。"
            )

        messages_to_send = [{"role": "system", "content": combined_prompt}]
        for message in self.messages[1:]:
            messages_to_send.append(message.copy())
        return messages_to_send

    def _get_editable_context(self):
        return self.kb_textbox.get("1.0", "end-1c").strip()

    def _refresh_rag_results(self, search_source_text, announce_source):
        cat1, cat2, cat3 = self._get_active_filters()
        filter_bits = [v[0] for v in [cat1, cat2, cat3] if v]
        filter_suffix = f" / フィルタ: {' > '.join(filter_bits)}" if filter_bits else ""
        # クォート解析：""囲みのフレーズは形態素解析をバイパスして forced_tokens へ
        forced_tokens = self._extract_forced_tokens(search_source_text)
        if forced_tokens:
            announce_source += " / 単純検索"
            # 検索欄からの検索でクォートが1つでもある場合、他の単語もすべて単純検索対象にする
            if "検索欄" in announce_source:
                forced_tokens = self._parse_search_tokens(search_source_text)
        
        is_text_search = bool(forced_tokens)
        search_type_str = "テキスト検索中" if is_text_search else "インデックス検索中"

        self.after(
            0,
            lambda: self.append_to_chat(
                "システム",
                f"{search_type_str} ({announce_source}{filter_suffix})...",
            ),
        )
        plain_text = self._strip_quoted_phrases(search_source_text)

        results, query_text, is_simple = self.rag.search(
            plain_text,
            limit=parse_int_or_default(self.config.get("rag_top_k", 5), 5),
            cat1=cat1,
            cat2=cat2,
            cat3=cat3,
            forced_tokens=forced_tokens if forced_tokens else None,
            ranking_mode=self.config.get("ranking_mode", "rrf"),
            emb_manager=self.emb_manager
        )

        self.last_search_query = query_text
        self.last_search_total_hits = len(results)
        self.after(
            0,
            lambda: self.search_status_label.configure(
                text=f"検索クエリ: {self.last_search_query or '-'} / 検索結果: {self.last_search_total_hits}"
            ),
        )

        rag_max_chars = parse_optional_int(self.config.get("rag_max_chars", ""))
        display_tokens = self._parse_search_tokens(query_text) if query_text else []

        self.after(0, lambda items=results, max_chars=rag_max_chars, simple=is_simple, tokens=display_tokens: self.update_kb_display_results(items, max_chars, simple, tokens))

        if not results:
            self.after(0, lambda: self.append_to_chat("システム", "関連する検索結果は見つかりませんでした。"))
            self.after(0, self._clear_all_highlights)
            return ""

        hit_count = len(results)
        self.after(
            0,
            lambda count=hit_count: self.append_to_chat(
                "システム", f"検索完了。{count}件の検索結果が見つかりました。"
            ),
        )

        ranking_mode = self.config.get("ranking_mode", "rrf")
        
        # Determine dynamic scaling factor for BM25 score if diluted
        bm25_scale = 1.0
        if ranking_mode == "bm25":
            max_abs_score = max([abs(r.get("score", 0.0)) for r in results] or [0.0])
            if 0.0 < max_abs_score < 0.001:
                bm25_scale = 1000000.0

        context_parts = []
        for index, result in enumerate(results, start=1):
            clipped_text = extract_centered_text(result["text"], display_tokens, rag_max_chars)
            
            if ranking_mode == "bm25" and "score" in result:
                score_info = f"BM25: {-result['score'] * bm25_scale:.2f}"
            elif ranking_mode == "rrf" and "rrf_score" in result:
                score_info = f"RRF: {result['rrf_score']:.4f}"
            else:
                score_info = f"ヒット数: {result['hit_count']}"

            meta = f"【検索結果{index}】 ({score_info}) / {result['filename']} / P.{result['page']} / {result['path']}"
            context_parts.append(f"{meta}\n{clipped_text}")

        context_text = "\n\n---\n\n".join(context_parts).strip()
        return context_text

    def _kb_textbox_set_state(self, state):
        """kb_textbox と内部 _textbox の state を同時に設定するヘルパー。"""
        self.kb_textbox.configure(state=state)
        self.kb_textbox._textbox.configure(state=state)

    def _kb_textbox_restore_state(self):
        """チェック状態に応じて kb_textbox の編集可否を復元する。"""
        if self.no_research_var.get():
            self._kb_textbox_set_state("normal")
        else:
            self._kb_textbox_set_state("disabled")

    def update_kb_display(self, text):
        self._is_system_updating_kb = True
        try:
            if getattr(self, "kb_header_label", None):
                if self.no_research_var.get():
                    self.kb_header_label.configure(text="検索結果 (編集可)")
                else:
                    self.kb_header_label.configure(text="検索結果")
            self._kb_textbox_set_state("normal")
            self.kb_textbox.delete("1.0", "end")
            self.kb_textbox.insert("1.0", text)
            if not self.kb_visible:
                self.toggle_kb()
            self.last_populated_kb_text = self._get_editable_context()
        finally:
            self._is_system_updating_kb = False
            self._kb_textbox_restore_state()
            self._on_kb_search_var_changed()

    def update_kb_display_results(self, results, rag_max_chars, is_simple=False, search_tokens=None):
        self._is_system_updating_kb = True
        try:
            # 常にハイライトをクリア（新旧キーワードに関わらず一旦消去）
            self._clear_all_highlights()

            if results:
                if getattr(self, "kb_header_label", None):
                    if self.no_research_var.get():
                        self.kb_header_label.configure(text="検索結果 (編集可)")
                    else:
                        self.kb_header_label.configure(text="検索結果")
                self._kb_textbox_set_state("normal")
                self.kb_textbox.delete("1.0", "end")
                self.kb_link_targets = {}
                link_tags = [tag for tag in self.kb_textbox.tag_names() if tag.startswith("doc_link_")]
                if link_tags:
                    self.kb_textbox.tag_delete(*link_tags)

                ranking_mode = self.config.get("ranking_mode", "rrf")
                
                # Determine dynamic scaling factor for BM25 score if diluted
                bm25_scale = 1.0
                if ranking_mode == "bm25":
                    max_abs_score = max([abs(r.get("score", 0.0)) for r in results] or [0.0])
                    if 0.0 < max_abs_score < 0.001:
                        bm25_scale = 1000000.0

                for index, result in enumerate(results, start=1):
                    clipped_text = extract_centered_text(result["text"], search_tokens or [], rag_max_chars)
                    file_label = f"{result['filename']} / P.{result['page']}"
                    
                    if ranking_mode == "bm25" and "score" in result:
                        score_info = f"BM25: {-result['score'] * bm25_scale:.2f}"
                    elif ranking_mode == "rrf" and "rrf_score" in result:
                        score_info = f"RRF: {result['rrf_score']:.4f}"
                    else:
                        score_info = f"ヒット数: {result['hit_count']}"

                    header_prefix = f"【検索結果{index}】 ({score_info}) / "
                    header_suffix = f" / {result['path']}\n"

                    resolved_path = self.resolve_result_path(result)
                    has_file = resolved_path is not None

                    self.kb_textbox.insert("end", header_prefix)
                    link_start = self.kb_textbox.index("end-1c")
                    self.kb_textbox.insert("end", file_label)
                    link_end = self.kb_textbox.index("end-1c")
                    self.kb_textbox.insert("end", header_suffix)
                    self.kb_textbox.insert("end", clipped_text + "\n\n")

                    if has_file:
                        tag_name = f"doc_link_{index}"
                        self.kb_link_targets[tag_name] = result
                        self.kb_textbox.tag_add(tag_name, link_start, link_end)
                        self.kb_textbox.tag_config(tag_name, foreground="#1f6aa5", underline=True)
                        self.kb_textbox.tag_bind(tag_name, "<Button-1>", lambda _event, tag=tag_name: self.open_result_document(tag))
                        self.kb_textbox.tag_bind(tag_name, "<Button-3>", lambda _event, tag=tag_name: self.show_result_context_menu(_event, tag))
                        self.kb_textbox.tag_bind(tag_name, "<Enter>", lambda _event, tag=tag_name: self.on_link_enter(tag))
                        self.kb_textbox.tag_bind(tag_name, "<Leave>", lambda _event: self.on_link_leave())

                if not self.kb_visible:
                    self.toggle_kb()

                # 検索キーワードの強調表示（結果がある場合のみ）
                if self.last_search_query:
                    search_tokens = self._parse_search_tokens(self.last_search_query)
                    color = "#ADD8E6" if is_simple else "#FFD700"
                    self._apply_highlights(search_tokens, tag=self._HIGHLIGHT_TAG, bg=color, fg="black")
            else:
                # 検索結果が空の場合：
                # 1. パネルを閉じない（以前の状態を維持）
                # 2. テキストを消さない（以前の内容を維持）
                # 3. ハイライトは冒頭の _clear_all_highlights() で消去済み
                pass

            self.last_populated_kb_text = self._get_editable_context()
        finally:
            self._is_system_updating_kb = False
            self._kb_textbox_restore_state()
            self._on_kb_search_var_changed()

    def show_result_context_menu(self, event, tag_name):
        if self.is_thinking:
            messagebox.showwarning("考え中", "現在応答を生成中です。\n完了後に操作してください。")
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="ビューアーで開く", command=lambda: self.open_result_document(tag_name))
        menu.add_command(label="検索アプリで開く", command=lambda: self.open_in_search_app(tag_name))
        menu.add_command(label="編集アプリで開く", command=lambda: self.open_in_ingest_editor(tag_name))
        menu.add_command(label="編集する", command=lambda: self.edit_result_text(tag_name))
        menu.post(event.x_root, event.y_root)

    def open_in_search_app(self, tag_name):
        result = self.kb_link_targets.get(tag_name)
        if not result:
            return
        resolved_path = self.resolve_result_path(result)
        if not resolved_path:
            messagebox.showwarning("ファイル未検出", "ベースパス配下で該当ファイルを見つけられませんでした。")
            return
        
        page = result.get("page", 1)
        query = self.kb_search_entry.get().strip() or getattr(self, "last_search_query", "")
        
        # Get active filters
        cat1, cat2, cat3 = self._get_active_filters()
        
        script_path = os.path.join(os.path.dirname(__file__), "hgnn-search.py")
        cmd = [sys.executable, script_path, "--file", str(resolved_path), "--page", str(page)]
        if query:
            cmd.extend(["--query", query])
        
        if cat1:
            cmd.extend(["--cat1", cat1[0], "--cat1_folder", str(cat1[1])])
        if cat2:
            cmd.extend(["--cat2", cat2[0], "--cat2_folder", str(cat2[1])])
        if cat3:
            cmd.extend(["--cat3", cat3[0], "--cat3_folder", str(cat3[1])])
            
        subprocess.Popen(cmd)
        
        self._show_timed_info(f"検索アプリを開きます (P.{page}):\n{resolved_path.name}", duration_ms=3000)

    def open_in_ingest_editor(self, tag_name):
        result = self.kb_link_targets.get(tag_name)
        if not result:
            return
        resolved_path = self.resolve_result_path(result)
        if not resolved_path:
            messagebox.showwarning("ファイル未検出", "ベースパス配下で該当ファイルを見つけられませんでした。")
            return
        
        page = result.get("page", 1)
        # hgnn-ingestor.py は scripts ディレクトリにあるはず
        script_path = os.path.join(os.path.dirname(__file__), "hgnn-ingestor.py")
        
        # 修正モード (--file) で hgnn-ingestor.py を起動
        subprocess.Popen([sys.executable, script_path, "--file", str(resolved_path), "--page", str(page)])
        
        self._show_timed_info(f"編集画面を開きます (P.{page}):\n{resolved_path.name}", duration_ms=3000)

    def edit_result_text(self, tag_name):
        result = self.kb_link_targets.get(tag_name)
        if not result:
            return

        doc_raw = result.get("doc_id")
        try:
            doc_id = int(str(doc_raw).strip())
        except (TypeError, ValueError):
            messagebox.showerror("エラー", f"ドキュメント ID が無効です: {doc_raw!r}")
            return

        filename = result.get("filename", "Unknown")
        page = result.get("page", "-")
        initial_text = result.get("text", "")

        def on_save(new_text):
            if self.rag.update_record(doc_id, new_text):
                # Update the result in our memory too
                result["text"] = new_text
                messagebox.showinfo("成功", f"{filename} (P.{page}) のデータを更新しました。")
            else:
                messagebox.showerror("エラー", "データの更新に失敗しました。")

        title = f"編集: {filename} (P.{page})"
        TextEditDialog(self, title, initial_text, on_save)

    def open_result_document(self, tag_name):
        if self.is_thinking:
            messagebox.showwarning("考え中", "現在応答を生成中です。\n完了後に操作してください。")
            return
        result = self.kb_link_targets.get(tag_name)
        if not result:
            return

        resolved_path = self.resolve_result_path(result)
        if resolved_path is None:
            base = resolve_rag_base_if_set(self.config)
            if base is None or not base.exists():
                messagebox.showwarning("設定エラー", "ベースパスが設定されていないか、フォルダが見つかりません。設定画面からKBフォルダを指定してください。")
            else:
                messagebox.showwarning("ファイル未検出", "ベースパス配下で該当ファイルを見つけられませんでした。")
            return

        try:
            page = result.get("page")
            opened = False

            if resolved_path.suffix.lower() == ".pdf" and page not in (None, "-", "", 0):
                opened = self._open_pdf_at_page(resolved_path, page)

            if not opened:
                os.startfile(str(resolved_path))

            page_info = f"  (P.{page})" if (resolved_path.suffix.lower() == ".pdf" and page not in (None, "-", "", 0)) else ""
            self._show_timed_info(f"ファイルを開きます:{page_info}\n{resolved_path.name}", duration_ms=3000)
        except Exception as exc:
            messagebox.showerror("起動失敗", f"ファイルを開けませんでした: {exc}")

    def _open_pdf_at_page(self, pdf_path: Path, page) -> bool:
        """
        PDF をページ指定で開く。成功した場合 True を返す。
        試行順：
          1. Adobe Acrobat / Acrobat Reader  (/A "page=N")
          2. SumatraPDF                       (-page N)
          3. ブラウザ系 (Chrome / Edge / Firefox)  #page=N
        """
        try:
            page_num = int(str(page))
        except (ValueError, TypeError):
            return False

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

        # --- Browsers (Chrome / Edge / Firefox) ---
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

    def _show_timed_info(self, message, duration_ms=3000):
        """指定ミリ秒後に自動で閉じる情報ダイアログを表示する。"""
        if not SHOW_FILE_OPEN_INFO:
            return
        dialog = tk.Toplevel(self)
        dialog.title("ファイルを開く")
        dialog.resizable(False, False)
        dialog.attributes("-topmost", True)

        # 中央寄せ
        dialog.update_idletasks()
        w, h = 360, 110
        sx = self.winfo_rootx() + (self.winfo_width() - w) // 2
        sy = self.winfo_rooty() + (self.winfo_height() - h) // 2
        dialog.geometry(f"{w}x{h}+{sx}+{sy}")

        tk.Label(
            dialog,
            text=message,
            font=("Meiryo", 11),
            wraplength=320,
            justify="center",
            pady=10,
        ).pack(expand=True, fill="both")

        status_var = tk.StringVar(value="開くまでお待ちください。この画面は自動で閉じます。")
        tk.Label(dialog, textvariable=status_var, font=("Meiryo", 9), fg="#666666").pack(pady=(0, 8))

        def _countdown(ms_left):
            if not dialog.winfo_exists():
                return
            if ms_left <= 0:
                dialog.destroy()
                return
            dialog.after(200, lambda: _countdown(ms_left - 200))

        _countdown(duration_ms)

    def on_link_enter(self, tag_name):
        """リンクにマウスカーソルenterした時の処理"""
        if self.is_thinking:
            return
        result = self.kb_link_targets.get(tag_name)
        if result:
            resolved_path = self.resolve_result_path(result)
            if resolved_path:
                self.kb_textbox._textbox.config(cursor="hand2")
                self.search_status_label.configure(text=f"クリックでファイルを開きます: {resolved_path.name}")
            else:
                self.kb_textbox._textbox.config(cursor="arrow")
        else:
            self.kb_textbox._textbox.config(cursor="arrow")

    def on_link_leave(self):
        """リンクからマウスカーソルleaveした時の処理"""
        self.kb_textbox._textbox.config(cursor="xterm")
        self.search_status_label.configure(text=f"検索クエリ: {self.last_search_query or '-'} / 検索結果: {self.last_search_total_hits}")

    def _invalidate_kb_path_index(self) -> None:
        self._kb_index_root = None
        self._kb_index_by_name = None
        self._kb_index_by_stem = None

    def _ensure_kb_filename_index(self, base: Path) -> tuple[dict[str, Path], dict[str, Path]]:
        """KB 直下の全ファイルパスを1回だけ走査し、ファイル名検索を O(1) にする。"""
        if self._kb_index_root == base and self._kb_index_by_name is not None and self._kb_index_by_stem is not None:
            return self._kb_index_by_name, self._kb_index_by_stem

        by_name: dict[str, Path] = {}
        by_stem: dict[str, Path] = {}
        try:
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                if path.name not in by_name:
                    by_name[path.name] = path
                if path.stem not in by_stem:
                    by_stem[path.stem] = path
        except OSError:
            pass
        self._kb_index_root = base
        self._kb_index_by_name = by_name
        self._kb_index_by_stem = by_stem
        return by_name, by_stem

    def resolve_result_path(self, result):
        base = resolve_rag_base_if_set(self.config)
        if base is None or not base.exists():
            return None

        raw_path = (result.get("path") or "").replace("\\", "/")
        rel_dir = Path(*(part for part in raw_path.split("/") if part))
        filename = (result.get("filename") or "").strip()
        if not filename:
            return None

        direct_candidate = base / rel_dir / filename
        if direct_candidate.is_file():
            return direct_candidate

        parent_dir = base / rel_dir
        if parent_dir.is_dir():
            for p in parent_dir.iterdir():
                if not p.is_file():
                    continue
                if p.name == filename or p.stem == filename:
                    return p

        by_name, by_stem = self._ensure_kb_filename_index(base)
        if filename in by_name:
            return by_name[filename]
        if filename in by_stem:
            return by_stem[filename]
        return None

    def generate_response(self, messages_to_send):
        try:
            stream = self.llm.create_chat_completion(
                messages=messages_to_send,
                max_tokens=parse_int_or_default(self.config.get("max_tokens", 2048), 2048),
                temperature=float(self.config.get("temperature", 0.7)),
                stream=True,
            )
            self.after(0, lambda: self.append_to_chat("AI", "", end=""))
            full_response = ""

            for chunk in stream:
                if self.stop_requested:
                    self.after(0, lambda: self.append_to_chat(None, "\n[停止しました]"))
                    break
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")
                if not content:
                    continue
                full_response += content
                self.after(0, lambda value=content: self.append_to_chat(None, value, end=""))

            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})
            if not self.stop_requested:
                self.after(0, lambda: self.append_to_chat(None, "\n", end=""))
        except Exception as exc:
            self.after(0, lambda exc=exc: self.append_to_chat("エラー", f"エラー: {exc}"))
        finally:
            self.after(0, self.stop_status)
            self.after(0, self.reenable_ui)

    def request_stop(self):
        self.stop_requested = True
        self.append_to_chat("システム", "停止を受け付けました...")
        self.stop_button.configure(state="disabled")

    def animate_indicator(self, state=0):
        if not self.is_thinking:
            return
        colors = ["#FF0000", "#CC0000", "#990000", "#660000", "#990000", "#CC0000"]
        self.indicator.configure(text_color=colors[state % len(colors)])
        self.after(200, lambda: self.animate_indicator(state + 1))

    def stop_status(self):
        self.is_thinking = False
        self.indicator.grid_forget()

    def reenable_ui(self):
        # ヘッダー
        self.kb_toggle_button.configure(state="normal")
        self.reset_button.configure(state="normal")
        self.settings_button.configure(state="normal")
        # フィルター
        self.filter_cat1.configure(state="normal")
        self.filter_cat2.configure(state="normal")
        self.filter_cat3.configure(state="normal")
        self.filter_clear_btn.configure(state="normal")
        # 検索結果パネル
        self.no_research_check.configure(state="normal")
        # チェック状態に応じて kb_textbox の編集可否を復元
        self._kb_textbox_restore_state()
        # ボタン状態を現在の入力欄と検索結果の内容に基づいて更新
        self._on_kb_search_var_changed()
        # 入力エリア
        self.kb_search_entry._entry.configure(state="normal")
        self.input_entry._entry.configure(state="normal")
        self.input_entry.set("")
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.input_entry.focus()



if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config_ai.txt"
    app = ChatApp(str(config_path))
    app.mainloop()
