"""hgnn-ragchat / hgnn-ingestor / hgnn-search が共有する config_ai.txt の読み書き。

ファイルが無い、または既定キーが欠けている場合は ensure_config が追記・作成する。
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "あなたは親切なアシスタントです。日本語で丁寧に回答してください。"

# 全ツールで参照する既定値（欠損時に ensure で補完される）
DEFAULT_CONFIG: dict[str, str] = {
    "model_path": "../models",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_tokens": "2048",
    "temperature": "0.7",
    "n_ctx": "8192",
    "n_threads": "4",
    "chat_format": "chatml",
    "rag_db_path": "../db/knowledge.db",
    "rag_base_path": "../kb",
    "rag_top_k": "5",
    "rag_max_chars": "",
    "ranking_mode": "rrf",
    "embedding_model": "onnx-community/harrier-oss-v1-270m-ONNX",
    "topic_detection": "off",
    "last_model_dir": "../models",
    "last_kb_dir": "",
    "last_editor_dir": "",
    "interaction_mode": "separate",
    "gpu_backend": "cpu",
    "ocr_det_score_threshold": "0.2",
    "ocr_det_conf_threshold": "0.25",
    "ocr_det_iou_threshold": "0.2",
    "ocr_device": "auto",
    "ocr_scale": "2.0",
}

def _parse_file_keys_only(config_path: Path) -> set[str]:
    """ファイルに明示的に出現したキー（欠損検知用）。"""
    keys: set[str] = set()
    if not config_path.is_file():
        return keys
    with open(config_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _ = line.split("=", 1)
            keys.add(key.strip())
    return keys


def load_config(config_path: str | Path) -> dict[str, str]:
    """DEFAULT をベースにファイルの key=value を上書き。未知のキーはそのまま保持。"""
    path = Path(config_path)
    merged: dict[str, str] = dict(DEFAULT_CONFIG)
    if not path.is_file():
        return merged
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            merged[key.strip()] = value.strip()
    return merged


def _write_config_file(config_path: Path, config: dict[str, str]) -> None:
    """セクションコメント付きで保存（ingest / chat_rag / search で同一形式）。"""
    lines: list[str] = [
        "# config_ai.txt — hgnn-ragchat / hgnn-ingestor / hgnn-search で共有",
        "# 行頭が # の行はコメント。各設定は key=value（値に改行不可）。",
        "",
        "# --- LLM（hgnn-ragchat の gguf モデルと生成パラメータ）---",
    ]
    llm_keys = (
        "model_path",
        "system_prompt",
        "max_tokens",
        "temperature",
        "n_ctx",
        "n_threads",
        "chat_format",
        "gpu_backend",
    )
    for key in llm_keys:
        if key in config:
            lines.append(f"{key}={config[key]}")
    lines.extend(
        [
            "",
            "# --- RAG（SQLite DB パス・KB ルート・チャット検索の件数・スニペット長）---",
        ]
    )
    rag_keys = ("rag_db_path", "rag_base_path", "rag_top_k", "rag_max_chars", "ranking_mode", "embedding_model", "topic_detection")
    for key in rag_keys:
        if key in config:
            lines.append(f"{key}={config[key]}")
    lines.extend(
        [
            "",
            "# --- OCR（NDLOCR-Lite 設定項目）---",
        ]
    )
    ocr_keys = ("ocr_det_score_threshold", "ocr_det_conf_threshold", "ocr_det_iou_threshold", "ocr_device", "ocr_scale")
    for key in ocr_keys:
        if key in config:
            lines.append(f"{key}={config[key]}")
    lines.extend(
        [
            "",
            "# --- UI（ファイルダイアログの最終ディレクトリ・チャットの対話モード）---",
        ]
    )
    ui_keys = ("last_model_dir", "last_kb_dir", "last_editor_dir", "interaction_mode")
    for key in ui_keys:
        if key in config:
            lines.append(f"{key}={config[key]}")

    known = set(llm_keys) | set(rag_keys) | set(ocr_keys) | set(ui_keys)
    extra = sorted(k for k in config.keys() if k not in known)
    if extra:
        lines.extend(["", "# --- その他（手動で追加したキー）---"])
        for key in extra:
            lines.append(f"{key}={config[key]}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def save_config(config_path: str | Path, updates: dict[str, str] | None = None) -> None:
    """現在のファイル内容と既定値をマージし、updates を反映して保存する。"""
    path = Path(config_path)
    current = load_config(path)
    if updates:
        current.update(updates)
    _write_config_file(path, current)


def ensure_config(config_path: str | Path) -> dict[str, str]:
    """ファイルが無い、または DEFAULT のキーがファイルに無い場合に補完して保存する。"""
    path = Path(config_path)
    keys_in_file = _parse_file_keys_only(path)
    cfg = load_config(path)
    needs_save = not path.is_file() or any(k not in keys_in_file for k in DEFAULT_CONFIG)
    if needs_save:
        _write_config_file(path, cfg)
    return cfg


def scripts_dir() -> Path:
    """このモジュール（scripts）のディレクトリ。相対パス rag_* の基準にする。"""
    return Path(__file__).resolve().parent


def base_dir() -> Path:
    """リポジトリルート（scripts の親）。"""
    return scripts_dir().parent


def resolve_rag_db_path(config: dict[str, str]) -> Path:
    """rag_db_path を scripts 基準で絶対化。絶対パスはそのまま。"""
    raw = (config.get("rag_db_path") or DEFAULT_CONFIG["rag_db_path"]).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (scripts_dir() / path).resolve()


def resolve_rag_base_path(config: dict[str, str]) -> Path:
    """rag_base_path を scripts 基準で絶対化。未設定または空なら既定の ../kb を使う。"""
    raw = (config.get("rag_base_path") or "").strip()
    if not raw:
        raw = DEFAULT_CONFIG["rag_base_path"]
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (scripts_dir() / path).resolve()


def resolve_rag_base_if_set(config: dict[str, str]) -> Path | None:
    """rag_base_path が空なら None（ファイルリンク無効）。設定済みなら scripts 基準で絶対化。"""
    raw = (config.get("rag_base_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (scripts_dir() / path).resolve()


def find_newest_gguf(search_dir: str | Path) -> Path | None:
    """search_dir 内（直下）の .gguf ファイルのうち、最終更新日時が最も新しいものを返す。"""
    d = Path(search_dir)
    if not d.is_dir():
        return None
    try:
        ggufs = list(d.glob("*.gguf"))
        if not ggufs:
            return None
        # st_mtime でソートして最新のものを取得
        return max(ggufs, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None
