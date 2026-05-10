"""OCR RAG で共有する Sudachi 名詞抽出・SQLite FTS5 同期（hgnn-search / hgnn-ingestor / hgnn-ragchat）。"""

from __future__ import annotations

import sqlite3

from sudachipy import dictionary, tokenizer as sudachi_tokenizer


def logical_path_parts(path: str | None) -> list[str]:
    """DB の path に \\ が含まれていても階層セグメントに分割する（空要素は除去）。"""
    if not path:
        return []
    normalized = path.replace("\\", "/")
    return [p for p in normalized.split("/") if p]


def fts_category1_category2(path: str | None) -> tuple[str, str]:
    parts = logical_path_parts(path)
    c1 = parts[0] if len(parts) > 0 else ""
    c2 = parts[1] if len(parts) > 1 else ""
    return c1, c2


class SudachiNounExtractor:
    """Sudachi が利用できない場合は extract_nouns がクエリそのものにフォールバックする。"""

    def __init__(self):
        try:
            self.tokenizer = dictionary.Dictionary().create()
            self.mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        except Exception as exc:
            print(f"Sudachi initialization error: {exc}")
            self.tokenizer = None

    def extract_nouns(self, text, noun_only=False):
        if not self.tokenizer:
            return [text] if text else []

        tokens = []
        seen = set()
        # 名詞に加えて動詞、形容詞も抽出対象とするが、noun_only=True の場合は名詞のみとする
        target_pos = {"名詞"} if noun_only else {"名詞", "動詞", "形容詞"}
        
        for morpheme in self.tokenizer.tokenize(text or "", self.mode):
            pos = morpheme.part_of_speech()
            if not pos or pos[0] not in target_pos:
                continue

            base = morpheme.dictionary_form()
            token = base if base and base != "*" else morpheme.surface()
            token = (token or "").strip()
            if not token:
                continue
            
            # 1文字の非ASCII文字（漢字・かな等）は許可する
            if len(token) == 1 and token.isascii():
                continue
                
            if all(ch in " 　\t\r\n-_/.,:;()[]{}<>!?\"'" for ch in token):
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return tokens


def fts_sync_single_record(
    cursor: sqlite3.Cursor,
    extractor: SudachiNounExtractor,
    doc_id: int,
    filename: str | None,
    page,
    path: str | None,
    text: str | None,
) -> None:
    """ocr_texts の1レコードから ocr_texts_fts に反映する。"""
    cleaned_text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if not cleaned_text:
        cursor.execute("DELETE FROM ocr_texts_fts WHERE rowid = ?", (doc_id,))
        return

    tokens = extractor.extract_nouns(cleaned_text)
    tokenized_text = " ".join(tokens)
    if not tokenized_text:
        tokenized_text = cleaned_text

    cat1, cat2 = fts_category1_category2(path)
    cursor.execute("DELETE FROM ocr_texts_fts WHERE rowid = ?", (doc_id,))
    cursor.execute(
        """
        INSERT INTO ocr_texts_fts(rowid, filename, page, path, category1, category2, text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            filename or "Unknown",
            str(page if page is not None else "-"),
            path or "",
            cat1,
            cat2,
            tokenized_text,
        ),
    )
    cursor.execute("UPDATE ocr_texts SET fts_synced = 1 WHERE id = ?", (doc_id,))

