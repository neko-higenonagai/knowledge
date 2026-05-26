import sqlite3
import os
import sys
from pathlib import Path

scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

from rag_ft_common import SudachiNounExtractor, fts_sync_single_record, embedding_sync_single_record
from config_ai_common import ensure_config, resolve_rag_db_path
from embedding_manager import EmbeddingManager

def rebuild_index():
    config_path = scripts_dir / "config_ai.txt"
    config = ensure_config(config_path)
    db_path = resolve_rag_db_path(config)
    
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Clearing FTS index...")
    cursor.execute("DELETE FROM ocr_texts_fts")
    
    print("Resetting sync status in ocr_texts...")
    cursor.execute("UPDATE ocr_texts SET fts_synced = 0")
    conn.commit()
    
    print("Extracting nouns and rebuilding index (this may take a while)...")
    extractor = SudachiNounExtractor()
    
    emb_manager = EmbeddingManager(config.get("embedding_model", "Xenova/harrier-oss-v1-270m"))
    if emb_manager.is_available():
        print("Loading embedding model for vector index rebuild...")
        emb_manager.load_model()
    else:
        print("sentence-transformers not available. Skipping vector index.")
    
    cursor.execute("SELECT id, filename, page, path, text FROM ocr_texts")
    rows = cursor.fetchall()
    total = len(rows)
    
    for i, row in enumerate(rows):
        doc_id, filename, page, path, text = row
        if i % 10 == 0:
            print(f"Processing {i}/{total}...")
        
        fts_sync_single_record(cursor, extractor, doc_id, filename, page, path, text)
        embedding_sync_single_record(cursor, emb_manager, doc_id, text)
        
        if i % 50 == 0:
            conn.commit()
            
    conn.commit()
    conn.close()
    print("Index rebuild completed successfully!")

if __name__ == "__main__":
    rebuild_index()
