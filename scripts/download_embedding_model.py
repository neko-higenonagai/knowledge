"""
download_embedding_model.py
harrier-oss-v1-270m-ONNX (INT4) をダウンロードするスクリプト。
外部ライブラリ（huggingface_hubなど）に依存せず、標準ライブラリのみを使用する。
"""

import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO_ID = "onnx-community/harrier-oss-v1-270m-ONNX"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/"
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/model_q4f16.onnx",
    "onnx/model_q4f16.onnx_data"
]

def download_file(url, dest_path):
    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            # 大きなファイル対応のためチャンクで書き込む
            while True:
                chunk = response.read(8192 * 1024) # 8MB chunks
                if not chunk:
                    break
                out_file.write(chunk)
        print(f" -> OK ({os.path.basename(dest_path)})")
        return True
    except Exception as e:
        print(f" -> Failed ({os.path.basename(dest_path)}): {e}")
        return False

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models", "harrier-onnx")

    print(f"========================================")
    print(f"Harrier ONNX (INT4) Download Task")
    print(f"Target: {model_dir}")
    print(f"========================================")

    if not os.path.exists(model_dir):
        print(f"Creating directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)

    print("Downloading files... (This may take a few minutes)")
    
    # Minimum expected sizes to ensure we don't skip truncated/corrupted downloads
    min_sizes = {
        "config.json": 100,
        "tokenizer.json": 1000000,          # ~2MB actual
        "tokenizer_config.json": 100,
        "model_q4f16.onnx": 100000,         # ~318KB actual
        "model_q4f16.onnx_data": 50000000   # ~135MB actual
    }

    success_count = 0
    # INT4の外部データはサイズが大きいので、並列度を絞る
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for file_path in FILES:
            url = BASE_URL + file_path
            dest_name = os.path.basename(file_path)
            dest_path = os.path.join(model_dir, dest_name)
            
            if os.path.exists(dest_path):
                # 最小サイズチェックで破損/未完了ファイルを検出
                min_size = min_sizes.get(dest_name, 1)
                file_size = os.path.getsize(dest_path)
                if file_size >= min_size:
                    print(f"Skipping {dest_name} (already exists with valid size)")
                    success_count += 1
                    continue
                else:
                    print(f"File {dest_name} exists but is too small ({file_size} bytes < {min_size} bytes). Re-downloading...")
                    try:
                        os.unlink(dest_path)
                    except Exception:
                        pass
                
            futures.append(executor.submit(download_file, url, dest_path))
            
        for future in futures:
            if future.result():
                success_count += 1

    if success_count == len(FILES):
        print(f"\n[SUCCESS] All files downloaded successfully.")
        
        # 動作確認
        try:
            sys.path.insert(0, current_dir)
            from embedding_manager import EmbeddingManager
            model = EmbeddingManager()
            model.load_model()
            vec = model.encode_np("test", is_query=True)
            if vec is not None:
                print(f"Verification OK: vector norm={sum(vec**2)**0.5:.4f}")
            else:
                print("Verification failed: encode returned None")
        except Exception as e:
            print(f"Warning: Model downloaded but verification failed: {e}")
    else:
        print(f"\n[ERROR] Download failed for some files.")
        sys.exit(1)

if __name__ == "__main__":
    main()
