"""Harrier embedding モデルの管理（ONNX Runtime 完全版）。

PyTorch / sentence-transformers を使用せず、onnxruntime と tokenizers のみで動作します。
モデルファイルは自動的に ../models/harrier-onnx にダウンロードされます。
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path

# Suppress requests and urllib3 version mismatch warnings
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except ImportError:
    pass

import numpy as np
import requests

_ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    _ONNX_AVAILABLE = True
except ImportError:
    pass

# Harrier は query 側に instruction prefix を付与することで精度が向上する
_QUERY_PREFIX = "Instruct: 類似テキスト検索\nQuery: "
EMBEDDING_DIM = 640

class EmbeddingManager:
    """ONNX Runtime で Harrier-OSS-v1-270M をロード・管理する。"""

    def __init__(self, model_name: str = "onnx-community/harrier-oss-v1-270m-ONNX"):
        self.model_name = model_name
        self.model_dir = Path(__file__).parent.parent / "models" / "harrier-onnx"
        self._session = None
        self._tokenizer = None

    @staticmethod
    def is_available() -> bool:
        """onnxruntime と tokenizers がインストールされているか。"""
        return _ONNX_AVAILABLE

    def is_loaded(self) -> bool:
        return self._session is not None

    def _download_file(self, url: str, dest: Path, status_callback=None):
        if dest.exists():
            return
        if status_callback:
            status_callback(f"ファイルをダウンロード中: {dest.name} ...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    def load_model(self, status_callback=None) -> bool:
        """モデルファイルをダウンロードしてロードする。"""
        if self._session is not None:
            return True
        if not _ONNX_AVAILABLE:
            if status_callback:
                status_callback("onnxruntime または tokenizers が未インストールです。")
            return False

        try:
            # 必要なファイルのリスト
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
            base_url = f"{hf_endpoint}/{self.model_name}/resolve/main"
            
            # 整合性チェックのための最小サイズ定義
            min_sizes = {
                "config.json": 100,
                "tokenizer.json": 1000000,          # ~2MB
                "tokenizer_config.json": 100,
                "model_q4f16.onnx": 100000,         # ~318KB
                "model_q4f16.onnx_data": 50000000,  # ~135MB
                "model_quantized.onnx": 50000000,
                "model.onnx": 50000000,
            }

            # 既存モデルファイルの探索と整合性/サイズ検証
            onnx_path = None
            for onnx_name in ["model_q4f16.onnx", "model_quantized.onnx", "model.onnx"]:
                candidate = self.model_dir / onnx_name
                if candidate.exists():
                    onnx_path = candidate
                    break

            is_valid = True
            if onnx_path is not None:
                # 検出されたモデルに応じた必要ファイルを定義してチェック
                required_names = ["tokenizer.json", "tokenizer_config.json", "config.json"]
                required_names.append(onnx_path.name)
                if onnx_path.name == "model_q4f16.onnx":
                    required_names.append("model_q4f16.onnx_data")
                
                for fname in required_names:
                    fpath = self.model_dir / fname
                    if not fpath.exists():
                        is_valid = False
                        break
                    
                    file_size = fpath.stat().st_size
                    min_sz = min_sizes.get(fname, 100)
                    if file_size < min_sz:
                        is_valid = False
                        break
            else:
                is_valid = False

            if not is_valid:
                # 整合性チェックに失敗した、またはファイルが存在しない場合は既存ファイルを全て削除して再ダウンロード
                if status_callback and onnx_path is not None:
                    status_callback("モデルファイルまたは構成ファイルの破損・未完了が検出されました。再ダウンロードを開始します...")
                
                # 古い/破損した可能性があるファイルをクリーンアップ
                for clean_name in ["model_q4f16.onnx", "model_q4f16.onnx_data", "model_quantized.onnx", "model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json"]:
                    bad_file = self.model_dir / clean_name
                    if bad_file.exists():
                        try:
                            bad_file.unlink()
                        except Exception:
                            pass
                
                onnx_path = None

            if onnx_path is None:
                # ダウンロードされていない、または検証失敗した場合はダウンロードを実行
                files = {
                    "model_q4f16.onnx": f"{base_url}/onnx/model_q4f16.onnx",
                    "model_q4f16.onnx_data": f"{base_url}/onnx/model_q4f16.onnx_data",
                    "tokenizer.json": f"{base_url}/tokenizer.json",
                    "tokenizer_config.json": f"{base_url}/tokenizer_config.json",
                    "config.json": f"{base_url}/config.json",
                }

                self.model_dir.mkdir(parents=True, exist_ok=True)

                for name, url in files.items():
                    self._download_file(url, self.model_dir / name, status_callback)

                onnx_path = self.model_dir / "model_q4f16.onnx"
                onnx_data_path = self.model_dir / "model_q4f16.onnx_data"
                
                # ダウンロード直後の最終サイズ検証
                if onnx_path.exists():
                    if onnx_data_path.exists():
                        check_file = onnx_data_path
                        min_size = min_sizes["model_q4f16.onnx_data"]
                    else:
                        check_file = onnx_path
                        min_size = min_sizes["model_q4f16.onnx"]
                        
                    file_size = check_file.stat().st_size
                    if file_size < min_size:
                        if status_callback:
                            status_callback(f"警告: {check_file.name} のダウンロードサイズが小さすぎます({file_size} bytes)。再試行します。")
                        if onnx_path.exists():
                            onnx_path.unlink()
                        if onnx_data_path.exists():
                            onnx_data_path.unlink()
                        self._download_file(files["model_q4f16.onnx"], onnx_path, status_callback)
                        if "model_q4f16.onnx_data" in files:
                            self._download_file(files["model_q4f16.onnx_data"], onnx_data_path, status_callback)

            if status_callback:
                status_callback("ONNX セッションを初期化中...")

            # パスを Windows の絶対パス形式に確実に変換
            abs_model_path = str(onnx_path.absolute())
            self._abs_model_path = abs_model_path

            # Harrier ONNX モデルは CUDAExecutionProvider の GQA (GroupQueryAttention) カーネルと互換性がなく、
            # テンソル拡張時の巨大なメモリ確保バグ (370GB超の要求) が発生するため、安定動作する CPUExecutionProvider を強制します。
            # ※このモデルは非常に軽量（270Mパラメータ）なため、CPU 実行でも数ミリ秒〜数十ミリ秒で超高速に動作します。
            providers = ["CPUExecutionProvider"]

            # ONNX Session の作成 (外部データサポートのためパス文字列を渡す)
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self._session = ort.InferenceSession(
                abs_model_path, 
                sess_options,
                providers=providers
            )


            # Tokenizer のロード
            self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            
            # Tokenizerのパディング設定 (右埋め)
            self._tokenizer.enable_padding(direction='right')
            self._tokenizer.enable_truncation(max_length=8192)

            if status_callback:
                status_callback("Embedding モデル (ONNX) の準備が完了しました。")
            return True

        except Exception as exc:
            if status_callback:
                status_callback(f"Embedding モデルのロードに失敗しました: {exc}")
            return False

    def encode_np(self, text: str, is_query: bool = False) -> np.ndarray | None:
        """テキストをベクトル化する (Torch不使用)"""
        if self._session is None or self._tokenizer is None:
            return None
            
        input_text = f"{_QUERY_PREFIX}{text}" if is_query else text
        
        try:
            # Tokenize
            encoded = self._tokenizer.encode(input_text)
            
            # ONNX 入力作成 (int64)
            ids = np.array([encoded.ids], dtype=np.int64)
            mask = np.array([encoded.attention_mask], dtype=np.int64)
            tids = np.array([encoded.type_ids], dtype=np.int64)
            
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": tids
            }
            
            # Harrier ONNX は token_type_ids を必要とする場合があります
            # モデルの入力名を確認して動的に対応
            model_inputs = [i.name for i in self._session.get_inputs()]
            actual_inputs = {k: v for k, v in inputs.items() if k in model_inputs}

            # Run 推論
            outputs = self._session.run(None, actual_inputs)
            
            # onnx-community のモデルは既にプーリング済みの sentence_embedding (batch_size, dim) を返す
            pooled = outputs[0][0]  # Take the first item in the batch -> shape: (dim,)

            # L2 Normalization
            norm = np.linalg.norm(pooled)
            normalized_embedding = pooled / np.clip(norm, a_min=1e-10, a_max=None)

            return normalized_embedding.astype(np.float32)

        except Exception:
            return None

    def encode_to_blob(self, text: str, is_query: bool = False) -> bytes | None:
        vec = self.encode_np(text, is_query=is_query)
        return vec.tobytes() if vec is not None else None

    @staticmethod
    def decode_blob(blob: bytes | None) -> np.ndarray | None:
        if blob is None: return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return dot / (norm_a * norm_b)

    def should_research(self, new_query: str, last_query: str, threshold: float = 0.65) -> bool:
        if not last_query or not self.is_loaded(): return True
        vec_new = self.encode_np(new_query, is_query=True)
        vec_old = self.encode_np(last_query, is_query=True)
        if vec_new is None or vec_old is None: return True
        return self.cosine_similarity(vec_new, vec_old) < threshold
