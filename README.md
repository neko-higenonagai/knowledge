# HGNN(HiGeNagaNeko)のRAGナレッジベース

PDFドキュメントをNDLOCR-Lite（国立国会図書館）でOCRして、テキスト検索およびAIによる対話型検索を行うツールです。

<div align="center">
<img src="docs/hgnn-rag-tool.png" alt="HGNN tools" width="600">
</div>

NDLOCR-Lite（国立国会図書館）を使用することでOCRが以前より楽になりました。プログラムはAIに指示して作ったので、指示ミスによる動作不良があると思います。LLMによっては回答が適切でなかったり、回答がエンドレスで終わらなくて停止が必要な時もあります。PDFが頻繁に変更される用途に向いてないと思います。資料などの変更がないものが良いです。このツールは動作の正確性・完全性を保証するものではありません。使用は自己責任でお願いします。<br><br>
動作確認のためサンプルでkbフォルダにPDFが入ってます。新規に使用するときはフォルダ内をすべて削除して、推奨の構成でフォルダを作成してPDFを置いてください。PDFを取り込むときはデータベース（DB）を新規に作成しなおして下さい。<br><br>
検索キーワードと一致したヒット数、キーワードの出現頻度を考慮した統計的な検索（BM25）、キーワード検索とベクトル検索（埋め込みモデル）を組み合わせたハイブリッド検索（RRF）のいずれかを選ぶことができます。

## アプリケーション概要

本システムは、形態素解析に Sudachi と、ローカルベクトル埋め込み（Embedding）を用いたセマンティック検索を RRF (Reciprocal Rank Fusion: 相互順位融合) によってハイブリッド検索しています。これにより、ローカル環境で高速でありながら極めて精度の高いドキュメント検索を実現しています。

1. **HGNN-editor (`hgnn-editor.py`)**
   - PDFのページ構成を編集（並べ替え、削除、回転、結合）するためのツールです。
   - スキャン前の書類整理や、資料の統合に役立ちます。
   - 両面スキャン時の裏面の反転をスキップ選択による反転機能で戻します。
   - データベースの自動更新はできません。ingestorで再度更新します。

<div align="center">
<img src="docs/hgnn-editor.png" alt="Editor" width="300">
</div>

2. **HGNN-ingestor (`hgnn-ingestor.py`)**
   - PDFや画像ファイルを読み込み、OCR処理を行ってデータベースを構築します。
   - 新しい資料の追加や、既存データのメンテナンス（OCRテキストの修正）に使用します。

<div align="center">
<img src="docs/hgnn-ingestor.png" alt="Ingestor" width="300">
</div>

3. **HGNN-search (`hgnn-search.py`)**
   - 構築されたデータベースに対して、キーワードによる全文検索を行います。
   - 該当するPDFページをプレビューし、必要に応じてテキストの修正も可能です。

<div align="center">
<img src="docs/hgnn-search.png" alt="Search" width="300">
</div>

4. **HGNN-ragchat (`hgnn-ragchat.py`)**
   - AI（LLM）と対話しながらナレッジベースの内容を検索・要約します。
   - 関連するドキュメントを検索し、それに基づいた回答を生成します。
   - うまく検索しないときは「」または""で囲んでください。囲んだ文字はテキスト一致で検索します。
   - 前回の質問や回答したことを含めて会話が進みます。会話を進めると前回の回答の影響を受けます。回答が適切でないときは、会話をリセットしてください。
   - 自動で会話の内容の切り替わりを検出してデータベースから検索しなおします。

<div align="center">
<img src="docs/hgnn-ragchat.png" alt="RAGChat" width="300">
</div>

## 動作要件

> [!IMPORTANT]
> **推奨構成（CPU）およびGPU動作に関する重要な注意点**
>
> - **推奨環境**: 本システムは、最も安定して動作する **「CPU環境」を推奨構成（動作保証）** としています。特別な理由がない限り、セットアップ時は「CPU のみ」を選択しての導入を強くおすすめします。
> - **OCR処理・埋め込みのCPU固定**: 文字認識（NDLOCR-Lite）およびベクトル埋め込み（Embedding）は、GPUとライブラリの相性問題やメモリ確保の不具合を回避し、かつ極めて軽量で高速なため、GPU構成を選択した場合でも**常に安定した CPU 上で実行される設計**となっています。GPUによって高速化されるのは対話AI（LLM）の推論のみです。
> - **GPU環境の動作性**: GPUはPC構成やドライバのバージョンによってビルドエラーが発生したり、正常に動作しなかったりする場合があります。
> - **実機動作確認済みGPU**: 本システムにおいて実機で動作確認を行ったGPUは **「NVIDIA GeForce RTX 2060 (6GB)」のみ** です。
> - **未検証・サポート外の環境**: Intel Iris や AMD Radeon などの環境（DirectML / Vulkanアクセラレーション）については、実機での検証を行っておらず**未検証・サポート外**となります。

- Windows 11
- **[Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/) (必須)** — CPU/GPU の選択を問わず、セットアップ時の `llama-cpp-python` のコンパイルに必要です（「C++ によるデスクトップ開発」を有効にする必要があります）。
- [WinPython 3.12 (dot版)](https://winpython.github.io/) — インストール不要のポータブルPython環境
  - 本プロジェクトは WinPython の `python/python.exe` を使用して動作確認しています
  - 通常の Python 3.12 でも動作しますが、パスの設定が必要になる場合があります
- NDLOCR-Lite | Sudachi | llama-cpp-python | pypdfium2 | ONNX Runtime (Embedding)
  - ※ 文字認識モデル（`NDLOCR-Lite`）およびベクトル埋め込みモデル（`harrier-oss-v1-270m`）は、安全かつ安定した動作を確保するため、GPU環境であっても意図的に**CPU実行で動作するよう設計されています**（非常に軽量なため、CPUでも十分高速に動作し、クラッシュや不具合を防ぎます）。GPUによって高速化されるのは対話型AI（LLM）のみです。

## インストール

CodeからZIPファイルをダウンロードして解凍した後、下記のセットアップ用 BATファイル を実行してください。<br>
動作させるにはLLM（AIモデル）を別途ダウンロードする必要があります。

```bash
HGNN_setup.bat
```

[HGNN_setupの説明](HGNN_setup_guide.md)

## LLM(AIモデル)のダウンロード

モデルを自動ダウンロードするためのバッチファイルが2種類用意されています。用途やお好みに合わせてダウンロードしてください。また、`models` フォルダに別の GGUF モデルを直接配置すれば、アプリの設定画面からいつでも切り替え可能です。

### 1. Liquid LFM 2.5 (1.2B)

超軽量で高速動作する日本語対応モデル（約1.2GB）をダウンロードします。まずは手軽に試したい場合におすすめです。

```bash
download_model.bat
```

### 2. Google Gemma 4 E2B-it (5B)

最新の軽量高性能・高精度な instruction-tuned モデル（Q4_K_M版、約3.08GB）をダウンロードします。より高度な対話や要約を行いたい場合におすすめです。

```bash
download_gemma4_e2b.bat
```

---

## 使い方

### 1. PDFの整理

資料を整理したい場合は `HGNN-editor` を使用します。

- `run_editor.bat` を実行します。
- ページの入れ替え、画像の向きの補正、いらないページの削除などをして保存します。

### 2. データの取り込み

整理したPDFを`HGNN-ingestor` を使って資料を取り込みます。

- `run_ingestor.bat` を実行します。
- 取り込みたいPDFを選択し、「取り込み開始」を押すとOCR処理が始まります。
- フォルダの階層は後述の構成にするのが望ましいです。

### 3. 検索と閲覧

取り込まれたデータは `HGNN-search` で確認できます。

- `run_search.bat` を実行します。
- 検索窓に単語を入力すると、該当ページがリストアップされます。

### 4. AIとの対話

ナレッジベースに質問したい場合は `HGNN-ragchat` を使用します。

- `run_ragchat.bat` を実行します。
- 下部の入力欄に質問を入力すると、AIが関連資料を探して回答します。

## 推奨されるフォルダ構成

本システムはフォルダ階層を「大分類 / 中分類 / 小分類（ファイル名）」として認識します。
検索性を高めるため、以下のような **2階層のフォルダ構成** にした上で、各フォルダ内にPDFファイルを配置することを推奨します。

```text
kb (ベースフォルダ)
├── 01_プロジェクトA (大分類)
│   ├── 01_仕様書 (中分類)
│   │   └── 要件定義書.pdf (小分類/ファイル名)
│   └── 02_議事録 (中分類)
│       └── 2026-05-04_会議.pdf
└── 02_製品マニュアル (大分類)
    └── 01_取扱説明書 (中分類)
        └── ユーザーガイド.pdf
```

※ フォルダ階層が深すぎると取り込み対象外（範囲外）となる場合があります。原則としてベースフォルダから2階層下までにファイルを配置してください。

## ドキュメント

各ツールの使い方は `docs` 内の各説明書を参照してください。

- [hgnn-editor](docs/hgnn-editor_manual.md)
- [hgnn-ingestor](docs/hgnn-ingestor_manual.md)
- [hgnn-search](docs/hgnn-search_manual.md)
- [hgnn-ragchat](docs/hgnn-ragchat_manual.md)

## License

Copyright (c) 2026 HiGeNagaNeko
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Licenses

This software uses the following third-party libraries.

### llama-cpp-python

- **License**: MIT
- **Copyright**: Copyright (c) 2023 Andrei Betlen
- **URL**: https://github.com/abetlen/llama-cpp-python

### SudachiPy

- **License**: Apache 2.0
- **Copyright**: Copyright (c) 2017-2023 Works Applications Co., Ltd.
- **URL**: https://github.com/WorksApplications/SudachiPy

### SudachiDict-core

- **License**: Apache 2.0
- **Copyright**: Copyright (c) 2017-2023 Works Applications Co., Ltd.
- **URL**: https://github.com/WorksApplications/SudachiDict

### pypdfium2

- **License**: Apache-2.0 / BSD-3-Clause
- **Copyright**: Copyright (c) 2022 pypdfium2-team
- **URL**: https://github.com/pypdfium2-team/pypdfium2

### NDLOCR-Lite

- **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Copyright**: Copyright (c) 国立国会図書館 (National Diet Library of Japan)
- **URL**: https://github.com/ndl-lab/ndlocr-lite
- **Changes**: 変更なし

### ONNX Runtime

- **License**: MIT
- **Copyright**: Copyright (c) Microsoft Corporation. All rights reserved.
- **URL**: https://github.com/microsoft/onnxruntime

### CustomTkinter

- **License**: MIT
- **Copyright**: Copyright (c) 2021 Tom Schimansky
- **URL**: https://github.com/TomSchimansky/CustomTkinter

### harrier-oss-v1-270m-ONNX (Embedding Model)

- **License**: Apache 2.0
- **Copyright**: Copyright (c) Microsoft Corporation.
- **URL**: https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX
