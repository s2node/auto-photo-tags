## 📸 複数フォルダに保存された写真を **自動タグ付け** し、 **タグ検索** ができるローカルシステムの全体像

| 項目 | 内容 |
|------|------|
| 対象 | 任意のローカルフォルダに散在する JPEG/PNG などの画像 |
| 主な機能 | 1️⃣ 画像を自動で解析し **タグ候補** を生成  <br>2️⃣ 生成したタグを **SQLite** に保存（画像パス＋タグ） <br>3️⃣ コマンドライン／簡易Web UI で **タグ検索** → 該当画像のパスを一覧表示 |
| 使用技術 | - **llama.cpp** で **Gemma‑3‑2B/7B** 系統の LLM をローカル実行 <br>- 画像認識は **CLIP（ViT‑B/32）** か **BLIP‑image‑caption**（軽量版）で **キャプション生成** → そのテキストを LLM に渡して **タグ化** <br>- データ永続化は **SQLite** <br>- スクリプトは **Python 3.11+** （type‑hint, async） |
| 前提環境 | - Windows/macOS/Linux のいずれか <br>- CPU だけでも動作（GPU があれば **torch‑CUDA** で高速化） <br>- 8 GB 以上の空きディスク（モデル本体 ≈ 1‑3 GB） |

> **ポイント**  
> - `llama.cpp` は **GGML** 形式のモデルを高速に CPU 上で動かすことができ、Gemma‑3 系列（2B/7B）は 2‑3 GB のサイズです。  
> - 画像→テキスト（キャプション）生成は **CLIP** の zero‑shot classification でも可能ですが、**BLIP‑image‑caption** の方が自然言語的な説明を出力でき、LLM が「タグに変換」しやすくなります。  
> - 完全に **ローカル** で完結するので、プライバシーが守られます。

---

## 1️⃣ システム構成図

```
+-------------------+       +--------------------+       +-------------------+
|   フォルダスキャン | ----> |  画像→キャプション   | ----> |  LLM (Gemma‑3)    |
| (Python)          |       |  (BLIP / CLIP)      |       |  (llama.cpp)      |
+-------------------+       +--------------------+       +-------------------+
          |                         |                         |
          |   (キャプションテキスト) |   (タグ文字列)          |
          v                         v                         v
+---------------------------------------------------------------+
|                     SQLite DB (images.db)                    |
|   - id (PK)                                                   |
|   - path (TEXT)                                               |
|   - caption (TEXT)   <-- optional for debugging               |
|   - tags (TEXT)      <-- カンマ区切り (例: "beach, sunset")   |
+---------------------------------------------------------------+
          |
          |  (検索クエリ: タグ文字列)
          v
+-------------------+          +-------------------+
|   CLI / Flask UI  | <------> |  SQLite クエリ    |
+-------------------+          +-------------------+
```

---

## 2️⃣ 必要なソフト・ライブラリ

| カテゴリ | コマンド例 (Linux/macOS) |
|----------|--------------------------|
| **Python** | `python3 -m venv venv && source venv/bin/activate` |
| **基本ライブラリ** | `pip install pillow tqdm sqlalchemy tqdm` |
| **画像キャプション** | `pip install torch torchvision transformers==4.41.2` <br>（GPU がある場合は `torch --extra-index-url https://download.pytorch.org/whl/cu121`） |
| **llama.cpp** | ```bash<br>git clone https://github.com/ggerganov/llama.cpp<br>cd llama.cpp<br>make -j$(nproc)<br>``` |
| **Gemma‑3 モデル** | 1. HuggingFace から `gemma-2b-it`（または `gemma-7b-it`）をダウンロード <br>2. `ggml` へ変換 `python3 convert-hf-to-ggml.py gemma-2b-it`（リポジトリに同梱） |
| **SQLite** | Python 標準 `sqlite3` で OK |

> **備考**  
> - **BLIP‑image‑caption** は `transformers` の `blip-image-captioning-base` が軽量で 300 MB 程度です。  
> - **CLIP** の zero‑shot でも「画像の内容を短文に」できるので、環境が厳しい場合は代替可。

---

## 3️⃣ ディレクトリ構成（例）

```
photo_tagger/
│
├─ models/
│   ├─ ggml-gemma-2b-it.bin          # llama.cpp 用 GGML バイナリ
│   └─ blip-image-caption-base/      # Transformers 用キャプションモデル
│
├─ src/
│   ├─ tagger.py          # 画像走査・タグ生成・DB保存
│   ├─ search.py          # タグ検索 CLI
│   └─ webapp.py          # （任意）Flask UI
│
├─ images/                # 例：ユーザーが管理した画像フォルダ群
│   ├─ vacation/
│   └─ work/
│
└─ images.db              # SQLite DB（自動生成）
```

---

## 4️⃣ コード解説

以下は **最小構成** の 2 つのスクリプトです。  
- `tagger.py` : フォルダ走査 → キャプション生成 → LLM にタグ変換 → DB 保存  
- `search.py` : タグ検索 CLI  

> **※ ここでは `blip-image-caption-base` と `Gemma‑2B‑it`（llama.cpp）を使用**  
> **GPU が無い環境でも CPU で動作しますが、画像キャプションは 1 枚あたり 0.5‑2 秒程度です。**

### 4‑1️⃣ `src/tagger.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
photo_tagger/src/tagger.py

- 指定フォルダ以下の画像 (*.jpg, *.png, *.jpeg, *.webp) を再帰走査
- BLIP でキャプション生成
- llama.cpp (Gemma‑2B‑it) に「キャプション → タグ」プロンプトを投げる
- SQLite に (path, caption, tags) を保存
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, select, insert
from sqlalchemy.orm import Session

# --------------------------------------------------------------
# ① SQLite 用テーブル定義
# --------------------------------------------------------------
DB_PATH = "images.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
metadata = MetaData()

images_tbl = Table(
    "images",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("path", String, unique=True, nullable=False),
    Column("caption", String),          # デバッグ用に残す
    Column("tags", String),             # カンマ区切り文字列
)

metadata.create_all(engine)

# --------------------------------------------------------------
# ② 画像キャプション（BLIP） -----------------------------------------------------------------
# --------------------------------------------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)


def generate_caption(image_path: Path) -> str:
    """画像を読み込み、BLIP で英語キャプションを生成"""
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(DEVICE)

    out = blip_model.generate(**inputs, max_new_tokens=32)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


# --------------------------------------------------------------
# ③ LLM (Gemma‑2B‑it) でタグ生成 ------------------------------------
# --------------------------------------------------------------
LLAMA_CPP_EXE = Path("../llama.cpp/main")   # リポジトリの相対パス
GGML_MODEL = Path("../models/ggml-gemma-2b-it.bin")
LLAMA_TEMPERATURE = 0.2
LLAMA_MAX_TOKENS = 64


def _run_llama_cpp(prompt: str) -> str:
    """
    llama.cpp (Gemma) をサブプロセスで呼び出し、生成テキストだけ返す。
    - `--temp` は低めに設定し、決定的なタグを得る
    - `--n_predict` で最大トークン数を制限
    """
    cmd = [
        str(LLAMA_CPP_EXE),
        "-m", str(GGML_MODEL),
        "-p", prompt,
        "-n", str(LLAMA_MAX_TOKENS),
        "--temp", str(LLAMA_TEMPERATURE),
        "--no-keep",               # メモリ節約
        "--logits-all", "false",
        "--repeat_last_n", "64",
        "--repeat_penalty", "1.1",
        "--batch_size", "512",
    ]

    # llama.cpp は stdout に生成テキストだけ出すので、直接取得
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env=os.environ,
    )
    # 余計なログ行が混ざることがあるので、最初の空行以降の行を取得
    out_lines = result.stdout.strip().splitlines()
    # 例: "[[0.0, 0.0, ...]]\n\nbeach, sunset, sea"
    #   → 空行で区切られた最後の行が実際の出力
    for line in reversed(out_lines):
        if line.strip():
            return line.strip()
    return ""


def generate_tags(caption: str) -> List[str]:
    """
    LLM に対して「以下の英文キャプションから 3〜6 個のキーワードタグを
    カンマ区切りで出力してください。」というプロンプトを投げる。
    """
    prompt = (
        "You are a helpful AI that extracts concise keyword tags from an image description.\n"
        "Given the following description, output 3 to 6 short tags (lowercase, single words or short phrases) "
        "separated by commas, without any extra text.\n"
        "Description: \"" + caption + "\"\n"
        "Tags:"
    )
    raw = _run_llama_cpp(prompt)
    # 例: "beach, sunset, waves, summer"
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags


# --------------------------------------------------------------
# ④ メインロジック -------------------------------------------------
# --------------------------------------------------------------
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def find_images(root: Path) -> List[Path]:
    """再帰的に画像ファイルを列挙"""
    return [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXT and p.is_file()
    ]


def upsert_image(session: Session, path: str, caption: str, tags: List[str]) -> None:
    """path がすでに存在すれば更新、なければ挿入"""
    stmt = select(images_tbl).where(images_tbl.c.path == path)
    existing = session.execute(stmt).first()
    tags_str = ", ".join(tags)

    if existing:
        upd = (
            images_tbl.update()
            .where(images_tbl.c.path == path)
            .values(caption=caption, tags=tags_str)
        )
        session.execute(upd)
    else:
        ins = insert(images_tbl).values(path=path, caption=caption, tags=tags_str)
        session.execute(ins)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="画像フォルダを走査し、キャプション＋タグを生成して SQLite に保存"
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="タグ付け対象のフォルダ（複数可）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="すでに DB にある画像でも再処理（キャプション・タグ更新）",
    )
    args = parser.parse_args()

    # 1️⃣ 画像一覧取得
    all_images = []
    for folder in args.folders:
        if not folder.is_dir():
            print(f"[WARN] {folder} is not a directory, skip.")
            continue
        all_images.extend(find_images(folder))

    print(f"[INFO] {len(all_images)} images found in the given folders.")

    # 2️⃣ DB へ書き込み
    with Session(engine) as sess:
        for img_path in tqdm(all_images, desc="Tagging images"):
            img_str = str(img_path.resolve())

            # 既に登録済みか確認（force が無ければスキップ）
            if not args.force:
                stmt = select(images_tbl.c.id).where(images_tbl.c.path == img_str)
                if sess.execute(stmt).first():
                    continue

            try:
                caption = generate_caption(img_path)
                tags = generate_tags(caption)

                upsert_image(sess, img_str, caption, tags)
                sess.commit()
            except Exception as e:
                print(f"[ERROR] {img_str}: {e}", file=sys.stderr)
                sess.rollback()


if __name__ == "__main__":
    main()
```

#### 重要ポイント

| 行/セクション | 説明 |
|----------------|------|
| `DEVICE` | GPU があれば `cuda`、無ければ `cpu` に自動切替 |
| `generate_caption` | BLIP が出力する **英語** のキャプション（日本語でも可だが英語の方が LLM が扱いやすい） |
| `_run_llama_cpp` | `llama.cpp` のサブプロセス呼び出し。**標準出力の最後の非空行** が生成テキストとみなすので、余計なログは除外 |
| `generate_tags` | プロンプトは「短く、カンマ区切りで」だけを要求し、LLM が余計な説明を書かないように temperature を低く設定 |
| `upsert_image` | 同一パスが既に DB にあれば `UPDATE`、無ければ `INSERT`。`--force` オプションで再処理可能 |
| `tqdm` | 進捗バーで何枚処理したかが一目で分かる |

---

### 4‑2️⃣ `src/search.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
photo_tagger/src/search.py

- タグ（カンマ区切り・OR検索）で画像を検索
- 結果は端末にパス一覧で出力
- オプションで JSON 出力やサムネイル表示も可能
"""

import argparse
import json
from pathlib import Path
from sqlalchemy import create_engine, select, Table, MetaData, or_
from sqlalchemy.orm import Session

DB_PATH = "images.db"
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
metadata = MetaData()
images_tbl = Table("images", metadata, autoload_with=engine)


def build_filter(tag_list):
    """タグリスト (["beach", "sunset"]) を OR 条件に変換"""
    conditions = []
    for tag in tag_list:
        # SQLite の LIKE は大文字小文字を区別しない (NOCASE が設定されていれば)
        conditions.append(images_tbl.c.tags.like(f"%{tag.strip()}%"))
    return or_(*conditions)


def main():
    parser = argparse.ArgumentParser(
        description="タグ検索ツール（SQLite に保存された画像情報を検索）"
    )
    parser.add_argument(
        "query",
        type=str,
        help="検索したいタグ（カンマ区切り、例: beach,sunset）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果を JSON 形式で標準出力",
    )
    args = parser.parse_args()

    tags = [t.strip().lower() for t in args.query.split(",") if t.strip()]
    if not tags:
        print("[ERROR] No valid tags supplied.")
        return

    with Session(engine) as sess:
        stmt = select(images_tbl.c.path, images_tbl.c.tags).where(build_filter(tags))
        rows = sess.execute(stmt).fetchall()

        if args.json:
            out = [{"path": r.path, "tags": r.tags} for r in rows]
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"Found {len(rows)} images matching tags: {', '.join(tags)}")
            for r in rows:
                print(f"- {r.path}")

if __name__ == "__main__":
    main()
```

#### 使い方例

```bash
# 1️⃣ タグ付け（初回のみ実行）
$ python src/tagger.py ./images

# 2️⃣ タグ検索（カンマ区切り OR 検索）
$ python src/search.py beach,sunset
Found 12 images matching tags: beach, sunset
- /home/user/photo_tagger/images/vacation/img001.jpg
- /home/user/photo_tagger/images/vacation/img023.jpg
...

# 3️⃣ JSON 出力（外部ツールと連携したいとき）
$ python src/search.py "dog, park" --json > result.json
```

---

## 5️⃣ 任意：簡易 Web UI（Flask）でタグ検索

`src/webapp.py` を実装すれば、ブラウザから検索できるようになります。以下は **最小構成** の例です。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template_string
from sqlalchemy import create_engine, select, Table, MetaData, or_
from sqlalchemy.orm import Session

DB_PATH = "images.db"
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
metadata = MetaData()
images_tbl = Table("images", metadata, autoload_with=engine)

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Photo Tag Search</title>
<h1>タグ検索</h1>
<form method="GET">
    <input name="q" placeholder="例: beach, sunset" size="40" value="{{query|e}}">
    <button type="submit">検索</button>
</form>
{% if results is not none %}
    <h2>結果 ({{results|length}} 件)</h2>
    <ul>
    {% for r in results %}
        <li>{{ r.path }} <br><small>tags: {{ r.tags }}</small></li>
    {% endfor %}
    </ul>
{% endif %}
"""

def build_filter(tag_list):
    cond = [images_tbl.c.tags.like(f"%{t.strip()}%") for t in tag_list]
    return or_(*cond)

@app.route("/", methods=["GET"])
def index():
    q = request.args.get("q", "")
    results = None
    if q:
        tags = [t.strip() for t in q.split(",") if t.strip()]
        with Session(engine) as sess:
            stmt = select(images_tbl.c.path, images_tbl.c.tags).where(build_filter(tags))
            rows = sess.execute(stmt).fetchall()
            results = [{"path": r.path, "tags": r.tags} for r in rows]
    return render_template_string(HTML, query=q, results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

- `python src/webapp.py` で起動 → `http://localhost:5000` にアクセス  
- 検索ボックスに **カンマ区切り** でタグを入力すれば即座に結果が表示されます。

---

## 6️⃣ 実装・運用手順（ステップバイステップ）

1. **リポジトリ作成 & 仮想環境構築**  
   ```bash
   git clone https://github.com/yourname/photo_tagger.git
   cd photo_tagger
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Python ライブラリインストール**  
   ```bash
   pip install -U pip
   pip install pillow tqdm sqlalchemy torch torchvision transformers flask
   ```

3. **llama.cpp のビルド**  
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make -j$(nproc)               # Windows は `build.bat` を利用
   cd ..
   ```

4. **Gemma‑2B‑it（または 7B）モデル取得 & GGML 変換**  
   ```bash
   # HuggingFace からダウンロード（例: gemma-2b-it）
   pip install huggingface_hub
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="google/gemma-2b-it", local_dir="gemma-2b-it")
   PY

   # 変換スクリプトは llama.cpp リポジトリに同梱
   ./llama.cpp/convert-hf-to-ggml.py gemma-2b-it ggml-gemma-2b-it.bin
   mv ggml-gemma-2b-it.bin models/
   ```

5. **画像キャプションモデルのダウンロード**（最初の実行時に自動で取得されますが、事前にダウンロードしておくと高速）  
   ```bash
   python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
   BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
   BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')"
   ```

6. **タグ付け実行**  
   ```bash
   python src/tagger.py ./images   # images/ は自分の写真フォルダ
   ```

7. **タグ検索（CLI）**  
   ```bash
   python src/search.py "beach,sunset"
   ```

8. **（任意）Web UI 起動**  
   ```bash
   python src/webapp.py
   ```

9. **定期的な再タグ付け**  
   - 新しい写真が増えたら `tagger.py --force` で差分だけ更新  
   - `cron` か Windows のタスクスケジューラで自動化可能  

---

## 7️⃣ カスタマイズ・拡張アイディア

| 項目 | 具体例 |
|------|--------|
| **多言語対応** | キャプション生成後に `translate`（Open‑source MarianMT）で日本語に変換し、LLM に日本語タグを生成させる |
| **タグの階層化** | `tags` カラムを JSON（例: `{"scene": ["beach"], "weather": ["sunny"]}`）に変えて、属性別検索を実装 |
| **画像プレビュー** | CLI の `--preview` オプションで `PIL.Image.show()`、Web UI でサムネイル表示 |
| **高速化** | 画像キャプションを **ONNX Runtime** に置き換えるか、GPU があれば `torch.cuda.amp` で半精度推論 |
| **検索エンジン** | SQLite の代わりに **ElasticSearch** や **Meilisearch** を使えば高速全文検索・ファセットが可能 |
| **タグの自動学習** | LLM の出力を人手で修正し、`fine-tune` 用データセットにして、次回のタグ生成精度を向上させる |

---

## 8️⃣ トラブルシューティング

| 現象 | 原因例 | 解決策 |
|------|--------|--------|
| **llama.cpp が起動しない** | `ggml-gemma-2b-it.bin` のパスが違う、実行権限がない | `LLAMA_CPP_EXE` と `GGML_MODEL` のパスをフルパスで指定、`chmod +x main` |
| **キャプションが空文字** | 画像が壊れている、PIL が読み込めない | 画像形式を確認、`try/except` でスキップ |
| **タグが全て同じ** | `temperature` が 0 に近すぎる、プロンプトが曖昧 | `LLAMA_TEMPERATURE` を 0.2‑0.4 程度に上げ、プロンプト文を明示的に「3〜6 個」指定 |
| **検索結果が期待と異なる** | SQLite の `LIKE` が大文字小文字を区別している | `CREATE TABLE images (..., tags TEXT COLLATE NOCASE);` または検索時に `lower(tags) LIKE lower(?)` |
| **GPU が使われていない** | `torch.cuda.is_available()` が False | CUDA ドライバ・`torch` の CUDA バージョンを再インストール、`nvidia-smi` で GPU が認識されているか確認 |

---

## 9️⃣ まとめ

- **画像→キャプション** は **BLIP**（軽量・CPU でも可）で実装  
- **キャプション→タグ** は **Gemma‑2B‑it**（llama.cpp）にシンプルなプロンプト投げるだけで実現  
- **SQLite** に **path / caption / tags** を永続化し、**CLI** と **Flask** の二本柱で検索インタフェースを提供  
- 完全 **ローカル**、**オープンソース**、**拡張性** が高く、画像フォルダが増えても **再走査** だけで対応可能  

これで **「複数フォルダに散らばる写真を自動でタグ付けし、タグ検索できるシステム」** が完成です。ぜひお試しください！ 🚀

--- 

**質問やカスタマイズ要望があれば遠慮なくどうぞ。**