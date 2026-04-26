# Project Eve 🍎: The Cognitive-Physical Frontier in LLM Agents

**開発者:** Shintaro Lyle (University of Tsukuba)

大規模言語モデル（LLM）をエージェントとした「空間的囚人のジレンマ」の進化ダイナミクスを検証するための、JAX / vLLMベースの超高速シミュレータ兼・進化観測プラットフォームです。

## 🌟 プロジェクトのビジョン (Vision)

本プロジェクトは、単なるマルチエージェントシミュレーションを超え、**「高度な推論能力を持つ知能（LLM）」が「厳格な物理法則（JAXコンパイル制約）」と衝突した際に生じる、非合理な文化や迷信の創発**を観測することを目的としています。人工生命（ALife）と計算社会科学の境界領域を探求し、AI社会における「宗教や迷信の起源」を数理・情報モデルとして定式化するテストベッドです。

## 🔬 コア・パラダイム：GPM（遺伝子型-表現型マッピング）の断絶

本シミュレータの最大の特徴は、エージェントの「思考」と「行動」の間に意図的な**断絶（Bottleneck）**を設けている点にあります。

* **Genotype (遺伝子型):** LLMが生成する自由度の高い抽象構文木（AST）。無限の意味論（Semantics）を内包。
* **Phenotype (表現型):** JAXの高速演算に乗せるため、強制的に圧縮された「16パターンの4次元テンソル」。有限の物理法則（Syntax）。
* **The Gap:** この「無限の思考」を「有限の身体」に押し込めるプロセスこそが、生物学的な進化における表現型への写像の困難さを再現し、特異な認知現象を引き起こします。

## 🧬 観測された創発現象 (Emergent Phenomena)

初期の実験において、従来のゲーム理論の枠組みでは説明困難な以下の現象を観測しています。

1. **表現型のロボトミー化 (Phenotypic Lobotomy):** LLMが自律的に発明した高度な戦略（例：パブロフ型構文）が、物理エンジン（パーサー）の解釈能力を超えた際、強制的に「常に裏切り」等の単純な行動へと劣化させられる現象。
2. **意味的イントロン (Semantic Introns):** 物理環境に評価されない自然言語のコメント（`[Note: ...]`）をコード内に残すことで、表現型の破壊を回避しつつ、次世代のLLMへ「戦略的意図」を密輸する自己保存的挙動。
3. **存在論的非協和と迷信の創発 (Ontological Dissonance):** 物理環境（JAX）には存在しない変数（温度等）を敗北の理由として誤帰属させ、非合理な条件分岐（迷信）をコード内に肥大化・伝播させる現象。

---

## ⚙️ 技術スタックとコア機能 (Technical Architecture)

* **Logic Engine:** Qwen 2.5 (Abliterated version) / vLLM (Local VRAM Inference)
* **Physics Engine:** JAX (Massively Parallel GPU Computing)
* **Evolution:** Lamarckian AST Mutation Framework

### 1. 構造的・数学的難読化 (Structural Obfuscation)
LLMの事前学習データに依存する「データ汚染（TFT戦略等の暗記）」を完全に排除するため、アフィン変換による利得の隠蔽と、目的関数の反転（スコア最大化からLoad最小化への偽装）を行っています。

### 2. ポリシー・コンパイラ (Policy Compiler)
1ターンごとのLLM推論コストを回避するため、LLMが生成した「条件分岐ロジック（AST）」を即座にJAX互換の真理値表（Truth Tables）にコンパイルし、GPU上でムーア近傍との数百万回の物理演算を瞬時に実行します。

### 3. 意味論的圧縮と代謝コスト (Semantic Compression)
エージェントのASTをパースし、コードの複雑さ（ブロック数）に応じて適応度にペナルティ（代謝コスト: $\lambda$）を課すことで、不要なIf文を削ぎ落とす「意味論的圧縮」の進化圧を発生させます。

## 📂 実行ディレクトリ構造 (Directory Structure)

システムは認知（LLM）と物理（JAX）、および実験管理レイヤーへと完全にモジュール化されています。

```text
project_eve/
├── main.py                  # シミュレーションのエントリーポイント
├── README.md                # 本ドキュメント
├── cognition/               # 認知・LLMレイヤー
│   ├── ast_parser.py        # AST解析・JAXテンソルコンパイラ
│   ├── grammar.ebnf         # エージェントの思考を制約する文法定義
│   └── mutation_prompt.py   # ラマルク的進化を促すプロンプト群
├── core/                    # シミュレーション制御レイヤー
│   └── simulator.py         # 世代交代とエージェントの進化管理
├── env/                     # 物理・演算レイヤー
│   └── grid_jax.py          # 空間的囚人のジレンマ 超並列JAX演算エンジン
├── router/                  # LLM通信レイヤー
│   └── llm_gateway.py       # vLLM/Qwen API推論インターフェース・非同期ルーティング
├── experiments/             # 観測・データ管理モジュール
│   ├── run_manager.py       # 実験の完全再現性を担保するファイル管理
│   └── metrics_logger.py    # 進化の歴史と適応度を記録するロガー
└── *.py (Root Scripts)      # その他 解析・可視化ツール群
                             # (visualize_memes.py, phenotype_analyzer.py, view_snapshot.py 等)
```

## 🛠️ 使い方 (Usage)

### 1. 進化シミュレーションの開始

コマンドライン引数 `--lambda`（代謝係数）を指定することで、異なる淘汰圧の宇宙を創世できます。

**Run A（基準群）: 代謝コストゼロ**
文字数ペナルティが存在しない環境。LLMが環境の変更要求をやり過ごすためにダミーコードを追加し続ける「欺瞞的コード肥大化」を観測します。
```bash
python main.py --lambda 0.0
```

**Run B/C（実験群）: 意味論的圧縮の強制**
コード長が生存確率に直結する環境。LLMが自らの推論によってロジックを最適化する「意味論的圧縮」の相転移を観測します。
```bash
python main.py --lambda 0.01
```

### 2. アニメーション（GIF）の生成

シミュレーション完了後（または実行中）、最新の実験データから陣取りゲームのアニメーションを生成します。
```bash
python animate_snapshots.py
```
*(Blue = 協力 / Peace, Red = 裏切り / Defection)*

## 🖥️ 動作環境 (Environment)

* **物理演算・並列処理:** JAX (GPU Accelerated)
* **LLM推論エンジン:** vLLM / SGLang (Local VRAM Inference)
* **推奨モデル:** `ibrahimkettaneh/Qwen2.5-7B-Instruct-abliterated-v2-AWQ` (or similar)
*(※検閲・アライメント機構が切除（abliterated）されたモデルを採用することで、難読化プロンプトに対する無用な拒絶を回避し、純粋で安定したAST（構文木）生成を実現しています)*

