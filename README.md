# QPII Pipeline — 準識別子(Quasi-PII)抽出と再識別リスク学習データ生成

`nvidia/Nemotron-Personas-USA`（N = 1,000,000 レコード）の自由記述ペルソナから準識別子
（quasi-identifier, QI）を抽出し、テキスト span 単位の再識別リスク
`y = log10(N / support)`（`support` = その QI 集合を含むレコード数）を付与した
機械学習用データを生成するパイプライン。

主要ステージ: `method-a`（統計的トークン抽出）→ `method-b`（名詞句展開で QI 語彙を構築）
→ `cooccurrence`（各 clause へ語彙を適用し QI 集合を生成）→ `risk-sets` / `span-join`
→ `sample-spans`（学習データ化）。

---

## 学習データ仕様：`data/07_reid_samples.parquet`

機械学習担当者へ渡す最終成果物。**1 行 = 1 つのタグ付き clause（テキスト span）**で、
モデル学習・評価に必要な全て（入力テキスト・目的変数・分割・重み）を含む自己完結ファイル。
同内容を CSV（`07_reid_samples.csv`）と Parquet（`07_reid_samples.parquet`、`to_parquet.py`
で生成、列指向＋圧縮で配布・部分読込に有利）の2形式で提供する。

- 規模: **38,690,112 行 × 15 列**（N = 1,000,000 レコード由来）
- `provenance` 内訳: PROFILE 27,426,378（70.9 %）/ PII 8,917,993（23.1 %）/ NONE 2,345,741（6.1 %）
- `split` 内訳: train 30,896,561 / test 7,793,551（test は自然分布で固定）

### 列定義

| 列 | 型 | 役割 | 説明 |
|---|---|---|---|
| `text` | string | **入力 X** | タグ付き clause の生テキスト。BERT 等の埋め込み入力（埋め込み取得は別コード）|
| `y_combined` | float | **目的変数 Y（推奨）** | 階層合成: NONE→0 / `support≥2`→`y_risk` / `support=1`→`max(6, y_bits)` / PII→ceiling |
| `y_risk` | float | 目的変数（別案）| `log10(N/support)`。`support=1` で 6 に飽和 |
| `y_bits` | float | 目的変数（別案）| `Σ log10(N/df_t)`（識別情報量・非キャップ）。NONE=0 |
| `y_bits_capped` | float | 同上の `min(6, y_bits)` 版 | 比較用 |
| `provenance` | category | 層・カテゴリ | `PROFILE`(QIで段階評価) / `PII`(直接識別子=最大リスク, ceiling 固定) / `NONE`(非識別, risk 0) |
| `split` | category | **分割** | `train` / `test`。QI 構成語でグループ化したリーク防止分割。`test` は母集団の自然分布で固定 |
| `lds_weight` | float | 標本重み | `y_risk` 基準の LDS（密度逆数）重み。**`y_combined`/`y_bits` 学習時は要再計算**（後述）|
| `support` | int | メタ/特徴 | その QI 集合 `qi_json` を共有するレコード数（PII は便宜値、リスクは ceiling 固定）|
| `min_constituent_df` | int | 補助特徴 | 最レア構成語の出現レコード数（最も識別的な単一属性）。NONE=0 |
| `qi_json` | string | メタ/特徴 | マッチした QI 属性（基底名詞句）の集合（JSON 配列）|
| `size` | int | メタ/特徴 | QI 集合の要素数 |
| `tier_strictest` | string | メタ/特徴 | 構成語の最も厳しい alpha tier |
| `freq` | int | メタ | この clause テキストの出現回数（distinct text 集約前の件数）|
| `N` | int | 定数 | 総レコード数（1,000,000）|

### 使い方と注意

1. **X = `text` 列**から埋め込みを取得し、**Y = `y_combined`** を回帰（目的変数の設計は後述の節を参照）。`y_risk`（飽和）/`y_bits`（上界）も比較用に同梱。
2. **`split` を必ず使う。自前でランダム再分割しない**——QI 構成語でグループ化したリーク防止分割であり、ランダム再分割すると同一 QI 集合が train/test に跨ってリークする。`test` は母集団の自然分布で固定された評価集合。
3. **`lds_weight` は `y_risk` 基準**。`y_combined` で学習する場合は重みを `y_combined` の分布上で再計算する（同梱の `lds_weights()` 参照）。
4. **PII の扱い**：`provenance == "PII"`（23 %、`y_combined = ceiling`）は、回帰に含めても、二値「確実識別」クラスとして分離してもよい。
5. **評価**は同梱の `eval_risk.py`（`text, y_true, y_pred[, split]` を入力に、層別 MAE/RMSE・macro 平均・Spearman・unicity-AUC・自明ベースライン）。

### 読込例

```python
import pandas as pd

# 必要列だけ高速ロード(Parquet の列指向の利点)
df = pd.read_parquet("data/07_reid_samples.parquet",
                     columns=["text", "y_combined", "split", "provenance"])
train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]      # 自然分布の固定テスト
# X = embed(train["text"]);  y = train["y_combined"]
```

---

## QI 語彙の周波数バンドパス：`MAX_PHRASE_FREQ = 300,000` の選定根拠

### 背景と問題

`method-a`（Fisher の正確確率検定 + FDR）は PROFILE テキストと NONE テキストを**ジャンルとして
弁別する**語を選ぶ。しかしこの「ジャンル弁別性」は「**個人識別性**」とは異なる。たとえば
`enjoys` / `love` / `participates` のような語は PROFILE に偏って出現するため候補に残るが、
ほぼ全レコードに現れるため、どの個人かを絞り込む力（個人識別性）を持たない。これらは
準識別子ではなく**レジスター（文体）マーカー**である。

このような高頻度・非識別語が QI 集合に混入すると、再識別リスク `y = log10(N/support)` に二重の
歪みを与える。第一に、これらの語は**個体間で揺れる**（同義の活動を一方は `enjoys`、他方は
`love` と記述する）ため、本来一致するはずの属性集合を別集合へ分裂させ、`support = 1`
（＝一意）を人工的に水増しする。第二に、集合サイズを定数的に増やし、高次元化による
ユニシティ（de Montjoye et al., 2013; Rocher et al., 2019）を不必要に強める。

### 理論的根拠

ある属性の出現率を `p` とすると、その存在が運ぶ識別情報量は Shannon の自己情報量
`-log p` ビットで近似される。`p → 1`（ほぼ全員が持つ）の語は `-log p → 0` ビットであり、
情報を持たない。また k-匿名性（Sweeney, 2002）の観点では、出現率 `p` の属性値は母集団を
規模 `pN` の同値群に分割するに過ぎず、`p` が大きい属性は単独では再識別に寄与しない。

したがって準識別子は**頻度の中間帯**に存在すべきである。下限
（`MIN_PHRASE_FREQ`、稀すぎて準直接識別子・ノイズとなる語を除外）に加えて**上限**
（ありふれ過ぎて非識別な語を除外）を設けるこの操作は、情報検索における高文書頻度語の除去
（Luhn, 1958; 逆文書頻度 IDF, Spärck Jones, 1972）と同型の**周波数バンドパス**である。
上限を担うのが本パラメータ `MAX_PHRASE_FREQ` である。

### 経験的根拠と閾値の決定

フィルタは `method-b` 辞書（3,650,011 phrase）の出現頻度 `freq`（その phrase を含む
PROFILE clause 数）を閾値として切る。`freq` の分布は強い裾の重い分布で（中央値 = 2,
99 パーセンタイル = 312, 最大 = 1,016,680）、鋭い谷を持たない。したがって閾値は頻度の
切れ目ではなく**意味的境界**で定める。最頻語を頻度およびレコード出現率（`record_count / N`）
とともに観察すると、語が「非識別なジャンルマーカー」から「実在の属性」へ移行する明瞭な境界が
**頻度 300,000（最頻層ではレコード出現率 30 % 以上に相当）付近**に存在する：

| 代表語（freq / レコード出現率） | 性質 |
|---|---|
| `enjoys`(1,016,680 / 98 %), `love`(953,281 / 80 %), `participates`(622,052 / 57 %), `family`(686,377 / 56 %), `curiosity`(597,428 / 54 %), `cooking`(472,224 / 51 %) | 動詞・超一般名詞＝非識別 |
| `community`, `volunteer`, `excels`, `spends`, `reading`, `music`(301,392 / 29 %), `weekend`(303,618 / 36 %) | 一般活動・弱い QI |
| **— 閾値 freq = 300,000 —** | |
| `spanish`(299,655 / 20 %), `garden`(211,600), `knit`(203,304), `heritage`(202,287), `budgeting` | **実在の言語・趣味・技能** |
| `atlanta braves`(20,083), `green bay packers`(20,046), `tableau`, `joni mitchell`, `scandinavian` | **強い QI**（球団・固有ツール・固有名） |

`MAX_PHRASE_FREQ = 300,000` は、`freq` がこの値を超える 29 語を許可語彙から除外する。除外対象は
`enjoys`, `love`, `family`, `participates`, `curiosity`, …, `music`, `weekend`, `community service`
など、いずれもジャンルマーカー・超一般語に限られる（実測されたレコード出現率は最上位で 98 %、
境界の `music` で 29 %）。一方、これを下回る `spanish`（出現率 20 %）, `garden`, `knit`,
`heritage`（いずれも `freq ≈ 200,000`）, 球団名・固有名（`freq ≈ 20,000`）といった
**実在の準識別子は全て保持される**。

閾値を 200,000 未満へ下げると、境界付近に `knit` / `garden` / `heritage` 等の実在の趣味属性が
現れて除外され始めるため、実属性への食い込みを避ける**保守的な下限として 300,000 を採用**した。
すなわち本閾値は、(i) 情報理論・k-匿名性に基づく「中頻度帯に QI を限定する」原理と、
(ii) 実データ上で非識別マーカーと実属性が分離する経験的境界の双方から正当化される、
データ駆動かつ最小侵襲の選択である。

---

## 再識別リスクの目的変数：support 飽和と `y_combined` 階層設計

### support ベースリスクの飽和問題

各 span の再識別リスクを `y_risk = log10(N / support)`（`support` = その QI 集合を共有する
レコード数）で定義すると、`support = 1`（標本中で一意）の span はすべて上限
`y_risk = log10(N) = 6` を取る。問題は、**リッチな自由記述ペルソナでは大多数の個人が一意になる**
こと（unicity; de Montjoye et al., 2013; Rocher et al., 2019）である。本データでは PROFILE
span 27,426,378 のうち **19,242,549（70.2 %）が support = 1** であり、これらが全て `y_risk = 6`
に潰れる。順位学習（AUC 評価）にとってラベルが飽和した退化状態となる。

この飽和は QI 抽出の粒度を変えても解けない。長い名詞句（連鎖 NP）では一意な**文字列**として、
基底名詞句では高次元属性集合の一意な**組合せ**として現れるだけで、source が変わるだけである
（実測で support = 1 比率は連鎖 NP で約 40 %、基底 NP で約 73 %）。すなわち飽和は前処理の
不備ではなく、データの真の性質である。

### `y_bits`：識別情報量（独立仮定）

そこで support とは別軸の連続量として、構成属性の**識別情報量**を導入する：

```
y_bits = Σ_{t ∈ QI集合} −log10( df_t / N ) = Σ_{t} log10( N / df_t )
```

`df_t` は属性 `t` の出現レコード数。`−log10(df_t/N)` は Shannon の自己情報量（surprisal;
Shannon, 1948）で、「その属性を知ると母集団が何桁絞られるか」を表す。和は各属性の絞り込みの
合計であり、`log10(N) = 6` を超えて伸びる。`y_risk` が `support = 1` で飽和するのに対し、
`y_bits` は**属性の希少度と数に応じて連続に増加**するため、飽和域（一意化された塊）を
「同定の頑健さ＝外部データへの照合容易さ」によって段階化できる。

`y_bits` は属性間の相関を無視（独立を仮定）するため相関属性を二重計上し、識別性の**上界**を
与える。NONE span は annotation により非識別とみなし `y_bits = 0` とする。

### `y_combined`：相関の推定可能性に基づく階層設計

`y_risk`（経験的 support、相関を織り込む）と `y_bits`（周辺頻度、相関を無視）は競合ではなく
**領域ごとに補完**する。鍵は「相関は反復観測がある領域でのみ推定できる」点である。
`support ≥ 2` の集合は複数レコードで観測されるため共起＝相関の推定が信頼でき、`y_risk` が
正確。一方 `support = 1`（n = 1）の集合では相関を推定する情報が無く、`y_risk` も飽和する。
そこで推定可能な領域でのみ経験的リスクを用い、推定不能な飽和域は周辺頻度ベースに委ねる：

```
y_combined = 0                          (NONE; 非識別アンカー)
           = log10(N / support)         (support ≥ 2; 経験的・相関考慮; 区間 [0, 6))
           = max( log10(N), y_bits )     (support = 1; 周辺頻度で段階化; 区間 [6, ∞))
```

独立仮定（相関無視）は飽和域では欠点ではなく、(i) n = 1 で推定できない相関を諦めた頑健な
近似であり、(ii) 標本固有の共起構造より周辺頻度の方が未知テキストへ**転移しやすく**、
(iii) 合成（LLM 生成）データに含まれる生成器由来の共起アーティファクトに頑健、という利点を持つ。

### PII tier：直接識別子

氏名等の**直接識別子**を含む PII span は、QI の希少度に依らず再識別が確実（certain
re-identification）であり最大リスクである。これを `provenance = "PII"` で PROFILE と区別
できる第3カテゴリとして保持しつつ、各リスクスケールの上限へ固定する（`y_risk = log10(N)`、
非有界の `y_combined`/`y_bits` は PROFILE 内最大値 = ceiling に揃え、全 PROFILE 以上に配置）。
区別フラグにより、下流は PII を回帰から分離して二値最大リスククラスとして扱うことも、
ceiling 値のまま回帰に含めることも選べる。

### 実証分布

上記により得た `y_combined` は、飽和した単峰ラベルから**連続・段階的な4層構造**へ変換される
（N = 1,000,000、span 38,690,112 件）：

| 層 | 件数（割合） | `y_combined` | 段階化の根拠 |
|---|---|---|---|
| NONE | 2,345,741（6.1 %）| 0 | 非識別アンカー |
| PROFILE `support ≥ 2` | 8,183,829（21.1 %）| [1, 6) | 経験的 `y_risk`（相関考慮）|
| PROFILE `support = 1` | 19,242,549（49.7 %）| [6, 60.65] | `y_bits`（識別情報量）|
| PII | 8,917,993（23.1 %）| 60.65（ceiling）| 直接識別子＝最大リスク |

飽和していた `support = 1` の塊は `y_combined` 上で滑らかな右裾分布へ展開された
（p25 = 6.41, median = 8.45, p75 = 11.09, p90 = 13.97, p99 = 20.26, max = 60.65）。
学習データには `y_risk`・`y_bits`・`y_combined`・`min_constituent_df`（最レア構成語の DF）を
併記し、下流が目的に応じて選択・合成できるようにしている。なお標本サイズ N = 10⁶ は
米国人口（約 3.4 × 10⁸）の部分標本であり、`support = 1` は母集団の一意性ではなく標本解像度の
下限（おおむね母集団で数百人規模）に対応する点に留意が必要である（右側打ち切り）。

### 参考文献

- Shannon, C. E. (1948). *A mathematical theory of communication.* Bell System Technical Journal, 27(3), 379–423.
- Sweeney, L. (2002). *k-anonymity: A model for protecting privacy.* International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5), 557–570.
- de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D. (2013). *Unique in the crowd: The privacy bounds of human mobility.* Scientific Reports, 3, 1376.
- Rocher, L., Hendrickx, J. M., & de Montjoye, Y.-A. (2019). *Estimating the success of re-identifications in incomplete datasets using generative models.* Nature Communications, 10, 3069.
- Luhn, H. P. (1958). *The automatic creation of literature abstracts.* IBM Journal of Research and Development, 2(2), 159–165.
- Spärck Jones, K. (1972). *A statistical interpretation of term specificity and its application in retrieval.* Journal of Documentation, 28(1), 11–21.
