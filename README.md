# QPII Pipeline — 準識別子(Quasi-PII)抽出と再識別リスク学習データ生成

`nvidia/Nemotron-Personas-USA`（N = 1,000,000 レコード）の自由記述ペルソナから準識別子
（quasi-identifier, QI）を抽出し、テキスト span 単位の再識別リスク
`y = log10(N / support)`（`support` = その QI 集合を含むレコード数）を付与した
機械学習用データを生成するパイプライン。

主要ステージ: `method-a`（統計的トークン抽出）→ `method-b`（名詞句展開で QI 語彙を構築）
→ `cooccurrence`（各 clause へ語彙を適用し QI 集合を生成）→ `risk-sets` / `span-join`
→ `sample-spans`（学習データ化）。

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

### 参考文献

- Sweeney, L. (2002). *k-anonymity: A model for protecting privacy.* International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5), 557–570.
- de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D. (2013). *Unique in the crowd: The privacy bounds of human mobility.* Scientific Reports, 3, 1376.
- Rocher, L., Hendrickx, J. M., & de Montjoye, Y.-A. (2019). *Estimating the success of re-identifications in incomplete datasets using generative models.* Nature Communications, 10, 3069.
- Luhn, H. P. (1958). *The automatic creation of literature abstracts.* IBM Journal of Research and Development, 2(2), 159–165.
- Spärck Jones, K. (1972). *A statistical interpretation of term specificity and its application in retrieval.* Journal of Documentation, 28(1), 11–21.
