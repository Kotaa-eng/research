# 数字1：波数間隔を細かくする比較

**基底修正後：** 現在のNotebookは `shift_aware` を使い、ケースは `..._shiftaware_v2/`、集計は `lsh_digit1_kr_spacing_outputs_rank100_v2/` へ保存する。旧center_onlyのv1結果は修正版として読み込まない。以下のv1保存先は旧方式の記録。左右対称化の原因・修正は [LSH数字1_左右対称化の修正.md](LSH数字1_左右対称化の修正.md) を参照。

入口は `lsh_digit1_kr_spacing.ipynb`。ランク100、中心モードの半径50 µm、同じkr端点、Thread配置、開口、CGH方式、伝搬範囲0〜50 cmを保ち、モード密度だけを変更する。既存の15モード結果は条件が一致すれば再利用し、半分間隔・4分の1間隔を新しい結果フォルダへ保存する。数値計算・テストはコード追加時点では未実行。

## 変更するのはkzの間隔

Frozen Waveの係数は `kz_m = Q + 2πm/T` に対応するFourier係数である。`kr_m = sqrt(k0²-kz_m²)` なのでkrは等間隔ではない。krだけを任意に等間隔へ変更して元の係数を使うことはしない。

標準のT=0.5 m、m=-7…7を基準に、密度倍率sに対してT=s×0.5 m、m=-7s…7sとする。これにより元のkr範囲の端点を維持できる。

| 密度倍率 | kz間隔 | 係数周期T | m | モード数/Thread |
|---|---|---|---|---|
| 1 | 元の間隔 | 0.5 m | -7…7 | 15 |
| 2 | 1/2 | 1.0 m | -14…14 | 29 |
| 4 | 1/4 | 2.0 m | -28…28 | 57 |

係数周期を伸ばしても、伝搬・撮影範囲は0〜0.5 mのまま。目標は元の数字の位置・形のまま、追加した周期領域では0とする。各密度で係数を積分し直す。積分サンプル数もs倍にして、z方向のサンプル間隔を固定する。

**モード数を増やしても波数の帯域は広げていない。** したがって密度増加だけで数字の輪郭の解像度が上がるとは限らない。周期境界からの影響や合成場がどう変わるかを見る検証である。細部の再現には帯域・Thread間隔・Thread間干渉の設計も関係する。

また、元のランク100の結果は「圧縮だけ」より「移動後の再射影」の誤差が大きかった。密な中心ビームで基底を作り直しても、移動後の場を十分表せるとは限らない。モード密度の変更が射影誤差を改善したかも別に評価する。

## 実行・保存先

Notebookを上から実行する。設定は `RANK=100`、`SPACING_FACTORS=(1,2,4)`。比較条件を読み込む理論フォルダは `reference_results_dir`。

- 倍率1：`lsh_digit1_proposed_rank100_v1/`（既存結果を条件一致時のみ再利用）
- 倍率2：`lsh_digit1_proposed_rank100_spacing2_v1/`
- 倍率4：`lsh_digit1_proposed_rank100_spacing4_v1/`
- 集計：`lsh_digit1_kr_spacing_outputs_rank100_v1/`
- 各密度の辞書：`lsh_digit1_library_rank100_spacing2.npz` など。倍率1は既存の専用辞書。

集計の `spacing_comparison.csv` と `spacing_comparison.json` に、モード数・kz間隔・直接/提案の目標との一致・提案/直接の一致・射影誤差・生成時間を並べる。`spacing_comparison.png` はその主要指標。各ケースの実際の再生画像はそれぞれの `target_direct_proposed.png` に保存する。

集計グラフの誤差は相対L2の百分率。SSIMは0〜1の値のまま。直接法が目標へ近づいたか、提案法が直接法へ近づいたかを両方確認する。

## 1ケースだけ実行する場合

```powershell
python lsh_digit1_proposed.py --rank 100 --mode-spacing-factor 2 --theory-results lsh_digit1_theory_outputs_v1
```

既存の `lsh_digit1_proposed.ipynb` は従来の倍率1のまま使用できる。追加したオプションの標準値は1なので、従来のコード呼び出し・専用辞書・結果もそのまま使える。
