# 数字「1」の直接法・提案法・伝搬比較

**修正版の変更点：** 中心モードだけから作った旧基底は左右対称化を生じたため、標準戦略を移動後の奇成分も学習する `shift_aware` へ変更した。現在のNotebookの辞書は `lsh_digit1_library_rank100_shiftaware.npz`、結果は `lsh_digit1_proposed_rank100_shiftaware_v2/`。以下の旧v1の保存先・中心基底の説明は旧方式の記録として残す。新しい基底の手順と診断は [LSH数字1_左右対称化の修正.md](LSH数字1_左右対称化の修正.md) を参照。

実行入口は `lsh_digit1_proposed.ipynb`。これまでの理論版と同じ数字1を、直接生成と共有FFTコアを用いた提案法で作り、同じ光学モデルで伝搬させる。圧縮法に一致しているかと、目標の数字に一致しているかは別々に評価する。

## 実行と変更箇所

Notebookを上から実行する。最初の設定セルの `RANK = 100` を変更するとランクを変更できる。初期値では `lsh_digit1_theory_outputs_v1/summary.json` から理論版の設定を読み、画像や場の保存値は使わず直接法を再計算する。理論版のフォルダがない場合は理論版の標準設定を使う。

- 圧縮辞書：`lsh_digit1_library_rank100.npz`。初回に作り、次回から同じ仕様なら読み込む。
- 結果：`lsh_digit1_proposed_rank100_v1/`。実行後に作られる。
- 同じ設定の完了結果があればNotebookは結果を読み込む。再計算する場合は `output_dir` の末尾を `v2` などへ変更する。
- 波長・ピッチ・画素数・中心半径・設計長・ランク・seedを変更した場合、既存辞書とは不一致になる。`library_path` も新しい名前に変更する。旧辞書・旧結果は上書きしない。
- Thread配置や目標だけを変更する場合は同じ辞書を使える。ただし結果フォルダは変更する。

CLIの場合：

```powershell
python lsh_digit1_proposed.py --rank 100 --theory-results lsh_digit1_theory_outputs_v1 --output-dir lsh_digit1_proposed_rank100_v1
```

依存ライブラリは理論版と同じ `requirements-stage-a.txt`。このコードを追加した時点では数値実行していない。

## 提案法が行う計算

オフラインでは各中心Besselモードの **開口で切る前のJ0複素場** をFFTし、全モード共通の実直交因子 `Uy,Ux` とモード別複素コア `Cm` を作る。空間2軸のみをランクrに圧縮し、モード軸は保持する。因子の各列を1次元IFFTした `Vy,Vx` も保存する。非圧縮の全モードFFTは一時ファイルから削除し、恒久保存しない。

理論版は原点を整数画素に置く。旧Stage Aの半画素中心の辞書をそのまま使うと座標が異なるため、この検証は専用辞書を使う。

オンラインは以下の順序。**Threadごとに異なる係数**を用いる。

1. 目標の強度から平方根を取り、各Threadの縦方向の振幅をFourier係数 `A[m,q]` にする。この設計処理は両手法で共通で、場生成時間には含めない。
2. Thread qのコアを合成：`Hq = sum_m A[m,q] Cm`。
3. x座標シフトを射影：`Axq = Ux.T diag(exp(-i*kx*xq)) Ux`。
4. シフトしたコアを合算：`Htotal = sum_q Hq @ Axq`。
5. 1回だけ復元：`E = Vy @ Htotal @ Vx.T`。全画素IFFTはオンラインで行わない。通常の転置 `.T` を使い、複素共役転置ではない。
6. 全Threadを加えた後に、直接法と同じ開口・入射照明を適用する。

直接法は各Thread位置で `J0(kr*sqrt((x-xq)^2+y^2))` を直接計算して重み付き加算する。保存したビーム画像・FFT・コアは使わない。

## 伝搬・CGHの比較

両手法の場を**直接法の同じ定数**で正規化する。提案法だけ最大値を合わせて誤差を隠す操作はしない。

各手法について、有限複素場をそのまま角スペクトル法で伝搬する経路と、振幅を符号化した8-bit位相CGHを理想4fの+1次フィルタで再生してから伝搬する経路を計算する。4経路に同じ開口・波長・CGH方式・フィルタ・パディング・伝搬距離を使う。センサーのy=0線を連結しx–zシートを得る。CGH再生は既知の0.58の場減衰だけ補正し、実際の減衰を含む強度もNPZに保存する。

提案場の振幅が共通の符号化範囲を超えた画素は、理論版と同じ符号化器でクリップされる。`encoding.proposed.amplitude_saturation_fraction` を確認する。これもCGH化による誤差の要因となる。

仮想1:1の理想4fと理想センサーの検証であり、実機SLM・レンズ収差・センサーの画素積分・ノイズ・実際の倍率の校正は含まない。

## 保存画像

| ファイル | 比較内容 |
|---|---|
| `target_direct_proposed.png` | 目標、直接法CGH再生、提案法CGH再生のx–z強度。共通スケール |
| `complex_sheet_comparison.png` | CGH化前の直接法・提案法の伝搬強度と差分 |
| `cgh_sheet_comparison.png` | CGH再生後の直接法・提案法の伝搬強度と差分 |
| `initial_field_comparison.png` | z=0の振幅・マスク付き位相・円周位相差 |
| `direct_cgh_8bit.png` | 直接生成した場から作った原寸CGH |
| `proposed_cgh_8bit.png` | 圧縮コアから作った原寸CGH |
| `cgh_mask_comparison.png` | 上のCGH2枚と円周コード差 |
| `sensor_xy/cgh_z0250.0mm.png` など | 各z位置での直接・提案センサーx–y強度と差分 |
| `factors/` | 共通因子行列と全モードコアの可視化 |

差分画像は小さな差も見えるように専用スケール。必ずカラーバーの数値を見る。

## 数値指標：summary.json

- `metrics.initial_complex_proposed_vs_direct`：z=0複素場の相対L2、複素SNR、振幅・強度SSIM/PSNR/SNR、明るい領域の円周位相誤差。
- `metrics.cgh_mask_proposed_vs_direct`：8-bit CGHの参考SSIM・円周コードPSNR・位相差。これだけで光学再生の成功とは判定しない。
- `metrics.complex_sheet_proposed_vs_direct`：CGH化前の伝搬後x–z強度SSIM/PSNR/SNR、複素場・強度相対L2。
- `metrics.cgh_sheet_proposed_vs_direct`：CGH再生後の同じ指標。**提案法による伝搬結果が直接法と一致するか**を見る主項目。
- `metrics.direct_cgh_sheet_vs_target` / `proposed_cgh_sheet_vs_target`：**数字の目標自体を再現しているか**。元の直接法がぼけていれば、提案法が完全に一致しても目標再現の問題は残る。
- `metrics.sensor_xy_proposed_vs_direct`：各センサー面での強度評価。
- `active_thread_count` / `nonzero_bessel_term_count`：実際に係数が非零のThread数とBessel項数。21本の配置すべてを点灯しているとは限らない。

PSNR/SNRが無限大になる場合はJSONでは文字列 `"Infinity"` と保存する（理論版の共通保存処理）。定義できない値は `null`。位相は基準と復元がともに基準最大振幅の1%以上の画素で評価し、失われた明るい画素も別途数える。

### シフト誤差の診断

`shift_diagnostics` は、追加計算をオンライン時間に入れずに次の3差を評価する。

1. `analytic_direct_vs_exact_periodic`：ずらした座標の直接J0と、有限の中心J0配列をFFTで周期シフトした結果の差。画像端のモデル差。
2. `exact_periodic_vs_compressed_periodic`：上の周期シフトに、コア圧縮だけを適用した差。シフト後の再射影はしない。
3. `compressed_periodic_vs_projected_proposed`：圧縮FFTの厳密な周期シフトと、コア内シフト射影の差。共有部分空間に収まらない成分の影響。

全窓と開口適用後の相対L2を記録する。各項は参照が異なり、単純に足せない。ランクを上げても周期境界と解析J0のモデル差は消えない。診断を省略する場合は `run_shift_diagnostics=False`。

`propagation_sampling` は各保存zで伝搬の帯域制限により除外したスペクトルパワーの割合。`config.check_padding=True` にすると保存z位置でパディング2倍と3倍の比較も行う。未確認の場合、長距離伝搬の数値収束まで確認済みとは扱わない。

## 時間の範囲

`timings_seconds` に直接場生成、提案場生成の内訳、各CGH符号化＋理想4f再生、4経路の伝搬合計を保存する。オフライン辞書構築／読込は `offline` に分離する。場生成時間は設計係数、正規化・開口、画像保存、誤差診断、伝搬を除く。CGH時間には位相マスク作成に加えて理想4fのFFTフィルタも含む。

今回は1回測定で、速度の安定した比較ベンチマークではない。CGH画像生成を高速化する研究と、再生結果の正しさを調べるための伝搬計算時間を混同しない。

## 小規模テスト

`test_lsh_digit1_proposed.py` に、フルランクでのFFT周期シフトとの一致、ランク制限時の射影式との一致、辞書仕様不一致の拒否を確認する小規模テストを用意する。コード追加時点では実行していない。任意に `python -m unittest test_lsh_digit1_proposed` で実行できる。
