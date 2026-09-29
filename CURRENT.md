# 現在の研究条件（2026-09-29）

目的は、Bessel/Frozen Waveの複素初期場からphase-only CGHを作るオンライン処理を、FFT後の共通Tucker基底とモード別コアで短縮できるか検証すること。SLMへの実表示や実測結果ではない。

| 項目 | 現在の設定 |
|---|---|
| 波長 | 532 nm |
| SLM | 2048×2048、3.74 µmピッチの実機候補 |
| 4f横倍率の絶対値 | 1/4 |
| 出力結像面 | 0.935 µm/画素、約1.915 mm全幅。ここをz=0とする |
| spot第1零点半径 | 主実行の暫定値100 µm。30,40,50,60,80,100,120 µmを掃引して選ぶ |
| z方向設計長 | 0.50 m。50 cmでの有意強度を保証する値ではない |
| レーザー照明直径・waist | 未確定。`None` は全面一様照明の理想上限 |
| レンズ・絞り有効径 | 未確定。制限が分かり次第、出力面換算直径を設定する |

`lsh_parameters.py` が数値のsource of truth。`test.ipynb` は主検証、`test2.ipynb` は移動位相と再射影の小例。`lsh_core.py` に物理開口、共通FFT基底、コア移動、空間基底、phase-only符号化を置く。`rank_survey.py` がランク調査、`radius_tucker_study.py` が半径調査、`shifted_rank80_study.py` と `multi_pattern_rank80_study.py` が移動Thread/CGH比較、`lsh_online_benchmark.py` が5方式のオンライン内訳計測。

基本は方位次数n=0。FWのスペクトル番号mは別物。`kr`は `kz_m=Q+2πm/L` と分散関係から計算し、等間隔の `kr` を仮定しない。1本のThread内では全mを同軸に重み付けし、Thread単位で同じ位置へ移動する。

`F≈Uy C Ux.T` の移動後は `Ay=Uy.T Dy Uy`, `Ax=Ux.T Dx Ux`, `C_shift=Ay C Ax`。同じ共通基底への**再射影近似**であり、移動した場を厳密に表現する保証はない。FFTの連続移動は周期的帯域制限補間。固定開口を持つ直接生成との差は別の誤差として記録する。

現時点の幾何学的到達距離は `R kz/kr` の目安。`LASER_DIAMETER=None` 等のときはSLM全面の**理想上限**で、実際の50 cm到達を主張できない。Gaussian照明のwaistは硬い開口ではなく、1/e振幅で定義した目安。有限開口の伝搬計算と収束確認が別途必要。CGH再生の一致も実機画質の保証ではない。

`LSH_validation_report.md` と `lsh_audit_results.json` は旧1024画素/旧モード構成の履歴。`visualize_bessel_fft.py` は `kr=10` の無次元legacy図。`audit_lsh.py`, `plot_lsh_audit.py`, `run_bessel_notebook.py` と既存の出力ディレクトリも主条件の根拠にしない。設定変更後はノートブックを先頭から再実行し、同じconfig IDの出力だけ比較する。
