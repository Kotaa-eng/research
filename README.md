複数シートの直接法で背景光が強い結果を確認したため、次は [lsh_multisheet_design.ipynb](lsh_multisheet_design.ipynb) で全体の係数設計を検証する。最初は解析的な直接法のみ。新しい画質は未検証、数値実行はユーザー側。手順は [LSH複数シート_係数設計.md](LSH複数シート_係数設計.md)。

同じ「123」をy方向へ重ねる厚みの検証は [lsh_multisheet.ipynb](lsh_multisheet.ipynb)。初期設定は5枚・75 µm間隔・ランク450。直接J0生成とx/y両軸のコアシフトを比較し、全センサー面・各y断面・3D ROI・厚みを評価する。説明は [LSH複数シート検証.md](LSH複数シート検証.md)。数値計算はユーザー側で実行する。
保存済みデータを黒い背景・白い立体表面で表示する入口は [lsh_multisheet_3d.ipynb](lsh_multisheet_3d.ipynb)。伝搬を再計算せず、表示の視点と共通強度閾値を変更できる。

数字123はランク512で直接法のCGH再生強度との相対L2誤差0.164%、SSIM 0.999984を確認した。次は [lsh_digits123_rank_refinement.ipynb](lsh_digits123_rank_refinement.ipynb) で384・400〜500（10刻み）・512を比較し、合格した最小候補のCGH・伝搬まで検証する。既存の親基底と光学条件を固定する。手順・保存先は [LSH数字123_ランク細分化.md](LSH数字123_ランク細分化.md)。新しい計算はユーザー側で実行する。

低ランクの失敗も含めて再生像を見る入口は [lsh_digits123_low_rank.ipynb](lsh_digits123_low_rank.ipynb)。50・100・200・300・450・512を比較する。既存の200以上のCGH・伝搬結果を再利用し、追加分だけ伝搬する。場生成時間は全候補で測り直す。保存先は [LSH数字123_低ランク診断.md](LSH数字123_低ランク診断.md)。

最初のランク検証は [lsh_digits123_rank_audit.ipynb](lsh_digits123_rank_audit.ipynb)。広帯域123のランク100が圧縮で失敗した診断は [LSH数字123_圧縮失敗の診断.md](LSH数字123_圧縮失敗の診断.md)。

数字123の画質改善は [lsh_digits123_improved.ipynb](lsh_digits123_improved.ipynb)。kr帯域・Thread間隔・干渉を考慮した係数設計の5条件を比較し、選択候補へ同じ提案法を適用する。説明は [LSH数字123_画質改善.md](LSH数字123_画質改善.md)。計算はユーザー側で実行する。

数字「123」を1枚のシートで比較する入口は [lsh_digits123.ipynb](lsh_digits123.ipynb)。横z・縦x、初期設定50 cm・ランク100。目標から係数を再設計し、直接法と提案法のCGH再生・画質・時間を比較する。説明は [LSH数字123_比較.md](LSH数字123_比較.md)。

# LSH高速化研究コード

横z・縦xで正立する数字1の検証は [lsh_digit1_zx.ipynb](lsh_digit1_zx.ipynb)。目標とThread係数を再設計し、同じ伝搬方向・モード・圧縮辞書で理論/提案法を比較する。説明は [LSH数字1_zx方向比較.md](LSH数字1_zx方向比較.md)。

CGH生成からSLMフレーム準備・表示までの計測は [lsh_digit1_slm_timing.ipynb](lsh_digit1_slm_timing.ipynb)。検証用4f再生・伝搬を除き、実機SDKの同期表示関数へ接続できる。未接続なら準備時間のみ保存し、表示時間は未計測とする。詳細は [LSH数字1_SLM表示時間.md](LSH数字1_SLM表示時間.md)。

提案法の左右対称化を修正：中心の偶対称モードだけに偏ったx基底を、正負の移動成分も学習する基底へ変更した。[lsh_digit1_proposed.ipynb](lsh_digit1_proposed.ipynb) の修正版は旧結果を残し `shiftaware_v2` へ保存する。原因・修正・追加診断は [LSH数字1_左右対称化の修正.md](LSH数字1_左右対称化の修正.md)。数値検証はユーザー側で行う。

数字1の波数間隔を半分・4分の1にする比較は [lsh_digit1_kr_spacing.ipynb](lsh_digit1_kr_spacing.ipynb)。kr範囲とランクを固定し、15・29・57モードの直接法・提案法を比較する。説明は [LSH数字1_kr間隔比較.md](LSH数字1_kr間隔比較.md)。

数字1の直接生成と提案法（共有FFTコア加算・シフト）のCGH再生・伝搬比較は [lsh_digit1_proposed.ipynb](lsh_digit1_proposed.ipynb)。初期ランク100、理論版と同じ条件を使用し、目標再現と直接法への一致を別々に評価する。説明は [LSH数字1_提案法検証.md](LSH数字1_提案法検証.md)。

数字「1」を1枚のx–zシートとして理論伝搬させる検証は [lsh_digit1_theory.ipynb](lsh_digit1_theory.ipynb)。圧縮データを使わず、解析Bessel重ね合わせ・有限開口の角スペクトル伝搬・振幅符号化CGHの理想4f再生を比較する。説明は [LSH数字1_理論検証.md](LSH数字1_理論検証.md)。

現在の主題は **Stage A：Tucker近似・コア加算・x方向コアシフト・理想数値CGHの整合性** です。2048×2048画素、3.74 µm/画素の数値グリッドで、全Besselモードを同じ共通基底へ圧縮します。この画素ピッチはシミュレーションの座標間隔であり、Stage AではSLMや4f系をモデル化しません。

主実行入口と結果の読み方は [CURRENT.md](CURRENT.md) を参照してください。`test.ipynb`、`lsh_parameters.py`、物理開口・Gaussian照明・伝搬・4f倍率を扱う既存コードと `verified_thread_runs/` の出力は **Stage B候補／過去の検証** として残します。Stage Aの結果と混同しないでください。

Stage Aの依存ライブラリは `requirements-stage-a.txt` にまとめ、実行手順は [コード説明.md](コード説明.md) に記載しています。
