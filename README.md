# LSH高速化研究コード

CGH生成からSLMフレーム準備・表示までの計測は [lsh_digit1_slm_timing.ipynb](lsh_digit1_slm_timing.ipynb)。検証用4f再生・伝搬を除き、実機SDKの同期表示関数へ接続できる。未接続なら準備時間のみ保存し、表示時間は未計測とする。詳細は [LSH数字1_SLM表示時間.md](LSH数字1_SLM表示時間.md)。

提案法の左右対称化を修正：中心の偶対称モードだけに偏ったx基底を、正負の移動成分も学習する基底へ変更した。[lsh_digit1_proposed.ipynb](lsh_digit1_proposed.ipynb) の修正版は旧結果を残し `shiftaware_v2` へ保存する。原因・修正・追加診断は [LSH数字1_左右対称化の修正.md](LSH数字1_左右対称化の修正.md)。数値検証はユーザー側で行う。

数字1の波数間隔を半分・4分の1にする比較は [lsh_digit1_kr_spacing.ipynb](lsh_digit1_kr_spacing.ipynb)。kr範囲とランクを固定し、15・29・57モードの直接法・提案法を比較する。説明は [LSH数字1_kr間隔比較.md](LSH数字1_kr間隔比較.md)。

数字1の直接生成と提案法（共有FFTコア加算・シフト）のCGH再生・伝搬比較は [lsh_digit1_proposed.ipynb](lsh_digit1_proposed.ipynb)。初期ランク100、理論版と同じ条件を使用し、目標再現と直接法への一致を別々に評価する。説明は [LSH数字1_提案法検証.md](LSH数字1_提案法検証.md)。

数字「1」を1枚のx–zシートとして理論伝搬させる検証は [lsh_digit1_theory.ipynb](lsh_digit1_theory.ipynb)。圧縮データを使わず、解析Bessel重ね合わせ・有限開口の角スペクトル伝搬・振幅符号化CGHの理想4f再生を比較する。説明は [LSH数字1_理論検証.md](LSH数字1_理論検証.md)。

現在の主題は **Stage A：Tucker近似・コア加算・x方向コアシフト・理想数値CGHの整合性** です。2048×2048画素、3.74 µm/画素の数値グリッドで、全Besselモードを同じ共通基底へ圧縮します。この画素ピッチはシミュレーションの座標間隔であり、Stage AではSLMや4f系をモデル化しません。

主実行入口と結果の読み方は [CURRENT.md](CURRENT.md) を参照してください。`test.ipynb`、`lsh_parameters.py`、物理開口・Gaussian照明・伝搬・4f倍率を扱う既存コードと `verified_thread_runs/` の出力は **Stage B候補／過去の検証** として残します。Stage Aの結果と混同しないでください。

Stage Aの依存ライブラリは `requirements-stage-a.txt` にまとめ、実行手順は [コード説明.md](コード説明.md) に記載しています。
