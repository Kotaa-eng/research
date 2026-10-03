# 数字1・1枚のシートを理論計算で確認する

入口は [lsh_digit1_theory.ipynb](lsh_digit1_theory.ipynb)。上から順に実行する。計算本体は [lsh_digit1_theory.py](lsh_digit1_theory.py)。圧縮コアや保存済みBessel画像を使わず、各位置のJ0を解析式で直接生成する。Stage Aの復元精度調査とは別の実験である。

数字1は **y=0のx–z面** に設計する。固定zでセンサーが読み取る画像はx–y断面なので、それぞれの画像に数字全体が出るわけではない。各zのセンサー画像からy=0の1D横断線を抽出し、z方向に並べたものが数字1のシートになる。この読み取り方は論文のFig. 2dとMethodsに対応する。

## 初期設定

| 項目 | 数値・意味 |
|---|---|
| 波長 | 532 nm |
| 中心モードのスポット半径 | 50 µm：J0の第1零点 |
| 数値格子 | 2048×2048、3.74 µm/画素 |
| Thread配置 | 最大21本、75 µm間隔、x方向±0.75 mm、y=0 |
| モード | 現条件でm=-7…7の15モードをすべて使用 |
| 数字1の範囲 | 横幅1.2 mm、z=75〜425 mm |
| センサー撮影位置 | z=0〜500 mm、標準101点（5 mm間隔） |
| 有限開口 | 半径3.5 mmの円形開口、標準は開口内の一様照明 |
| CGH | 振幅を含めるArrizónの正弦位相符号化、8-bit、斜め8画素周期キャリア |
| 理想4f | ＋1回折次数を円形フィルタで抽出、キャリアを除去。数値座標は仮想1:1結像 |
| 伝搬 | スカラー角スペクトル、2倍ゼロパディング、伝達関数のサンプリング条件による帯域制限 |

Threadごとに数字の横断位置が異なるため、各Threadの係数も異なる。Stage Aで行った「同じThreadを10か所へ移動」の検証とは異なる。目標画像の外にあるThreadの係数は0なので、実際の非零Thread数はsummary.jsonに保存する。最大21×15＝315個の係数を持つが、暗いThreadのビーム生成は省く。

この設定は初期の数値検証用。論文の実機の1920×1200・8 µmピッチ・1/4倍結像へは換算していない。z=0は理想4f出力面に対応させているが、今回のCGHをそのまま実機SLMへ表示する設定ではない。論文の有限Bessel–Gauss実装の完全な再現でもなく、まず有限開口J0と任意の固定Gaussian照明を検証できるようにしている。

## 計算の流れと、各比較の意味

1. 数字1の目標強度I(x,z)を作り、各Thread位置でF_q(z)=sqrt(I(x_q,z))を求める。
2. Fourier積分 `A_mq = (1/L) ∫ F_q(z) exp(-i 2π m z/L) dz` を等間隔の中点積分で評価する。
3. `kz_m=Q+2πm/L`、`kr_m=sqrt(k0²-kz_m²)` とし、各ThreadのJ0を加算する。
4. 無限開口の解析式 `Σ_q,m A_mq J0(kr_m ρ_q) exp(i kz_m z)` をy=0で評価する。これはThread間の干渉も含む。
5. z=0で直接生成した場へ有限開口・照明を掛け、角スペクトルで伝搬する。
6. 振幅と位相を8-bit位相CGHへ符号化し、理想4fフィルタ後の場を別に伝搬する。
7. 理想センサーの強度 `I=|E|²` を記録し、y=0の断面を並べる。

まず `thread_profile_design.png` で、各Thread単独の有限Fourier級数が目標のON/OFFを再現できるかを見る。次に `digit1_sheet_comparison.png` で、目標、無限開口の解析式、有限開口の伝搬、CGHからの再生を比較する。各段階で形が悪くなる理由は異なる。

- 単独Threadの時点で悪い：モード数・縦方向帯域不足、Fourier打切り、Gibbs現象。
- 単独Threadは良いが解析式のシートが悪い：Thread間のコヒーレント干渉や横方向のサンプリング。
- 解析式は良いが有限開口で悪い：開口・照明・伝搬の数値条件。
- 有限複素場は良いがCGH再生で悪い：符号化、8-bit量子化、回折次数フィルタ。

Stage Aの `angle(E)` だけのCGHでは目標振幅を失う。その画像の一致だけでLSHの再生を保証できないため、今回は論文の補足Note 5の振幅符号化を用いる。`J1(f(a))=0.58a` として、位相 `f(a) sin(phi+carrier)` を表示する。外側の照明は実際に開口で0にし、外側の位相0を遮光の代わりにしない。

比較用のCGH再生場は既知の定数0.58だけを補正する。画素ごとの補正や、画像ごとの最大値による別々の規格化は行わない。実際のセンサー強度に対応する未補正データは、補正後強度の0.58²倍で、NPZに別保存する。一次回折光へ入ったパワーの割合もsummary.jsonに記録する。

## 出力ファイル

標準保存先は `lsh_digit1_theory_outputs_v1/`。

| ファイル | 内容 |
|---|---|
| `digit1_sheet_comparison.png` | 目標・解析式・有限複素場・CGH再生のx–z比較 |
| `sheet_target_xz.png` | 目標の数字1 |
| `sheet_analytic_infinite_xz.png` | 全Bessel成分を直接計算した無限開口のシート |
| `sheet_finite_complex_xz.png` | 有限複素場を角スペクトルで伝搬したシート |
| `sheet_cgh_replay_xz.png` | 位相CGH＋理想4fからのシート（既知の定数ゲイン補正） |
| `thread_profile_design.png` | 各Threadの目標と有限Fourier級数による単独応答 |
| `digit1_cgh_8bit.png` | 位相CGH（仮想1:1数値系） |
| `sensor_xy/` | z=0,100,250,400,500 mmに近い撮影位置でのx–y強度画像。赤線は抽出するy=0 |
| `digit1_design.npz` | x,z座標、Thread位置、全モード、複素係数 |
| `digit1_initial_fields.npz` | z=0の有限複素場、4f再生場、照明 |
| `digit1_sheet_data.npz` | x–z強度と複素場。CGHの未補正センサー強度も含む |
| `sensor_frames.npz` | 各撮影位置の2Dセンサー強度。`cgh_raw_index...` は未補正 |
| `summary.json` | SSIM/PSNR/SNR、複素場誤差、符号化・伝搬条件 |
| `manifest.json` | この実行で保存されたファイル一覧 |

評価範囲はx=±0.9 mm、z=0〜500 mm。SSIM/PSNRは強度の一致度であり、全画面の大きな黒い背景だけで評価しない。目標との一致、無限開口と有限開口との差、複素場とCGH再生との差をそれぞれ保存する。複素場の伝搬誤差は位相も含むので、強度画像だけより厳しい確認となる。

角スペクトルの横断線は2D伝搬を1Dへ近似したものではなく、ky成分を先に加算して得た**2D伝搬場のy=0での値**。撮影位置では2D IFFTも行い、センサー全画面を保存する。入力は真のy=0画素を持つ座標 `(arange(N)-N//2)*pitch` で定義している。

この座標原点はStage Aの `(arange(N)-(N-1)/2)*pitch` と半画素異なる。今後圧縮法と比較するときは、辞書側も同じ座標で作るか、対応を明示してから比較する。

2倍のパディングで収束したとは自動判定しない。`config = theory.Digit1Config(check_padding=True)` にすると、代表センサー位置で3倍パディングと比較する。帯域制限で除いたスペクトルのパワーも記録する。CGH画像・計算結果が似ていても、センサーのノイズ、画素積分、実レンズ、SLM位相LUT、倍率・反転・入射光は別途実機に合わせる必要がある。

## 実行

Notebookを上から順に実行する。設定を変えて再計算する場合は `output_dir` を新しい名前に変える。同じ設定の完了結果なら再計算せず読み込む。

```powershell
python -m pip install -r requirements-stage-a.txt
python -m unittest test_lsh_digit1_theory
python lsh_digit1_theory.py --output-dir lsh_digit1_theory_outputs_v1
```

まず複素場の理論伝搬だけに限定する場合：

```powershell
python lsh_digit1_theory.py --no-cgh --output-dir lsh_digit1_complex_only_v1
```

Pythonテストと数値計算は、ユーザーが実行する。コード追加時点では未実行。

## 参照

- Dorrah et al., [Light sheets for continuous-depth holography and three-dimensional volumetric displays](https://doi.org/10.1038/s41566-023-01188-y), Methods・Fig. 2。
- 同論文の [Supplementary Information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41566-023-01188-y/MediaObjects/41566_2023_1188_MOESM1_ESM.pdf), Note 5。
- Zamboni-Rached, [Frozen Waves](https://arxiv.org/abs/physics/0407128)：縦方向のFourier係数とBessel重ね合わせ。
- Arrizón et al., [Pixelated phase computer holograms for the accurate encoding of scalar complex fields](https://doi.org/10.1364/JOSAA.24.003500)。
- Matsushima & Shimobaba, [Band-limited angular spectrum method](https://doi.org/10.1364/OE.17.019662)：角スペクトル伝達関数のサンプリング問題。
