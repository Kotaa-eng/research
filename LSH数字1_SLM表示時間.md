# CGHの計算開始からSLM表示までの時間

入口は `lsh_digit1_slm_timing.ipynb`。以前の `cgh_encode_and_ideal4f_replay` はCGH作成に加え検証用FFTフィルタを含んでいた。この計測はマスク作成だけを `encode_phase_mask()` に分離し、4f再生・伝搬・画像保存をオンライン時間から外す。符号化式と8-bit量子化は従来と同じ。

## 計測の範囲

`INCLUDE_COEFFICIENT_DESIGN=True` は目標の数字から係数を作る処理の開始が時刻0。Falseなら係数が準備済みの時刻を0とする。いずれもSDK接続・デバイス初期化、圧縮辞書の構築/読込、座標配列の準備は事前処理である。

1. 目標からThread係数を求める（選択可能）。
2. 提案法または直接法で初期複素場を生成する。
3. 既知の共通スケールで正規化し、符号化する開口領域を設定する。
4. 振幅情報を含む8-bit位相CGHを生成する。
5. 任意の校正済み `FRAME_TRANSFORM` を適用し、SLMサイズ・uint8・連続メモリを検査/準備する。
6. 任意の `DISPLAY_CALLBACK(frame)` へ渡し、その同期関数が戻るまでを計測する。

標準は提案法・ランク100・1回ウォームアップ後5回測定。同じ数字1を繰り返し送る。SDKが同じフレームを省略する場合は転送時間が変わるため、実更新の測定では当該フレームIDの完了確認など、使用SDKの動作を確認する必要がある。

## 未接続の場合

`DISPLAY_CALLBACK=None` のままなら、PC上でSLMへ渡す配列の準備までを測る。転送・表示・光学整定は計算せず `null`。保存PNGを作るだけの処理をSLM表示時間として扱わない。

標準CGHは仮想4f面の2048×2048。実機SLMが1920×1200など異なる場合、単純な画像リサイズで物理条件が一致するとは限らない。実機の座標・倍率・位相校正に合うフレーム生成が必要。実機接続には `EXPECTED_SLM_SHAPE=(高さ,幅)` が必須で、サイズ不一致はエラーにする。自動リサイズしない。画素値0は光の遮断を意味しないため、実際の開口/照明も別に整える。

## SDKへの接続

デバイス初期化は計測セルの前に行い、実際のSDKへ接続する同期関数を `DISPLAY_CALLBACK` に設定する。メーカーAPI名はSLM型番・SDKが分かってから実装する。

```python
from lsh_digit1_slm_timing import DisplayReceipt

def display_on_slm(frame):
    # frameは連続メモリの2次元uint8。ここで実際のSDKへ送る。
    # SDKが非同期なら、当該フレームの完了を待つ処理もこの関数内で行う。
    # return DisplayReceipt('submitted', '転送要求が戻っただけ')
    # または、当該フレームの表示完了をSDKで確認した場合：
    # return DisplayReceipt('controller_confirmed', '使用したSDKの確認方法')
    # センサー等で光学的な整定まで実測した場合：
    # return DisplayReceipt('optically_confirmed', '使用した測定方法')
    raise NotImplementedError('使用するSLMの実APIへ接続してください')
```

上のテンプレートはSDK送信を実装していない。接続してからコールバックに指定する。戻り値Noneはsubmitted扱いで、表示済みにはしない。フレームを非同期送信する場合、SDKが参照するバッファの保持/コピーも接続関数側で扱う。

表示APIの完了と液晶応答・光学的整定は別。後者を測る場合はフォトダイオード/センサー等で確認し、コールバックの中で待つ。公称リフレッシュレートの逆数や適当なsleepを実測値として加算しない。

## 結果

`lsh_digit1_slm_timing_outputs_proposed_rank100_v1/`（標準）に保存する。

- `summary.json`：各工程の中央値・最小・最大・p95、計測範囲、接続有無、完了確認方法。
- `timing_samples.csv`：各回の内訳。
- `last_cgh_8bit.png`：最後に生成した元のCGH。
- `last_slm_frame_8bit.png`：最後にSDKへ渡すため準備したフレーム（未接続でも保存）。

主に見るキーは `statistics_seconds` 内の以下。

| キー | 計測内容 |
|---|---|
| `phase_mask_encoding_seconds` | CGH位相マスクの作成だけ |
| `update_to_frame_ready_seconds` | 入力更新から、渡すフレームの準備完了まで |
| `display_callback_seconds` | SDKへの送信と、接続関数内の待機時間 |
| `update_to_api_return_seconds` | 入力更新から表示API接続関数が戻るまで |
| `update_to_controller_confirmed_seconds` | 当該フレームの表示完了をSDK等で確認した場合のみ |
| `update_to_optically_confirmed_seconds` | 光学的な表示・整定を実測確認した場合のみ |

`preparation_seconds` と `offline` はオンラインから除いた準備。物理的な光伝搬時間をシミュレーション計算時間として加算する処理は行わない。

別条件で計測する場合は、正規化用 `reference_scale` を同じ条件の比較結果から読む。Notebookはconfigとモード密度・ランクの整合性を検査する。旧データの別スケールを流用しない。

コード追加時点では実機制御・数値計算・テストを実行していない。ソフトウェアテストは `test_lsh_digit1_slm_timing.py` に用意した。
