# 感度・Hessian 実装マップ

[English edition](../docs/sensitivity_implementation_map.md)

同じ量 (位相速度 c、群速度 U、H/V、それらの感度 dc/dp・dU/dp・d(H/V)/dp、
Hessian d²c/dp²) を計算する経路が複数ある。互いをクロスチェックしながら開発した
経緯でこうなっており、どれが本番でどれが検算かが名前から読めない。その対応表。

日付・旧コミット・性能数値を伴う段落は開発時の記録であり、現行版の性能保証ではない。
当時の診断スクリプトや詳細レポートの一部は研究アーカイブに保持されている。
公開版の利用方法は [共通API](public_api.md) と [低水準API](lowlevel_api.md)、
現在実行できる回帰試験は末尾の参照を使う。

## 分類の軸

機能として効いているのは **variational か否か** と **AD を何段重ねるか** の 2 つで、
残りは付随的な軸（モデル、縮約）。

### variational は微分階数を 1 つ前借りする

これが分類の要になる。variational 経路は 1 階微分を**閉形式**で持っているので、
同じ階数を得るのに AD が 1 段少なくて済む。

| 得たい階数 | variational | 生の伝播子 |
|---|---|---|
| 1 階 (dc/dp, dU/dp) | **AD 0〜1 段** — 変分勾配の閉形式、または `U=I₃/(c·I₁)` を 1 階 AD | **AD 1 段** — 伝播子を AD で貫き IFT |
| 2 階 (d²c/dp²) | **AD 1 段** — 変分勾配をもう 1 階 AD。**O(L·N)** | **AD 2 段（nested）** — `HessianAD<N>`。**O(L·N²)** |

閉形式の変分勾配を微分するため、**「variational Hessian は
2 階なのに AD 1 段」**が variational の効きどころで、`_grid` の Hessian が
full-AD ではなく variational なのもこれが理由。

nested AD（AD 2 段）が今も使われているのは 2 か所だけ:
`u_kernel_mode="full_ad"` の dU/dp と、LAYER_CAKE の `HessianAD<N>` Hessian。
U の**値**も full-AD だが、これは各レベル 1 回で全体の 0.1%。

### 付随する軸

| 軸 | 見分け方 |
|---|---|
| **モデル** | `layers, thicknesses, half_space` → **LAYER_CAKE**（固定層、外挿なし）<br>`depths, params, is_discontinuity` → **GRID**（node ベース、2ⁿ 分割して Neville 外挿） |
| **縮約** | 末尾 `_reduced` → 基底方向だけ seed。AD パス数が `n_nodes*6` から `P` へ |

`debug_*` は内訳を出す診断で、本番経路ではない。

### 埋まっていないマス（2026-08-11 埋まった）

上の表を実装に当てはめると、**GRID × variational × 1 階（dc/dp）だけが空だった**。

| | LAYER_CAKE | GRID |
|---|---|---|
| 1 階 variational | `compute_kernels_variational_layer_cake` | `compute_kernels_variational_grid` |
| 1 階 生の伝播子 | `compute_kernels_ad_layer_cake` | `compute_kernels_ad_grid` |
| 2 階 variational | `compute_hessian_variational_ad_rayleigh_layer_cake` / `_love_layer_cake` | `compute_hessian_variational_ad_*_grid[_reduced]` |
| 2 階 生の伝播子 | `compute_hessian_ad_layer_cake` | 無し（不要） |

**空きマスは `compute_kernels_variational_grid` で埋まった**。命名は
[命名規則](naming_convention.md) に従う。
中身は `..._grid_reduced` が返す `gradient_node` と同じ量で、単独公開できていなかった
だけだった。現在は勾配自身の level 間相対変化で停止する専用 1 階 loop で実装され、
`P=0` Hessian の空行列 norm を収束判定に流用しない。GRID で変分勾配が欲しいときに
Hessian 関数を副産物目的で呼ぶ必要は無い。

### GRID の root は direct dispersion が所有する（2026-08-26）

各 Romberg level の離散化 root $c_k$ は `compute_dispersion_romberg_levels`
が保持する。AD kernel、単独 variational kernel、full/reduced Hessian、界面深さの
外挿 gradient/Hessian は `phase_velocity_roots` と `phase_accepted_level` が渡された場合、
accepted prefix の外挿値を direct phase velocity と照合した後、この $c_k$ を残差検証して
直接使う。外挿後の $c$ から各 mesh で Newton/scan をやり直さない。

これは単なる cache ではなく branch identity の契約である。direct が Love0 を
選んだ既知点で、旧 variational の level-0 局所窓は Love0 を含まず Love1 を選んだ。
trace が短い、accepted phase が一致しない、または direct と同じ $|F|<0.1$ の残差契約を
満たさない場合は fail closed とし、別 root を勝手に探して継続しない。追加 level は
branch 指定を必須として mode order で生成する。public caller は配列を継ぎ合わせず同じ
direct solution 由来の trace を渡す。`ForwardEngine` は `ModeSolution` 内でこの
provenance を保持する。trace API v1 は外挿開始 level 0 のみを受理し、通常 forward は
derivative 用の余分な refinement level を生成しない。詳細・評価回数・回帰条件は
[Rombergのroot trace契約](../docs/Romberg.md) と
`tests/python/test_variational_root_trace_consistency.py` を参照。

## 1 階: 感度カーネル

| binding | モデル | dc/dp | dU/dp | 外挿 |
|---|---|---|---|---|
| `compute_kernels_ad_grid` | GRID | **AD**（528 方向 `AutoDiffScalar<VectorXd>`、IFT `-dF/dp÷dF/dc`） | `u_kernel_mode`: `variational_frequency_ad`（既定、位相変分kernelを周波数1方向AD）/ `variational` / `full_ad` | 各レベルで計算 → Neville（`kernel_extrapolation` 既定 True） |
| `compute_kernels_ad_grid_velocity_iso` / `_vti` | GRID（速度基底） | 同上 | 同上 | 同上 |
| `compute_kernels_variational_layer_cake` | LAYER_CAKE | **変分（閉形式）** | `u_kernel_mode`: `variational_frequency_ad`（既定、位相変分kernelを周波数1方向AD）/ `variational`（dual, N-pass） | なし |
| `compute_kernels_ad_layer_cake` | LAYER_CAKE | full-AD | nested AD | なし |
| `compute_kernels_fd_layer_cake` | LAYER_CAKE | 有限差分 | 有限差分 | 参照用 |
| `compute_interface_kernels_variational_rayleigh_layer_cake` / `_love_layer_cake` | LAYER_CAKE | 界面カーネル K_d | — | なし |

GRID の変分カーネル単独入口は `compute_kernels_variational_grid`
（上記「埋まっていないマス」参照、2026-08-11 に追加）。

LAYER_CAKE の `variational_frequency_ad` は Rayleigh の解析積分と Love の材料カーネルで
有効になる。Rayleigh の台形積分（fluidを含む）と Love の層厚成分は、数値整合性を
保つため旧 `variational` 経路へ成分単位でfallbackする。層厚・不連続面カーネルを
省略するのではない。

### LAYER_CAKE kernel 経路のスイッチ

| kwarg | 既定 | 制御対象 |
|---|---|---|
| `compute_c` / `want_kernel_u` / `want_kernel_hv` | `True` | dc/dp, dU/dp, d(H/V)/dp。U/HVの未要求部分をゲートし、出力は0にする（dUには内部dc/dpが必要） |
| `u_kernel_mode` | `"variational_frequency_ad"` | dU/dp の解法。`"variational"` は旧parameter-width energy-integral AD |
| `compute_thickness` | `False` | 固定層厚の微分。Loveでは要求時に厚さ成分のみ旧経路を使う |
| `use_analytical` | `False` | Rayleigh層内積分。周波数ADは `True` のsolidモデルで有効 |

### GRID kernel 経路のスイッチ

`compute_kernels_ad_grid` の kwarg。**dc/dp・dU/dp・dHV/dp は同一の
AD パスの中で計算され、フラグは出力のゼロ埋めではなく計算自体をゲートする**。

| kwarg | 既定 | 制御対象 |
|---|---|---|
| `compute_c` / `want_kernel_u` / `want_kernel_hv` | `True` | **カーネル** dc/dp, dU/dp, d(H/V)/dp |
| `want_value_u` / `want_value_hv`（旧 `phase_only`） | `True` | **値** U, H/V（各レベルの nested full-AD 評価）。`False` は `want_kernel_u=want_kernel_hv=False` を要求する |
| `u_kernel_mode` | `"variational_frequency_ad"` | dU/dp の解法。旧energy-integral parameter ADは `"variational"`、nested ADは `"full_ad"`。非有限値は既存経路へfallback |
| `kernel_extrapolation` | `True` | カーネルを全レベルで計算して Neville 外挿するか |
| `adaptive_root_scan` | `True` | 根探索（P2、5.8x） |
| `phase_velocity_roots` | `None` | direct Romberg が選んだ level root。指定時は各 level の独立 root 探索を禁止し、残差不一致を例外にする |

**`want_value_u`/`want_value_hv` と `want_kernel_u`/`want_kernel_hv` は別物。**
律速しているのはカーネル側で、値の側ではない — 88 ノードの shallow grid での
実測（2026-08-11）:

| 構成 | s/datum | 落とした対象 |
|---|---:|---|
| `want_kernel_u=T, want_kernel_hv=T`（旧既定） | 6.833 | — |
| `want_kernel_u=F, want_kernel_hv=F` | 1.091 | **カーネル dU/dp, dHV/dp — 84%** |
| ＋ `want_value_u=want_value_hv=False`（旧 `phase_only=True`） | 1.083 | 値 U, H/V — 0.1% |

`c` と dc/dp はどの構成でもビット一致。`kernel_extrapolation` は別途約 2 倍を占める
（研究アーカイブでの計測値: 外挿なし 0.1085 / あり 0.2020 s）が、dU の精度を
2e-4 → 1.6e-9 に上げているもの。

2026-08-20以降、dU-onlyでは `variational_frequency_ad` が位相変分カーネル
$K_p=dc/dp$ を
$dU/dp=U^2 d(\omega K_p/c^2)/d\omega$ でwidth-1 AD微分するため、群速度部分の
複雑度は $O(LP)$ から $O(L+P)$ になる。`compute_c=False` かつ
`want_kernel_hv=False` では幅$P$のAD model自体を生成しない。H/V kernelを要求した
場合はH/V部分のみ従来どおり幅$P$である。現在の検証は
`tests/python/test_u_kernel_frequency_variational_grid.py` と
`tests/python/test_u_kernel_frequency_variational_layer_cake.py` を参照。

## 2 階: Hessian

| binding | モデル | 実装 | 外挿 |
|---|---|---|---|
| `compute_hessian_ad_layer_cake` | LAYER_CAKE | full-AD `HessianAD<N>`（nested 2 階、O(L·N²)）。無印・`_full_ad`・`_reference` の3名が統合され、この1つが正名（2026-08-11、Rayleigh/Love とも動作確認済み） | なし |
| `compute_hessian_variational_ad_rayleigh_layer_cake` / `_love_layer_cake` | LAYER_CAKE | variational（変分勾配を 1 階 dual で微分、O(L·N)）。Love 側が本番実装 | なし |
| `compute_hessian_variational_fd_love_layer_cake` | LAYER_CAKE | 変分勾配の有限差分。O(L·N²)、参照用 | なし |
| `compute_hessian_variational_ad_rayleigh_grid` / `_love_grid` | GRID | variational（`grid_hessian.hpp`）。全ノード列 | あり |
| `..._rayleigh_grid_reduced` / `..._love_grid_reduced` | GRID | **基底縮約版**。`J` を受け取り基底方向だけ seed | あり |
| `compute_interface_hessian_row_ad_grid` | GRID | 界面深さ列（nested forward AD） | — |
| `compute_interface_hessian_row_variational_ad_grid[_extrapolated]` | GRID | 界面 Hessian 行（variational） | 後者はあり |

旧 `_hybrid_love` / `_variational_love` / `debug_variational_love_hessian_breakdown`
は2026-08-11に削除済み（下記「Love の Hessian」参照）。

**GRID の Hessian は full-AD ではなく variational。** full-AD Hessian (`HessianAD<N>`)
は LAYER_CAKE のみ。

`_reduced` は node-VTI 単位ベクトルの代わりに基底 `J` を seed するので、AD パス数が
`n_nodes*6` から `P` に落ちる（528 → 約 19）。C++ 側は n_nodes*6 の正方 Hessian を
作らない。返すのは `gradient_node`（未射影、曲率補正に必要）と `hessian_reduced`。

### Love の Hessian は名前 4 つに対し実装 3 つ — hybrid 系は削除済み・検算用の履歴 (2026-08-11)

過去には `compute_phase_velocity_hessian_weakform_love`（本番）、
`_hybrid_love`（過渡実装、固有関数を AD 伝播子から回収する 2 段 nested AD、
O(L·N²)）、`_variational_love`（`_hybrid_love` への 8 行転送、実装を持たない
別名）、`_variational_fd_love`（勾配の有限差分、参照用）の 4 名に実装 3 つが
対応していた。`inversion/` からの使用がゼロで `tests/python/` の
`diagnose_*` 5 本 + 研究アーカイブの `verify_hessian_taylor_random.py` のみだったため、
`_hybrid_love` とその別名 `_variational_love`・内訳診断
`debug_variational_love_hessian_breakdown` を削除し、7 本の呼び出し元を
更新した。判断の根拠（`77e566d`→`1de506a`→`2cdf53b` の履歴、
「hybrid は固有関数を AD 伝播子から取るため variational の利点を得ていない」
という分類）は研究アーカイブの履歴に基づく。公開版で使用する名前は
[命名規則](naming_convention.md) を参照。

現在の Love Hessian は canonical に 2 つ:
`compute_hessian_variational_ad_love_layer_cake`（本番、旧 `_weakform_love`。
IFT シードを `computeLoveIFTGradientFirstOrder` で作り、そのうえで
variational の Hessian を組む。`wf-love-seed-gradient` で `HessianAD<80>`
の完全 2 階 AD から置き換えた箇所で、10 層 Iso 1.45 → 0.079 ms）と
`compute_hessian_variational_fd_love_layer_cake`（勾配の FD、O(L·N²)、参照用、
旧 `_variational_fd_love`）。

ここでいう **Hybrid Hessianはチェック・デバッグ専用の歴史的経路**である。
Variational ADが安定するまでの実装確認ではFull ADとの比較に使ったが、
`inversion/`の本番経路には入れていない。現在、Hessianの役割は
「Variational ADを本番、Full ADを独立reference、必要に応じてVariational FDを
数値検算」とする。したがって、古いテストや履歴に残る`hybrid`という名前を
現行の推奨API・本番手法と解釈してはならない。

[命名規則](naming_convention.md) のとおり、`weakform` という語自体も
`variational` の同義語として退役している（実体は Takeuchi & Saito (1972) の
ラグランジアン停留性で、FEM の意味の弱形式ではない）。`0d89d0f`
"Rename WeakForm → VariationalFD for clarity" が過去に逆方向へ振り替えた
のがこの混乱の直接の原因だった。

`diagnose_prem_grid_love_layer_cake_bridge.py:365-383` はかつて4経路を比較して
いるように見えて実質3経路だった（`variational` と `hybrid` が同じ関数への
別名で、その2行は自分自身と比較していただけ）。この問題も `_hybrid_love`
削除で解消済み。

**`backend` 引数の既定が層で違う。** binding は `"dual"`（N-pass 参照実装）、
`ForwardEngine` は `"vector"`（ブロック化 vectorized、grid Rayleigh 3.4x）。
開発時の比較では結果は 3e-12 以内で一致した。この数値は研究アーカイブの
過去の比較記録であり、公開版での新たな測定値ではない。

## 同じ dc/dp が 2 実装ある — 本番と検算

GRID の dc/dp は 2 経路あり、**パイプラインは両方を計算して突き合わせている**
(`pipeline.py:214-255`):

```python
G_hess = gradient_log(model0, res["g_a_variational"])
J[i] = project_gradient(G_hess, basis)                      # 本番は variational 側
if "g_a_ad" in res:
    G_kernel = gradient_log(model0, res["g_a_ad"])           # AD Romberg 経路
    ad_row = res.get("g_inv_kernel", project_gradient(G_kernel, basis))
    consistency[i] = max|J[i] - ad_row| / scale              # 検算
```

- **本番の Jacobian は variational 側** (`res["g_a_variational"]`、reduced Hessian
  呼び出しの `gradient_node`。深さ成分は `depth_config` があるとき variational K_d
  経路 `g_variational_depth` から)
- **AD 経路（`compute_kernels_ad_grid`）は独立な検算**で、
  `consistency` として記録される。`ad_kernel_every` で間引かれた点は
  `consistency` が NaN になる
- Hessian 本体は variational 側 (`hessian_reduced`)

コスト（shallow 24 点、逐次 1 点あたり、2026-08-11）:

| 段 | 実装 | ms/datum |
|---|---|---:|
| forward | `compute_dispersion_romberg` | 29 |
| Jacobian | AD 経路（dU/dp・dHV/dp off 後） | 733 |
| Hessian 増分 | variational `_reduced`（`gradient_node` 込み） | 36 |

**検算側のほうが本番側より桁で安い。** 精度差は 2026-08-11 に
AD 側のバグを直して解消した（下記「原因は AD 経路に紛れ込んだ片側差分だった」）。
両者とも FD に対して 1e-08 級で並ぶ。

### 方針 (2026-08-11 決定・同日実施完了)

**実用は variational カーネル、AD は検証用。** 精度はもう理由ではなく、
**コスト**が理由だった（733 ms 対 36 ms）。AD 側は独立実装として検証に据えられる
状態になり、B1/B2/B3（下記「実用経路の切り替え」）として同日中に実施済み。

### 決着までの実測

`consistency` は 42 ラン 2805 点ぶんが 研究アーカイブの実行結果に保存済み。

| 分位 | consistency |
|---|---:|
| p50 | 5.9e-08 |
| p90 | 1.2e-06 |
| p99 | 5.2e-05 |
| 最悪 | **4.0e+00** |

p50 の 5.9e-08 は「両者が各自の精度限界まで一致」を意味する。AD 経路のカーネル自体の
打ち切り誤差が 1.7e-08 （研究アーカイブの精度評価） なので、これ以上は良くならない。
問題は裾だけ。

外れ値の正体は、消去法で絞った:

- **モードの読み飛ばしではない。** 最悪点 (consistency 4.0) の `rel_err` は 5.4e-11、
  `branch_confirmation` も `accepted`。根そのものは機械精度近くまで収束している。
  外れ値が孤立せず周期方向に滑らかな山を描くことも、飛びとは整合しない。
- **`c≈Vs` の条件数崖でもない。** 悪い点が層 Vs に近いとは限らず、良好点のほうが
  近い場合すらある。L1_10s では `|c-Vs|/c = 0.65` で無関係。
- **すべて branch 1 (1 次オーバートーン)。** 基本モードは事実上すべて健全。

有限差分を第三の基準に立てると、どちらが誤っているかが決まる (L1_10s, 18 基底方向):

| | FD との相対誤差 |
|---|---:|
| variational | **1e-08 級** |
| AD | **-3.85e-05（全方向で一様）** |

`consistency` の保存値 3.84e-05 と AD 誤差 3.849e-05 が一致する。つまり
**`consistency` はほぼそのまま AD 側の誤差を測っていた**。

### 原因は AD 経路に紛れ込んだ片側差分だった (2026-08-11 修正)

誤差は方向ごとにばらつかない。`g_AD/g_FD` の平均 0.999961511、**ばらつき 5.2e-07** —
単一のスカラーがずれている。`dc/dp = -F_p/F_c` の**分子だけ AD、分母は片側差分**
だったのが原因。しかも 2 つの評価が `NormalizationMode::Vertical` の別呼び出しで、
`Vertical` は走行中の `Y(0)` で割るため**ゲージが c 依存** (`s(c)·F(c)`)。根では
`F = 0` なので

```
dF/dc(計算値) ≈ s(c+ε)·F_c      dF_dp = s(c)·F_p   (同一 AD パス、一貫ゲージ)
→ 相対誤差 −(d ln s/dc)·ε が分母だけに乗る = 全方向一様なスカラー倍
```

オーバートーン限定なのは、節で `Y(0)` がゼロ近傍を通り `s` が急変するため。基本モードは
ゲージが緩やかなので目立たなかった。

**修正**: ゲージを `Provided` で固定し、`F_c` も幅 1 の AD (`ScalarADc`) で厳密に取る。
同じ形が `dU/dc` (full_ad・流体経路) と `dHV/dc` にもあり、計 3 か所を直した。

| ケース | 修正前 | 修正後 |
|---|---:|---:|
| L1_10s (問題の点) | −3.85e−05 | **1e−08 級** |
| L0_10s | −2.39e−08 | 1.6e−09 |
| `consistency` L1_10s | 3.84e−05 | **2.81e−11** |
| `consistency` L0_10s | 2.00e−08 | **6.53e−12** |
| AD/FD 比 (L1_10s) | 0.999961511 | **0.999999986** |

健全に見えていた基本モードも 2〜3 桁改善した。以前の「良さ」は**ゲージ誤差と片側差分
誤差が偶然打ち消し合っていた**もので、ゲージだけ直すと L0_10s は 2.4e−08 → 2.5e−06 と
悪化した (片方だけ直すのは危険という実例)。

### 否定された仮説 — 再検討しないこと

- **レベル数不足ではない。** `kernel_extra_levels` を実装してレベルを 4→5→6 と
  増やしたが**誤差は 3.85e−05 のまま完全に不変**、コストだけ 4 倍。収束次数も
  2.00〜2.03 で正常だった。この kwarg は削除済み。
- **モードの読み飛ばしではない。** 最悪点で `rel_err` 5.4e−11、
  `branch_confirmation` も accepted。
- **`c≈Vs` の条件数崖でもない。** L1_10s は `|c−Vs|/c = 0.65`。

### 残した片側差分 — Newton 反復のヤコビアン (14 か所)

`sensitivity_utils.hpp` 1427/1515/1735/1907/3184/5277/5443/5853/5999/6081、
`bindings.cpp` 1434/1872、`grid_hessian.hpp` 348/545。
`df = (f(c+eps) − f(c))/eps` を Newton の勾配に使うだけで、**不正確なヤコビアンは
反復経路を変えるが不動点を動かさない**。根の精度は停止条件が決める。AD 化しても
精度は変わらず反復ごとにコストが増えるだけなので、意図的に残す。

設計上の FD (参照実装): `compute_kernels_fd_layer_cake`、
`compute_hessian_variational_fd_love_layer_cake`、
`FdDepthHessianBackend`。

死んだコード: `computeNumericalDerivative` (`disper26.cpp:1953`) は呼び出し 0 件。

未着手の確認: `Depth` カーネル (界面深さ列) は AD 側の `Depth` エントリから
作られている (`depth_backend.gradient_row`)。variational の `gradient_node` に
相当物があるかは未確認で、2 を実施するならここが要る。

歴史的な注記: `compute_kernels_ad_grid`（旧 `compute_dispersion_romberg_with_kernels`）
の kernel が単一レベルでしか計算されていなかったバグ (2026-07-05 `e5faa1d` で修正)
が長く見つからなかったのは、Taylor 検証 (研究アーカイブの `verify_hessian_taylor_random.py`) が
variational 側の gradient を使っていて、バグのある AD 側を通っていなかったため。
修正後は両者 ~1e-8 で一致する。

## inversion 側から見える範囲

`ForwardEngine` が公開しているスイッチ:

| 引数 | 既定 | 効果 |
|---|---|---|
| `want_kernel_u` / `want_kernel_hv` | `False` | dU/dp・dHV/dp を計算するか（上表の 84%） |
| `hessian_backend` | `"vector"` | variational Hessian の AD backend |
| `max_steps` / `tolerance` | 6 / 1e-10 | Romberg |
| `hessian_max_levels` / `hessian_tolerance` | 5 / 1e-8 | Hessian のレベル |

`compute_c` / `u_kernel_mode` / `kernel_extrapolation` / `want_value_u`・`want_value_hv`
（旧 `phase_only`）は**engine から渡していない**（C++ 既定のまま）。
`want_value_u`/`want_value_hv` を使わないのは、値の側の削減が 0.1% しか無いうえ
`solution.U` が NaN になり、`trace_dataset` の `_group_predict` が warm-start
ヒントにその U を使うため。

`derivatives.py` が読むのは `kernel.d_c` のみ。`d_U` / `d_HV` は `inversion/` の
どこからも読まれておらず、`Datum` も `(wave, branch, period)` だけで U・楕円率の
観測量を持たない。だから既定を off にしてある — 観測量を足すときに on にする。

`map_over_data` / `run_phase1` が公開しているスイッチ（`parallel.py`/`pipeline.py`）:

| 引数 | 既定 | 効果 |
|---|---|---|
| `ad_kernel_every` | `1`（全点で AD、旧挙動） | AD カーネルパスを何点に 1 回走らせるか。`J` は常に variational + K_d、AD は `consistency` のためだけ |

`Depth` 成分は `depth_backend.depth_gradient_variational`（既定の `variational`
バックエンドが持つ）から来る。持たないバックエンドを明示選択したまま
`ad_kernel_every` で AD を間引くと `RuntimeError`。

## 命名整理

**命名規則は [命名規則](naming_convention.md) に記載する。**
`m.def` 74名 + `inversion/` 公開名の全数監査と実リネームは
研究用の `naming-convention-rename` ブランチで完了した（2026-08-11）。
効いていた軸は variational か否かと AD 段数なのに
名前は `weakform` / `variational` / `full_ad` / `hybrid` / 無印が混在して対応
していなかった問題（`0d89d0f` の意味の振り替え）、Love の Hessian 4 名の整理
（`_hybrid_love` とその別名2つを削除）、`d_c`/`d_U`/`d_HV` と
`compute_c`/`want_kernel_u`/`want_kernel_hv`/`want_value_u`/`want_value_hv`
の関係の明示、GRID の 1 階 variational 単独公開（`compute_kernels_variational_grid`）
の実施記録は研究アーカイブにある。公開版の命名規則には、監査表全体を同梱していない。
この文書内の関数名は現時点の正名に更新済みで、以下の「実用経路の切り替え」節も
その表記に合わせてある。

GRID の 1 階 `K_d`（界面深さ勾配 dc/dz、`compute_interface_gradient_variational_grid`）
の単独公開は B3（下記）で完了済み。

### 実用経路の切り替え（B1/B2/B3）— 完了 (2026-08-11)

精度は 2026-08-11 の修正で両経路とも 1e-08 級に並んだので、残る理由はコスト
(733 ms 対 30〜77 ms、**約 16〜20 倍**)。予定どおりの順で実施し完了した:

1. **B2**（`cfa628a`）— AD と variational を突き合わせる比較オプション
   `ad_kernel_every`。既定 1（全点、旧挙動）。安全網を先に用意。
2. **B3**（`bfe6002`）— GRID の 1 階 `K_d` 単独入口
   `compute_interface_gradient_variational_grid`（当時の名は
   `compute_interface_gradient_variational_extrapolated`）。AD の `Depth` と
   7.7e-12 で一致、double のみで 2〜6 ms。
3. **B1** — `pipeline.py` の `J[i]` を variational + K_d へ切り替え。**役割が反転**:
   本番が variational、AD は `consistency` のためだけの検算になった。

**役割反転後の構成**:

- 速度成分: `g_a_variational`（reduced Hessian 呼び出しの `gradient_node`、常時計算）
- 深さ成分（`depth_config` あり）: `VariationalAdDepthHessianBackend.depth_gradient_variational`
  （K_d、AD 不要）。既定バックエンドがこれを持つ。持たないバックエンド
  （`fd`、無印の `ad`）を明示選択し、かつ `ad_kernel_every`
  で AD を間引いた場合は、深さ成分の情報源が無くなるため `RuntimeError`
  ——**黙って誤った Jacobian 行を作らない**
- `consistency`: `ad_kernel_every` が AD を回した点でのみ有限値、他は NaN
- `VariationalAdDepthHessianBackend.interface_hessian_row`（Hessian 列、旧 `column`）
  は元々 `level` しか要らず `dc_dz` は不要だったので、`depth_gradient_variational`
  が呼ばれた時点でレベルキャッシュを埋める。これにより
  `interface_hessian_row()` 側の「AD スキップ時に自前で
  `phase_velocity_with_kernels` を呼んでコストが移動するだけ」という制約
  （B2 コミットメッセージの caveat）も、既定バックエンドでは実質解消した

**検証**: 新テスト `test_phase1_weakform_jacobian.py` で `ad_kernel_every=1` と
`0` で **J がビット一致**すること、`consistency` の分位が 2.8e-11 に収まることを
確認。研究アーカイブの `verify_inv_phase1.py`（24 点、serial/pool 8 並列）で `consistency` 最大
6.33e-11（旧しきい値 1e-4 の 4 桁下）、pool と serial の J/H はビット一致。

**踏んだ落とし穴**: `depth_gradient_variational` の呼び出しを最初
`depth_backend is not None` だけで無条件ゲートしたところ、`landing.py` が
`want_hessian=False, want_kernels=False` で呼ぶ「勾配不要の純フォワード評価」
（LM のライン探索中）にまで K_d コストが付いた。`(want_kernels or want_hessian)`
を条件に足して修正 —— AD カーネルパスの既存ゲートと揃える形。

## 参照

- [命名規則](naming_convention.md) — 公開名と互換名
- [GRID variational Hessian](grid_wf_hessian.md) — 設計履歴と外挿
- [Romberg](../docs/Romberg.md) — 加速・root trace・kwarg
- [共通API](public_api.md) — パラメータ軸と微分出力
- [出力選択の回帰試験](../tests/python/test_kernel_target_switch.py)
- [root traceの回帰試験](../tests/python/test_variational_root_trace_consistency.py)
- [GRID群速度カーネルの回帰試験](../tests/python/test_u_kernel_frequency_variational_grid.py)
- [ADカーネル間引きの回帰試験](../tests/python/test_ad_kernel_thinning.py)
