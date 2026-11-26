# API 命名規則

[English edition](../docs/naming_convention.md)

Python の利用文書では **高水準 API**（`disper26`）と **低水準 API**（`disper26py`）と呼ぶ。
コード例での表記は、高水準側が `import disper26 as d26`、低水準側が `import disper26py as d26_lowlevel` とする。

## 1. 関数名の文法

```
<verb>_<quantity>[_<method>][_<wave>][_<model>][_<variant>…]
```

旧名 `compute_phase_velocity_hessian_weakform_rayleigh_grid_reduced` と同じ順序で各要素を並べる。

## 2. 統制語彙（各軸1語、同義語禁止）

### verb

数値を計算する新しい関数には `compute` を使う。旧名には `calculate` もある。
`debug_` は診断専用の接頭辞として `compute` の代わりに文頭へ立つ
（例: `debug_hessian_variational_ad_love_layer_cake`）。`create_*`、`search_roots`、
`apply_*`、`earth_flattening_*`、`get_*`、`eval_*`、`propagate_*` など、量の微分経路の
区別と無関係な慣用動詞は本文法の対象外とする。

### quantity

| token | 意味 |
|---|---|
| `dispersion` | c（および U, H/V）をまとめて返す値 |
| `dispersion_curve` | 単調な周期列に沿う位相速度cと収束・fallback診断の一括値 |
| `group_velocity` / `hv_ratio` | U または H/V を単独で返す関数の量（`dispersion` の部分成分を単独公開する場合はこちらを使う） |
| `frequency_curvature` | 実数解のモードブランチ（mode branch）に沿って求めた dU/dω、d²k/dω²、および無次元曲率診断をまとめて返す値 |
| `kernels` | ∂c/∂p の深さ分布（LAYER_CAKE の層別、または GRID のノード別） |
| `gradient` | ∂c/∂p を組み上げたベクトル |
| `hessian` | ∂²c/∂p² |
| `interface_kernels` | 各界面の深さに対する位相速度微分。Layer-cake の変分法では境界カーネル $K_d$ を返す |
| `interface_gradient` | 入力 GRID の深さ座標で求めた界面深さ勾配。Flat の変分法では $K_d$ を幾何重みで足し合わせる |
| `interface_hessian_row` | interface_gradient をもう1階微分した Hessian の行（GRID、2階）。C++ 側の「row」に統一し、Python 側の「column」呼称を退役させる（§5） |
| `eigenfunctions` / `energy_integrals` / `mode_count` / `continuous_mode_count` | そのまま |

### method（微分の計算法）

| token | 意味 | AD段数 |
|---|---|---|
| `variational` | 変位に関する停留性（$\delta_{\boldsymbol u}\mathcal L=0$）により、固有関数を固定して一次感度を評価する変分式 | 0 |
| `variational_ad` | 変分勾配または界面カーネル K_d を1階ADで微分する（従来の弱形式Hessian） | 1 |
| `variational_fd` | 変分勾配を有限差分で微分する | — |
| `ad` | 伝播子をADで微分し、根に陰関数定理を適用する | 1階なら1／2階なら2（nested） |
| `fd` | 位相速度 c を有限差分で微分する | — |

前半が「変分か否か」、後半が「AD段数」で、`sensitivity_implementation_map.md` が挙げる
2軸そのもの。**規則: method は quantity が kernels/gradient/hessian/interface_* の
ときは省略不可。** 無印は禁止（`compute_phase_velocity_hessian` が該当）。
dispersion/group_velocity/hv_ratio/eigenfunctions/energy_integrals/mode_count は
微分を取らないので method 軸自体が無い。

旧名には `weakform`、`hybrid`、`full_ad`、`stable`、`reference` がある。
`hybrid` は過渡的な実装を指し、具体的な計算法は関数ごとに異なる。

### wave

`rayleigh` または `love` を使う。引数で波種を選ぶ関数ではこの要素を省略する。

### model

`layer_cake` | `grid`。**両方明示**する。ただし model の区別自体が存在しない
関数（`kernels`/`hessian` 系以外の大半のユーティリティ）には付けない。
`layer_cake` は一つのモデル種別を表す語として扱う。`grid` の省略で暗黙に表さない。

既存名には GRID の `compute_dispersion_romberg` や
`debug_hessian_ad_love_upward_ift_layer_cake` がある。
引数名にも `cp`、`hs`、`half_space` が残る。キーワード指定時は各関数のシグネチャを参照する。

### variant

`_reduced`（基底縮約）/ `_extrapolated`（レベル外挿）。
**規則: 同名の非外挿版が併存するときだけ `_extrapolated` を付ける。**
GRID Hessianのように常に外挿する関数ではこの接尾辞を省略する。
`compute_interface_hessian_row_variational_ad_grid` と
`compute_interface_hessian_row_variational_ad_grid_extrapolated` は両方の形式を持つ例である。

## 3. 衝突語の裁定（1トークン1意味）

| 語 | 裁定 |
|---|---|
| `Reduced` | 基底縮約のみ。複合行列の5成分表現 `ReducedIndex` は内部実装に残る旧名 |
| `Romberg` | レベル外挿の意味に限定。層内求積（`integrateGridLayer_Romberg`、`propagate_romberg_vector`）は `quadrature` へ |
| `Liquid` / `Fluid` | `fluid` に統一 |
| `Boundary` / `Interface` / `Depth`（K_d） | `interface` に統一 |
| `Bran` / `AdModeCount` / `Total` / `Continuous` | `mode_count`（正）と `continuous_mode_count`（legacy）の2語のみ |
| `Pieces` / `Breakdown` / `Trace` / `Probe` | `debug_` 接頭辞に統一 |
| `Weakform` / `WeakForm` 表記揺れ | 公開名では `variational`。C++ 内部には `WeakForm` が残る |
| `compute` / `calculate` | `compute` |

## 4. kwarg・辞書キー・フィールド名

- 値の要求には `want_value_u` / `want_value_hv`、カーネルの要求には `want_kernel_u` / `want_kernel_hv` を使う。
  `phase_only` も現行の指定である。旧名には値の `compute_u`・`compute_hv`、逆解析の `want_u_kernel`・`want_hv_kernel` がある。
- 低水準関数の微分座標を選ぶ引数は `sensitivity_mode`、波種は `wave_type` とする。
- `SensitivityKernel.d_c` は層・ノード別の感度、`gradient` と `gradient_node` は組み上げた勾配ベクトルを表す。
  出所を区別する場合は `g_a_ad`、`g_a_variational`、`g_variational_depth` のように計算法を添える。
- 界面Hessianの行には `interface_hessian_row` を使う。内部には `DepthHessianColumn.column` が残る。旧 `column()` は転置した量を返した。
- Romberg trace は `phase_velocity_roots`（level ごとの生 root）と
  `phase_accepted_level`（direct phase の外挿 prefix 終端）を必ず組で渡す。
  追加生成深さは `minimum_trace_level`、root solve tolerance は `root_tolerance` とし、
  phase Romberg の `tolerance` と名前を分ける。
- 深さ Hessian のバックエンドの旧名と対応する現行名は次のとおり。

  | 旧 registry key | 旧クラス名 | 旧 `method` 文字列 | 正 registry key = method 文字列 |
  |---|---|---|---|
  | `finite_difference` | `FiniteDifferenceDepthHessianBackend` | `central_fd_of_ad_gradient` | `fd` |
  | `nested_ad` | `AutoDiffDepthHessianBackend` | `nested_ad` | `ad` |
  | `variational` | `VariationalDepthHessianBackend` | `variational_extrapolated` | `variational_ad`（K_d 自体は1階の閉形式=AD0だが、Hessian 列はそれをもう1階ADで微分するため） |

  クラス名は `<Key>DepthHessianBackend` の形（`FdDepthHessianBackend` /
  `AdDepthHessianBackend` / `VariationalAdDepthHessianBackend`）に揃える。
- Phase 1 は Flat の界面に `"variational_ad"`、球面の界面に `"spherical_ad"` を選ぶ。球面バックエンドが返す Hessian 行の計算法名は `method="ad"` である。過去の Phase 1 キャッシュ識別子には `"central_fd_of_ad_gradient"` も使われた。
- 変分HessianのAD実装を選ぶ `backend` は別の設定で、低水準バインディングの既定は `"dual"`、`ForwardEngine.hessian_backend` の既定は `"vector"` である。

## 5. 旧名の扱い

### 共通公開APIの名前（2026-09）

通常のモデル構築は `create_model(depths, kind=..., ...)` に統一する。
`kind` は `layer_cake` / `grid`、物性は等方性・速度VTI・弾性定数の配列を一組だけ指定する。
オブジェクト入力は `create_layer_cake_model` / `create_grid_model` を使う。

`pancake` は legacy 名として維持する。旧関数名・引数・既定値は呼び出し可能なまま残し、
新規コードと文書では `layer_cake` を使う。新たな実行時警告や削除期限は設けない。
`kind="layer_cake"` と `create_layer_cake_model` はモデル種別 `"layer_cake"` を返す。
`kind="pancake"` と `create_pancake_model` は従来どおり `"pancake"` を返す。
この表記は解に保持されるモデル、平坦化、Q 補正、C/Fortran のモデル情報でも保持する。
計算経路は共通とし、種別判定は両方の文字列を受け付ける。

`disper26` / `disper26::api` / `disper26_api` の型付き入口では、モデルと波種は
`Model` / `ModeSolution` が保持し、計算法は `Options.method` が指定する。
この層に限り `compute_kernels`、`compute_hessian`、`compute_hessian_reduced`、
`compute_interface_kernels`、`compute_interface_hessian_row` のmethod/model接尾辞を省略する。
これは既存低水準関数の別名ではなく、入力検証・所有権・軸情報を持つ共通契約である。
既存 `disper26py` の計算法別名とその規則は維持する。
共通APIの `wave`、`mode`、`basis`、`outputs`、`Options`、結果フィールドと軸順は
[共通公開API](public_api.md) を正とする。`mode` はモード番号専用。

### 既存低水準名

低水準Pythonでは正名と旧名を `m.def` に登録し、旧名のdocstringに
`Deprecated alias. Use <canonical name>.` と明記する。C++の転送関数も旧シグネチャを保持する。
新規コードとREADMEでは正名を使い、公開済みの旧名は同じ引数・既定値で呼び出せるようにする。

旧名 `compute_phase_velocity_hessian_reference` は `sensitivity_mode` と
`compute_thickness` を受け取るが計算には使わない。Layer-cake の物性 Hessian
に厚さの行・列は含まれない。
