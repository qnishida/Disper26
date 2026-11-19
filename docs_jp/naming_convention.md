# API naming convention

[English edition](../docs/naming_convention.md)

## 2. 関数名の文法

```
<verb>_<quantity>[_<method>][_<wave>][_<model>][_<variant>…]
```

現行の本番名 `compute_phase_velocity_hessian_weakform_rayleigh_grid_reduced` と
**スロット順が一致**するので、語順の記憶を変えずトークン値だけが変わる。

## 3. 統制語彙（各軸1語、同義語禁止）

### verb

`compute` のみ。`calculate` は退役（`compute` と意味差なし、`disper26.cpp` が古いだけ）。
`debug_` は診断専用の接頭辞として `compute` の代わりに文頭へ立つ
（例: `debug_hessian_variational_ad_love_layer_cake`）。`create_*`、`search_roots`、
`apply_*`、`earth_flattening_*`、`get_*`、`eval_*`、`propagate_*` など、量の微分経路の
区別と無関係な慣用動詞は本文法の対象外とし、§7 で個別に 据置 と判定する。

### quantity

| token | 意味 |
|---|---|
| `dispersion` | c（および U, H/V）をまとめて返す値 |
| `dispersion_curve` | 単調な周期列に沿う位相速度cと収束・fallback診断の一括値 |
| `group_velocity` / `hv_ratio` | U または H/V を単独で返す関数の量（`dispersion` の部分成分を単独公開する場合はこちらを使う） |
| `frequency_curvature` | 実数分散branch上の dU/dω、d²k/dω²、および無次元曲率診断をまとめて返す値 |
| `kernels` | ∂c/∂p の深さ分布（LAYER_CAKE の層別、または GRID のノード別） |
| `gradient` | ∂c/∂p を組み上げたベクトル |
| `hessian` | ∂²c/∂p² |
| `interface_kernels` | 界面ごとの K_d（LAYER_CAKE、生の境界カーネル） |
| `interface_gradient` | K_d を幾何重みで足し合わせた dc/dz（GRID、1階） |
| `interface_hessian_row` | interface_gradient をもう1階微分した Hessian の行（GRID、2階）。C++ 側の「row」に統一し、Python 側の「column」呼称を退役させる（§5） |
| `eigenfunctions` / `energy_integrals` / `mode_count` / `continuous_mode_count` | そのまま |

### method（今回の主眼。微分をどう取るか）

| token | 意味 | AD段数 |
|---|---|---|
| `variational` | 変分原理の閉形式（変位凍結、δ_uL=0） | 0 |
| `variational_ad` | 変分勾配（または K_d）を1階ADで微分 ← 現 `weakform` Hessian、`_row_variational_extrapolated` | 1 |
| `variational_fd` | 変分勾配をFDで微分 ← 現 `_variational_fd_love` | — |
| `ad` | 伝播子を AD で貫き IFT ← 現 `full_ad` | 1階なら1／2階なら2（nested） |
| `fd` | c 自体の有限差分 ← 現 `_fd` | — |

前半が「変分か否か」、後半が「AD段数」で、`sensitivity_implementation_map.md` が挙げる
2軸そのもの。**規則: method は quantity が kernels/gradient/hessian/interface_* の
ときは省略不可。** 無印は禁止（`compute_phase_velocity_hessian` が該当）。
dispersion/group_velocity/hv_ratio/eigenfunctions/energy_integrals/mode_count は
微分を取らないので method 軸自体が無い。

退役: `weakform` / `hybrid` / `full_ad` / `stable` / `reference`。`hybrid` は
2軸のどちらにも収まらない過渡実装で、退役後の扱いは §7 の該当行で個別に判断する
（後述、ユーザー確認が必要）。

### wave

`rayleigh` | `love`。**省略は「引数で分岐する」を意味する**（現状3通りの分岐レベル
—— 名前に焼き込む／`WaveType` 引数／`wave_type_str` 引数 —— が無規則に混在している
点を規則化する）。

### model

`layer_cake` | `grid`。**両方明示**する。ただし model の区別自体が存在しない
関数（`kernels`/`hessian` 系以外の大半のユーティリティ）には付けない。
`layer_cake` は一つのモデル種別を表す語として扱う。`grid` の省略で暗黙に表さない。

### variant

`_reduced`（基底縮約）/ `_extrapolated`（レベル外挿）。
**規則: 同名の非外挿版が併存するときだけ `_extrapolated` を付ける。**
現状 `*_Grid` 系の Hessian も内部で Neville 外挿しているのに付いていない
不整合があるが、これらには非外挿の対になる名前が存在しないため、本規則の
適用後もサフィックスを付けない（§7 の当該行に注記する）。逆に
`compute_interface_hessian_row_variational`（非外挿）と
`..._extrapolated`（外挿）は対で存在するため、外挿側だけがサフィックスを持つ。

## 4. 衝突語の裁定（1トークン1意味）

| 語 | 裁定 |
|---|---|
| `Reduced` | 基底縮約のみ。複合行列の5成分表現 `ReducedIndex`（`disper26.cpp:171`）は `MinorIndex` へ |
| `Romberg` | レベル外挿の意味に限定。層内求積（`integrateGridLayer_Romberg`、`propagate_romberg_vector`）は `quadrature` へ |
| `Liquid` / `Fluid` | `fluid` に統一 |
| `Boundary` / `Interface` / `Depth`（K_d） | `interface` に統一 |
| `Bran` / `AdModeCount` / `Total` / `Continuous` | `mode_count`（正）と `continuous_mode_count`（legacy）の2語のみ |
| `Pieces` / `Breakdown` / `Trace` / `Probe` | `debug_` 接頭辞に統一 |
| `Weakform` / `WeakForm` 表記揺れ | 語自体の退役で消滅 |
| `compute` / `calculate` | `compute` |

## 5. kwarg・辞書キー・フィールド名

- **「値」と「カーネル」の区別を名前に出す**: `want_value_u` / `want_value_hv`（値、
  現 `phase_only` の分解）と `want_kernel_u` / `want_kernel_hv`（カーネル、現
  binding kwarg `compute_u`/`compute_hv`、`inversion/` 側 `want_u_kernel`/`want_hv_kernel`
  の統合）。
- `sensitivity_mode`（`bindings.cpp:3365`）/ `mode`（3283）/ `mode_str`（3484）/
  `parameter_mode`（3251）の4綴り → `sensitivity_mode` に統一。`wave_type_str`（3484）
  → `wave_type`（3283 に合わせる）。
- **同一量 ∂c/∂p の4名**: `d_c`（LAYER_CAKE 層別分布、`SensitivityKernel`）/
  `gradient` / `gradient_node`（組み上げたベクトル、`HessianResult` /
  `GridHessianReducedResult`）と役割で分け、出所は method 語彙の接尾辞で示す。
  Python 側の `g_a_kernel` / `g_a_hessian`（`inversion/disper26_inv/parallel.py:94,116`）
  → `g_a_ad` / `g_a_variational`。同じ理由で `g_wf_depth`（parallel.py:110）→
  `g_variational_depth`。
- **Python `column` vs C++ `row`**（転置で同じ物）→ `interface_hessian_row` に統一し、
  `depth_hessian.py` の `VariationalDepthHessianBackend.column()` を
  `interface_hessian_row()` に改称する。
- Romberg trace は `phase_velocity_roots`（level ごとの生 root）と
  `phase_accepted_level`（direct phase の外挿 prefix 終端）を必ず組で渡す。
  追加生成深さは `minimum_trace_level`、root solve tolerance は `root_tolerance` とし、
  phase Romberg の `tolerance` と名前を分ける。
- **depth backend の三重命名**を1系統に統一する。

  | 現行 registry key | 現行クラス名 | 現行 `method` 文字列 | 正 registry key = method 文字列 |
  |---|---|---|---|
  | `finite_difference` | `FiniteDifferenceDepthHessianBackend` | `central_fd_of_ad_gradient` | `fd` |
  | `nested_ad` | `AutoDiffDepthHessianBackend` | `nested_ad` | `ad` |
  | `variational` | `VariationalDepthHessianBackend` | `variational_extrapolated` | `variational_ad`（K_d 自体は1階の閉形式=AD0だが、Hessian 列はそれをもう1階ADで微分するため） |

  クラス名は `<Key>DepthHessianBackend` の形（`FdDepthHessianBackend` /
  `AdDepthHessianBackend` / `VariationalAdDepthHessianBackend`）に揃える。
- `backend` 既定が層で違う（binding 既定 `"dual"` / `ForwardEngine` 既定 `"vector"`）
  → 規約として「既定は1箇所（呼び出し側の最上位、ここでは `ForwardEngine`）で
  定義し、C++ 側は明示指定を要求する」ことを明記する。C++ 側のデフォルト引数は
  後方互換のため残してよいが、新規呼び出しは明示する。
- **既知の陳腐化バグ**: `inversion/disper26_inv/pipeline.py:56-61` は Phase-1
  キャッシュキーに `"depth_backend": "central_fd_of_ad_gradient"` を焼き込んでいるが、
  既定バックエンドは `depth_hessian.py:513` の `DEFAULT_DEPTH_HESSIAN_BACKEND = "variational"`
  へ 2026-08-11 以前に切り替わっている。文字列自体は「Phase-1 キャッシュ契約を
  変えたら別ハッシュにする」ためのマーカーなので実害は当面ないが、
  文字列が指す内容と実装が一致していない点は正名採用時に `"variational_ad"`
  （新 method 語彙）へ直すべき、命名の別名化が誘発した典型例として記録する。

## 6. 旧名の扱い

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

`bindings.cpp:3220-3221, 3238-3240, 3248` の先例どおり：

- `m.def` を正名・旧名の2本登録し、旧名の docstring 冒頭に
  `Deprecated alias. Use <正名>.` を付ける。
- C++ コアは `inline` 転送関数として旧シグネチャを保つ。
- **新規コードと文書は正名のみ**を使う。
- `AGENTS.md` の「README 記載名を保存する」という既存の縛りは、README 記載名が
  「呼び出し可能なまま残る」ことで満たされる（§B 参照）。README 自体は正名を
  使うように書き直す。
