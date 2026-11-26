# 共通公開API

[English edition](../docs/public_api.md)

Python では、モデル作成から分散・感度・Hessian の計算まで、高水準 API（`disper26`）だけで完結する。
球面の弾性定数補間 GRID で地球平坦化変換（EFT）を含む界面微分を求める場合も同じである。
高水準 API の例では `import disper26 as d26` を使う。[練習例](../example_jp/README.md)を参照。

細分化レベルごとの根を自分で渡す場合や、計算法ごとの結果を直接比較する場合は、
低水準 API（`disper26py`）を使う。低水準 API の例での表記は `import disper26py as d26_lowlevel` である。
配布パッケージ `disper26py` をインストールすると、両方の API が使える。詳しくは[低水準 API 利用ガイド](lowlevel_api.md)を参照。

高水準 API は C++ の `disper26::api`、C の `api_c.h`、Fortran の `disper26_api` と同じ C++ 実装を使う。
従来の Fortran `disper26` も互換 API として引き続き利用できる。

## 海底重力オプションの削除

海底で法線応力に未検証の跳びを加える `gravity_seabed_scale` は削除した。
低水準 Python の `create_fluid_parameters` の `seabed_scale` 引数も削除したため、
旧呼び出しでは該当キーワードまたは第6位置引数を除く。
`Material` の `gravity_seabed_scale` キーワード・第10位置引数、物性への同名フィールド設定も除く。
海底では鉛直変位 $w$ と法線応力のラグランジュ摂動 $T_{zz}$ がそれぞれ連続する。
旧値がゼロの計算の数値動作は維持し、非ゼロ値の追加ジャンプは提供しない。

C++・C・Fortran の物性型は9成分となるため、利用側も再コンパイルが必要。
C API の `model_info` の `materials` は列優先の9行配列、`halfspace` は9要素で、
順序は `rho, A, C, F, L, N, gravity, gravity_internal_scale, gravity_surface_scale`。
`gravity`、`gravity_internal_scale`、`gravity_surface_scale` の仕様は変更しない。

## モデル → 解 → 必要な計算

```python
import disper26 as d26

model = d26.create_model(
    [0.0, 10.0], kind="layer_cake",
    vs=[3.0, 4.5], vp=[5.2, 8.0], rho=[2.5, 3.3],
)
solution = d26.compute_dispersion(
    model, period=5.0, wave="Rayleigh", mode=0,
    outputs=("phase_velocity", "group_velocity", "hv_ratio"),
)
kernels = d26.compute_kernels(solution, basis="velocity_iso")
hessian = d26.compute_hessian(solution, basis="velocity_iso")
print(solution.phase_velocity, solution.group_velocity)
print([(p.entity, p.index, p.name, p.active) for p in kernels.parameters])
print(kernels.gradient, hessian.hessian.shape)
```

単位は km、s、g/cm³、GPa。入力周波数は通常APIでは周期 `period` に統一し、
基本モードは全言語で `mode=0` とする。周波数曲率の微分変数だけはフィールド名に
記載した角周波数 ω。重力加速度は km/s²。

モデルは入力をコピーする不変スナップショット。解はモデル、波種、周期、モード、
計算オプション、採用した根を保持する。後続の微分に `c_guess` や生の根配列を
渡し直す必要はない。プロパティ参照は計算を起動しない。

```python
grid = d26.create_model(
    [0.0, 10.0, 30.0], kind="grid", interpolation="stiffness",
    vs=[3.0, 3.4, 4.5], vp=[5.2, 6.0, 8.0], rho=[2.5, 2.8, 3.3],
)
curve = d26.compute_dispersion_curve(grid, periods=[8.0, 5.0, 8.0])
assert curve.phase_velocity[0] == curve.phase_velocity[2]
print(curve.periods)  # 入力の順番と重複を保持する
```

GRIDは `interpolation="stiffness"` または `"velocity"` が必須。同じノード値でも
補間する物理量が違えば異なるモデルであり、Hessianの都合で自動変換しない。
重複する二つの深度は不連続面の上下を表す。最深ノードの物性値が、その深さ以深の半無限媒質を定める。
補間・逆解析に使う深さの選び方は[弾性定数補間 GRID の深さ座標](#grid-depth-coordinates)を参照。

曲線は周期を内部で整列・重複除去して計算し、入力順に戻す。このGRIDを既定の細分化上限で
計算すると、例では `ConvergenceWarning` が出ることがある。各解の `diagnostics` で採用レベルと
推定誤差を確認できる。`mode` は各周期の
位相速度順の番号であり、固有関数によるモード同一性の追跡ではない。

### 数値計算のオプション

| オプション | 既定値 | 用途 |
|---|---|---|
| `max_steps` / `tolerance` | 5 / 1e-10 | GRID の分散・曲線・周波数曲率の細分化 |
| `max_levels` / `derivative_tolerance` | 5 / 1e-6 | GRID 微分の細分化と、対応する演算での外挿 |
| `root_tolerance` | 1e-12 | 個々のメッシュでの根探索 |
| `num_sub_layers` | 10 | この設定を取る演算の層内分割数 |
| `backend` | `"dual"` | 変分 AD Hessian では `"vector"`、`"vector_dyn"` も選べる |
| `integration` | `"auto"` | 自動・解析・台形積分 |

Layer-cake Love 波の Hessian で `integration="trapezoidal"` を指定すると
`num_sub_layers` を使う。以前の台形積分結果を再現するには 200 を明示する。

### モード選択と追跡

分散計算の `Options.method` は次の方式を選べる。既定は `standard` のまま。

| method | mode count の使用 |
|---|---|
| `standard` / `mode_order` | 各周期・各使用レベルで番号を確認 |
| `continuation_recover` | 各レベルの初期探索と、局所追跡失敗時の復帰探索で番号を確認 |
| `continuation` | 初期探索・復帰も含め内部 count 0。近傍根を追跡 |

`continuation` では `c_guess` を目的の根の近くに指定する。`mode` は未確認のラベル。
`continuation_recover` は初期・復帰時に指定 `mode` を選ぶが、中間点の番号保証はない。
定期的な確認や最終点の自動確認は行わず、初期・復帰探索ではカウンタを複数回使う場合がある。
両追跡方式の `verify_mode=True` は全点確認に切り替える。
`None` / `False` は表の方式を使う（recoverの初期・復帰確認は無効化しない）。
`standard` と `mode_order` は `verify_mode=False` を拒否する。`verify="every"` などの方針を指定するときは `verify_mode` を省略する。`level_mode_verified` は実際に
確認した根だけtrueとなる。端点の確認だけでは、途中で別のモードブランチ（mode branch）に移ったことを見落とす場合がある。

`predictor="history"` は同じレベルの直近最大3点とレベル間情報を使う。
`"auto"` は既に求めた同じレベルの群速度を再利用する。群速度を出力しない場合も、
周期方向の予測に使うレベルの初回に群速度を計算して次点を予測する。
layer_cakeは1回のADを追加する。毎周期確認するGRIDはlevel 0のみ1回、
それ以外のGRID追跡はlevel 0・1だけ最大2回とする。
次の異なる周期がない場合、周波数差25%超、単点計算では追加しない。
群速度も出力する場合には、その計算を再利用する。
確認なしautoは予測誤差に応じた局所区間を使い、レベル2以降では従来GRID曲線と
同じレベル間の適応探索も候補にする。level間の変化が既存の直接Brent条件を
満たす場合はその探索を優先し、それ以外は推定探索幅が狭い方を選ぶ。
根と一緒に計算済み残差を受け取り、同じ残差判定を重複計算せずに行う。
必要な地点ではPython/C++の `compute_mode_count(solution, level=-1)` を明示的に呼ぶ。
1呼出しにつき整数カウンタを1回実行し、再探索や他レベルの確認は行わない。
GRIDの `level` は保存済みレベル番号、`-1` は最後に生成したレベルを選ぶ。
そのレベルの外挿前の根と対応するmeshを使用する。layer_cakeでは `0` または `-1`。
Cは `d26_api_compute_mode_count`、Fortranは `compute_mode_count` を提供する。
refinement levelは全言語で0始まり。
戻り値は0始まりの整数mode番号。要求番号と異なっても再探索や自動修正は行わない。
解の `mode`、診断の `mode_verified` / `level_mode_verified` は変更しない。
一地点の確認は他の周期・レベルの番号保証ではない。保存されていないレベルはエラー。

```python
periods = [5.0, 5.2, 5.4]
curve = d26.compute_dispersion_curve(
    grid, periods=periods,
    options=d26.Options(method="continuation", verify_mode=False, c_guess=3.0))
last_mode = d26.compute_mode_count(curve.solutions[-1])
first_coarse_mode = d26.compute_mode_count(curve.solutions[0], level=0)
```

曲線は既定で最長周期から最短周期へ内部計算するため、`c_guess` は最長周期の初期推定となる。
従来の計算方向を指定するには `Options(start="short")` を使う。
返却順は入力順なので、上の `[-1]` は入力の最後の地点を意味する。この例の二つのカウンタはともに `0` を返す。
kernel/Hessianなど微分計算が要求する保存根の番号確認は従来通り別途行う。
確認付きautoのレベル1以降では、同一周期の粗い根を使う。群速度予測を利用し得るのは
レベル0であり、cのみを要求したautoでも初回に群速度を計算する。群速度による一次予測は
`dc/dω=(c/ω)(1-c/U)` を使い、前の角周波数からの変化が25%以内の場合に候補とする。
条件に応じて履歴や粗いレベルの情報へ切り替える。
`"group_velocity"` はAD評価を許可し、出力にUを要求した場合はその評価を共用する。
未要求の公開群速度は引き続きNone。局所追跡に失敗した場合、`continuation` は
予測速度の±20%内の根を探索して最も近いものを選ぶ。既定の追跡設定で根がなければ `root_not_found` を返す。
`continuation_recover` と全点確認の方式はモード順探索へ戻り、失敗したら例外を返す。

全方式で `solution.mode` は入力した要求番号であり、検出した実際の番号へ自動更新しない。
特に `continuation_recover` の通常の追跡点でも、返す番号は未確認のラベルとなる。
`diagnostics.mode_verified` はその解の全使用レベルを確認した場合だけtrueであり、
`level_mode_verified` はレベルごとの状態。復帰が一部レベルだけなら全体は未確認。
これらは数値収束と独立した確認状態。
`strict=True` はモード確認を追加する指定ではない。`predictors`、`fallback_reasons`、
`fallback_count`、`mode_count_evaluations` で経路を確認できる。
`predictor_ad_evaluations` はautoの開始時および明示的な群速度予測のAD評価数
（予測用評価を出力Uと共用する分を含む）であり、群速度出力の全AD評価数ではない。
未確認解から固有関数・エネルギー積分・重力補正・感度・Hessian・周波数曲率を求める前には保存済みの各レベル根を再確認する。
不一致の場合、PythonとC++ではコード `mode_mismatch` の例外が発生する。
成功した確認は内部で共有し、公開の元の解・診断は変更せず、派生結果に確認状態を返す。

これらの指定はPython/C++の型付き入口とFortranの `computation_options` で利用できる。
Cは既存の `d26_api_options` を維持し、追加の `d26_api_tracking_options` を
`d26_api_default_tracking_options` で初期化して `d26_api_compute_dispersion_ex` /
`d26_api_compute_dispersion_curve_ex` に渡す。拡張入口のcontextはNULLなら新規作業領域を使う。
`verify` / `on_violation` の空文字列と `verify_mode=-1` は未指定を表す。
低水準 `disper26py` の既存バッチの引数・既定方針は維持する。
共通APIと低水準バッチは数値カーネルを共有するが、探索制御は同じではない。
共通曲線は1回の呼び出し内でメッシュ・追跡状態を共有する。呼び出し間の作業領域再利用は
後述の `DispersionContext` を使う。

### 開始方向・初期速度範囲・確認頻度

| Options | 指定内容 | 省略時 |
|---|---|---|
| `start` | `"long"`: 最長周期から、`"short"`: 最短周期から | `"long"` |
| `c_range` | 開始点の位相速度範囲 `(lo, hi)` [km/s] | なし |
| `verify` | `"none"`: 通常の番号確認を省略、`"endpoints"`: 有効区間の開始・終了、`"every"`: 毎周期 | 上記のmethod別の動作 |
| `on_violation` | `"fallback"`: 指定番号を再探索、`"error"`: 例外、`"missing"`: 欠損値 | 新しい確認・範囲設定では `"fallback"` |

```python
options = d26.Options(
    start="long", c_range=(2.8, 4.2),
    verify="endpoints", on_violation="fallback")
curve = d26.compute_dispersion_curve(model, periods=[2., 5., 10., 20.], options=options)
```

この例は20→10→5→2秒で計算し、結果は指定した2・5・10・20秒の順に返す。
重複周期は一度だけ計算する。`c_range` は最初に解を得られる周期の各生成GRIDレベルに
適用し、その開始点では `c_guess` を使わない。高次モードが最長の要求周期に存在しない場合、
欠損周期を飛ばした最初の有効周期に範囲を適用する。
範囲の両端は有限・正で `lo < hi` が必要。探索範囲の上端は、半無限媒質のS波速度を用いて
Rayleigh波では $0.9995\,V_{SV}$、Love波では $0.9995\,V_{SH}$ に制限する。切り詰めた範囲が空なら入力エラーとする。
初期範囲のモードカウントから指定したモードが1つだけ含まれると確認でき、区間の両端で分散関数が有限な値をとって根を挟む場合は、
その区間で直接Brent法を使う。接する根や残差の問題には従来の狭い区間での探索を使う。

`verify` を明示するとmethod別の確認頻度を置き換える。`verify_mode` との同時指定は拒否する。
`verify="none"` の走査は、走査の分解能で検出できた根が1つであることだけを調べ、
指定モードの番号や近接根の不在を保証しない。異常時に `on_violation="fallback"` を使うと、
その復旧探索ではモード数を計算し、要求番号の根を選ぶ。
新しい確認・範囲・復旧設定を省略した既存の `continuation` は、内部count 0の動作を保つ。

`endpoints` は各有効区間の最初と最後の、外挿前の各レベル根を確認する。
終点で番号不一致を検出した自動復旧は、直前の確認済み地点から終点までを毎周期確認して
計算し直す。`missing` は同区間を欠損にする。途中で別モードへ移って元へ戻るケースは
両端確認では排除できないため、全周期の番号保証が必要な逆解析では `every` を指定する。

探索上限でのカウントから要求番号の不存在が分かった周期は、既定では `NaN` を返す。
`diagnostics.status="missing"`、`missing_reason="mode_not_found"` を記録し、他周期を続行する。
追跡異常を欠損にする指定では理由が `"tracking_violation"` となる。
要求した群速度・H/Vも欠損値となり、未要求量やLoveのH/Vは従来どおりNoneとなる。
欠損を予測履歴へ入れず、欠損区間を越える追跡は初期化し直す。
`strict=True` はこれらをエラーにする。入力不正や未対応条件は常にエラーとなる。
モード数や派生量を欠損解から求めると `missing_solution` となる。
感度や固有関数、エネルギー積分、重力補正などの派生計算では、
保存した根のモード番号が要求と異なる場合は `mode_mismatch` となる。

### モデル配列

GRID の正のノード間隔は 1e-12 km 以上とする。深さの等しい2ノードは不連続面を表す。

`create_model` は両モデルで深度と物性の配列を受け取る。`kind` は必須で自動判定しない。
LayerCakeの深度は層の上端で、物性は次の深度まで一定、最終点から下は半無限媒質となる。
深度は0から厳密に増加し、重複を許さない。厚さは深度差から生成する。
GRIDは0からの非減少深度列で、不連続面の重複は2点まで。最小点数はLayerCakeが1点
（均質半無限媒質）、GRIDが2点。LayerCakeに `interpolation` を指定するとエラーになる。
1点のLayerCakeでも `compute_dispersion(..., wave="Rayleigh", mode=0)` で
均質半無限媒質の基本Rayleighモードを計算できる。

物性の指定は `vs/vp/rho`、`vph/vpv/vsh/vsv/eta/rho`、`A/C/F/L/N/rho` の
いずれか完全な一組。全配列は深度と同じ長さの有限値の1次元配列とし、自動ソートや
スカラー展開はしない。等方性の流体は `vs=0`。物性の入力方式とGRIDの補間方式は独立。
入力から既存の不変モデルへ変換するため、モデル情報や微分軸の契約は変わらない。
LayerCakeのモデル情報は引き続き `thicknesses` と別の `halfspace` を保持する。
従来のオブジェクト入力は `create_layer_cake_model` / `create_grid_model` で利用できる。

VTI固体では、`vpv`（$V_{PV}$）と `vph`（$V_{PH}$）はそれぞれ鉛直方向と水平方向に伝播するP波の速度である。
水平に伝播するS波のうち、鉛直偏波の速度は `vsv`（$V_{SV}=\sqrt{c_{44}/\rho}$）、
水平偏波の速度は `vsh`（$V_{SH}=\sqrt{c_{66}/\rho}$）である。

正式なモデル種別は `layer_cake`。旧 `kind="pancake"` と `create_pancake_model`
も引数・既定値を維持した legacy 入口として使える。新名で作ったモデルの `kind` は
`"layer_cake"`、旧名で作ったモデルは `"pancake"` を返す。この表記は解、平坦化、
Q 補正後も保持され、C++・C・Fortran でも同じ規則を使う。計算処理は共通である。


## 結果と診断の契約

| 項目 | 契約 |
|---|---|
| 出力量 | `outputs` で指定。位相速度は解の必須量。既定でU/HVは計算しない |
| 未要求・適用外 | Python/C++では `None` / `std::optional`。LoveのHVは適用外 |
| 数値ゼロ | 計算されたゼロと、未計算・固定を区別する |
| 勾配 | `gradients["phase_velocity"]` 等。Pythonの `gradient` は位相速度勾配 |
| パラメータ軸 | `parameters` に `name, entity, index, unit, active` を保持 |
| 固定パラメータ | GRIDの流体ノードは `active=False`、微分はゼロ |
| Hessian | 位相速度の2階微分。Pythonは2次元配列、C++/Cは列優先配列と形状 |
| 診断 | `diagnostics` に計算法・収束状態・採用レベル・誤差推定・出力量の状態 |
| 所有権 | Python配列はコピー。Cハンドルは参照カウント、Fortranは代入・終了処理で管理 |

既定では外挿が指定精度に届かなくても結果と診断を返す。Pythonは
`ConvergenceWarning` を発行する。`Options(strict=True)` は要求精度を確認できない
結果を例外にする。既存バックエンドが独立の誤差推定を返さないHessianなどは
`convergence="not_evaluated"` とし、収束済みとは表示しない。この場合もstrictでは失敗する。

分散の `relative_error` は位相速度の外挿誤差であり、U/HVの誤差推定ではない。
`convergence_by_quantity` で量別の状態を確認できる。U/HVを要求した場合は、
位相速度が収束していても全要求量の精度は `not_evaluated` となり、strictは失敗する。

派生計算のOptionsは各呼び出しの設定であり、解のOptionsを暗黙に継承しない。
根の追跡には解に保存したモード・root toleranceを使い、必要な追加レベルのみ生成する。

### 符号付きRayleigh楕円率

`hv_ratio` は符号付きの比 ε = y3/y1 で、`d_HV` と `hv_ratio` の勾配はその微分である。
従来の振幅比から意味が変わった。C・Fortranと旧名を含め、同じ規約を使う。
zを上向き、波の位相を exp(i(kx−ωt)) とすると、固有関数配列 `uz` と `ur` は
それぞれ y1 と y3 を格納し、水平変位の複素振幅は i·y3 である。
波が正の x 方向に伝播するとき、ε > 0 は媒質粒子の楕円運動が prograde、ε < 0 は retrograde であることを示す。
粒子が楕円の最上点を通る瞬間の水平速度は、prograde では波の進行方向、retrograde ではその反対を向く。

Layer-cakeとGRIDでは、固体自由表面、または海底の固体側で評価する。
せん断応力ゼロの条件から、$Y_{14}\ne 0$ のとき、固体側の小行列式を用いて $\epsilon=-Y_{12}/Y_{14}$ と表せる。
従来の海底振幅には自由表面の法線応力条件が使われていた。
GRIDでは値と勾配を符号付きで外挿する。一定物性の区間を一様に摂動した場合の層感度は、
その区間の両端ノードの感度を足して比較できる。

振幅比 E = |ε| の微分は、ε ≠ 0 では sign(ε)·dε/dp である。
水平変位がゼロを横切るときは |ε + 勾配 · Δp| から振幅を予測できる。
滑らかなゼロ交差では符号付き比を微分できるが、振幅の微分と正規化感度
(p/ε)·dε/dp は特異になる。鉛直変位がゼロの点は比の極で、値と感度は未定義である。共通APIで要求した分散値が非有限なら `numerical_failure` になる。

Love 波の H/V は未定義であり、要求していない値には既存の出力規約が適用される。

## 微分の座標系と対応範囲

`basis` は明示指定する。モデルの補間方式とは独立である。

| basis | 各層・ノードの軸順 |
|---|---|
| `stiffness_vti` | A, C, F, L, N, Rho |
| `stiffness_iso` | Lambda, Mu, Rho |
| `velocity_iso` | Vs, Vp, Rho（等方モデルに限定） |
| `velocity_vti` | rho, vph, vpv, vsh, vsv, eta |
| `log_velocity_vti` | 上記6量の自然対数。微分対象のパラメータ（`active=True`）は対数を取る前の値が正であること。 |

両方の `_iso` 基底は等方な基準媒質に対して等方性を保つ摂動を扱う。
`stiffness_iso` の独立な弾性定数はラメ定数 $\lambda$ と $\mu$（GPa）である。密度微分では $\lambda$ と $\mu$ を固定し、
`velocity_iso` の密度微分では両波速度を固定する。同じ等方解に対し、
`compute_kernels(solution, basis="velocity_iso")` は Vs と Vp を、
`basis="velocity_vti"` は $V_{PH}$、$V_{PV}$、$V_{SH}$、$V_{SV}$ を独立に摂動する。
`isotropic` と `stiffness` もそれぞれ `velocity_iso` と `stiffness_vti` の別名として使える。
結果の基底名には指定時の表記を返す。

A = 2L で VTI 速度座標を選ぶには、eta を含む VTI 速度配列を与える。
等方速度配列では eta = 1 である。この点では弾性定数だけから eta を復元できない。
速度補間 GRID の個々の VTI 弾性定数に対する感度には、速度への逆変換が一意に定まることが必要である。

半無限媒質は `entity="halfspace"` で区別する。速度・対数座標のHessianには
`Jᵀ H J` に加え、物性写像の2階微分と勾配の項も含める。
`compute_hessian_reduced(solution, projection)` は縮約パラメータの摂動を弾性定数・密度の摂動に変換する**線形**写像を受け取る。
非線形写像の曲率項はこの関数の対象外。フルHessianを生成する代替処理は行わない。

| 操作 | 対応 |
|---|---|
| 分散、曲線、カーネル | Layer-cake、弾性定数補間 GRID、速度補間 GRID |
| 物性のフルHessian | Layer-cake、弾性定数補間 GRID |
| 物性の縮約Hessian | 弾性定数補間 GRID |
| `compute_interface_kernels` | 全層が固体の flat Layer-cake または全層が固体の弾性定数補間 GRID（flat・球面） |
| `compute_interface_hessian_row` | 全層が固体の弾性定数補間 GRID（flat・球面） |
| `compute_eigenfunctions`、`compute_energy_integrals`、`compute_frequency_curvature` | 解を保持する各モデル。GRIDの波動場と積分は採用メッシュ上の量 |
| `compute_gravity_correction` | 非Cowling、flat LayerCakeのRayleigh。`region="fluid"/"solid"/"all"` |

弾性定数補間 GRID で界面を動かすときは、同じ深さにある上下のノードを一緒に動かし、各ノードの**入力座標系での物性値**を固定する。
`compute_interface_kernels` は各界面の深さに対する位相速度勾配を返す。`compute_interface_hessian_row` の各列は、指定した基底のノード物性、続いて全界面の深さに対応する。
既定手法は flat では勾配が `variational`、Hessian 行が `variational_ad`、球面ではどちらも `ad` である。
順計算・微分・結果表示で使う深さの選び方は[弾性定数補間 GRID の深さ座標](#grid-depth-coordinates)を参照。

Rayleighの固体半無限媒質の上に表面流体層が連続する場合、`compute_energy_integrals` の `integration="auto"` は解析積分を選ぶ。低水準の点列版 `compute_energy_integrals_analytical_rayleigh` は固体専用である。

GRIDの周波数曲率は固有のRomberg停止条件を使うため、戻り値の位相速度が元の
解と異なる場合がある。固有関数・積分の採用メッシュ上の値とは区別する。

流体を含むLoveのモード探索は対象外。実際のモデルについて
`get_capabilities(model, wave=...)` が対応可否と理由を返す。未対応の手法は `unsupported`、基底の不整合や変換の二重適用は `invalid_argument` となる。CとFortranはステータスコードで通知する。
共通 API と低水準 API の物性・界面微分には重力を無視した基準モデル（$g=0$）を使う。
流体 Cowling の順計算は、別途求める一次重力補正の検算に使える。

標準経路は `Options(method="standard")`。明示的な `ad`、`variational`、
`variational_ad` の可否は演算ごとに異なる。GRIDの固定レベルFull ADなどの
検算専用機能は従来の低水準APIで利用する。

## 変換とPREM

<a id="grid-depth-coordinates"></a>

### 弾性定数補間 GRID の深さ座標を選ぶ

**どの深さで弾性定数と密度を区分線形と仮定し、どの深さで界面を動かすか**で選ぶ。$d$ は地表から測った球面深さ、$z$ は平坦化深さで、どちらも km、下向き正とする。$R$ は地球（または対象天体）の半径である。

| 目的 | モデルの渡し方 | 順計算・逆解析で使う座標 |
|---|---|---|
| Flat で閉じる | Flat のノードをそのまま渡す。`Geometry("flat")` が既定 | 弾性定数と密度は $z$ で区分線形。界面勾配は $\partial c/\partial z$ |
| 球面で閉じる（内部で EFT） | 球面ノードを `Geometry("spherical_ba97")` または `Geometry("spherical_experimental_density", flattening_exponent=...)` で渡す | 弾性定数と密度は $d$ で区分線形。MCMC の提案、界面勾配 $\partial c/\partial d$、Hessian を球面のモデル変数で扱う |
| Flat で計算し、入出力を球面で扱う | 球面の参照ノードを最初に一度だけ `earth_flattening_transformation` で変換し、返された Flat モデルを使う | 1 行目と同じく、提案・勾配・Hessian は Flat 座標。推定したノードや各事後サンプルを、表示時に球面座標へ戻す |

たとえば「球面でのノード物性を固定し、不連続面を実深さで 1 km 下げたときの位相速度」を求めるなら、球面モデルをそのまま渡す。
ソルバーは各ノード間を $z$ で等分し、小層の中点を $d$ に戻す。その深さで弾性定数と密度を補間し、その半径で EFT を適用する。半無限媒質は上端の半径を使う。

界面を動かすと小層厚、補間位置、物性の変換係数が変わる。界面の勾配と Hessian 行を要求すると、各細分化レベルでこれらの変化と分散方程式の根を自動微分し、外挿する。
サロゲートを使わず MCMC で位相速度だけを求める場合は、各提案で順計算の EFT を行う。界面微分の計算は生じない。

同じ球面ノードから、球面深さと平坦化深さに対する勾配をそれぞれ求める例を示す（重力を無視した全層固体モデル）。

```python
import disper26 as d26

spherical = d26.create_grid_model(
    [0., 18., 18., 55.], vs=[3., 3.55, 4.04, 4.55],
    vp=[5.4, 6.3, 7.2, 8.22], rho=[2.55, 2.75, 3.05, 3.35],
    interpolation="stiffness", geometry=d26.Geometry("spherical_ba97"),
)
flat = d26.earth_flattening_transformation(spherical)
for model in (spherical, flat):
    solution = d26.compute_dispersion(model, period=26., options=d26.Options(max_steps=8))
    gradient = d26.compute_interface_kernels(solution)
    row = d26.compute_interface_hessian_row(solution, interface_index=0, basis="stiffness_vti")
    print(model.geometry.kind, solution.phase_velocity, gradient.gradient[0], row.hessian[0, -1])
```

出力の最終列は、それぞれ $\partial^2 c/\partial d^2$ と $\partial^2 c/\partial z^2$ である。2 列目の位相速度は、球面モデルで 3.865760 km/s、先に平坦化したモデルで 3.865075 km/s となる。後者は変換済みのノード間を $z$ で線形補間する。$z=R\log(R/(R-d))$ は $d$ に対して非線形で、物性の変換係数も半径に依存するため、両モデルでは一般にノード間の物性分布が異なる。

**結果表示だけなら**、推定深さを $d=R(1-e^{-z/R})$ で戻す。ノード物性も表示する場合は、その推定半径 $R-d$ で元と同じ EFT の係数を使い、物性の変換を逆にたどる。位相速度 $c$ は順計算の値をそのまま使う。
`earth_flattening_transformation` は球面から Flat への変換で、返されたモデルには変換履歴が残り、二重変換を拒否する。

**Flat モデルの Taylor 展開を球面変数で書き直す場合**は、深さと物性を合わせたノード変換 $y=T(x)$（球面 $x$、Flat $y$）が必要になる。
$J=\partial y/\partial x$ とすると、$g_x=J^\top g_y$、$H_x=J^\top H_yJ+\sum_i (g_y)_i\nabla_x^2 T_i$ である。
この連鎖律で得られるのは、先に平坦化したモデルの微分である。球面深さで線形補間するモデルのサロゲートには、球面モデルから直接得た勾配と Hessian を使う。

### Layer-cake・減衰補正・PREM

球面 Layer-cake は順計算前に明示的に平坦化する。均質な半無限媒質の物性は上端深度で平坦化する。

`apply_mineos_physical_dispersion` は半無限媒質を含むQ配列を受け取り、補正周期と
基準周期を保存する。LayerCakeでは層数+1、GRIDではノード数。補正済みモデルの
再補正・別周期での利用を拒否する。周期列を計算するときは元モデルから各周期用の
モデルを作る。

同梱 PREM を使う場合、`load_prem_spherical` は内部で EFT を行うための球面ノードを、`load_prem_flat` は一度平坦化した Flat 計算用のノードを返す。
いずれも Python 専用で、上部地殻を地表まで延長する。キーワード専用引数 `max_depth_km=500.0`、
`max_spacing_km=10.0`、`density_exponent=-2.275` で深さ・間隔・密度指数を指定する。同梱PREMでは深さ2,891 kmまで指定でき、
境界上ではマントル側の物性を半無限媒質に使う。それより深い指定では `ValueError` が発生する。

補正量は `delta_c = c(g) − c(0)`、`delta_u = U(g) − U(0)` で、観測値から引く。既定は `gravity=9.80665e-3` km/s²、`region="solid"`。既定は層内40分割、角周波数の相対差分幅
0.005で、これらの値を変える場合は対象モデルで収束を確認する。結果は元の解を
変更せず、補正前の速度と `delta_c` / `delta_u` を別フィールドで返す。

## C++・C・Fortran

C++ は `#include "api.hpp"` を使う。`add_subdirectory` で取り込むと CMake ターゲット `disper26::api` をリンクできる。
APIの公開ヘッダーにEigen・Python依存はない。従来の実装ライブラリ
`disper26_lib` と同時にリンクする必要はない。

```cpp
#include "api.hpp"
int main() {
    namespace d26 = disper26::api;
    d26::MaterialArrays parameters;
    parameters.vs = std::vector<double>{3.0, 4.5};
    parameters.vp = std::vector<double>{5.2, 8.0};
    parameters.rho = std::vector<double>{2.5, 3.3};
    auto model = d26::create_model({0.0, 10.0}, "layer_cake", parameters);
    auto solution = d26::compute_dispersion(model, 5.0);
    auto kernels = d26::compute_kernels(solution, "velocity_iso");
    return kernels.parameters.empty();
}
```

Cは `api_c.h` の `d26_api_*` を使用する。`d26_api_create_model` は深度列と
「成分×点」の列優先物性行列を受け取る。指定できる `parameterization` は次の3通りで、成分順は
`isotropic`: Vs,Vp,Rho、`velocity_vti`: rho,vph,vpv,vsh,vsv,eta、
`stiffness`: A,C,F,L,N,Rho。LayerCakeの補間引数はNULLとする。結果フィールドを列挙し、必要サイズを
照会してからバッファへコピーできる。複素数は実部・虚部の別配列。エラー文字列は
呼び出し側のバッファに入り、グローバル状態を使わない。戻り値の成功と数値収束は
別であり、成功時にも結果の診断を確認する。

Fortranは `use disper26_api` と `disper26_fortran` targetを使用する。
モデル・解・結果は派生型で、各演算に `status` を渡す。C++/C/Pythonの添字は0始まり、
Fortran の配列・曲線解・パラメータ・界面・ノードの添字は1始まりである。
モード番号、細分化レベル、`capability.0.operation` などの生のフィールド名は全言語で0始まり。
ハンドルやそれを成分に持つ型は代入でコピーする。allocatable 変数は領域を確保してから
代入する。`allocate(copy, source=original)` では参照カウント用の代入手続きが呼ばれない。
構築例は `call create_model(depths, "layer_cake", model_out, status, vs=vs, vp=vp, rho=rho)`。
GRIDでは `interpolation="stiffness"` または `"velocity"` を追加する。
使用例と代入・寿命の検証は [Fortran の回帰テスト](../tests/test_api_fortran.f90)を参照。

## 実装と検証

`api_backend.hpp` は既存Pythonバインディングから移設した計算アダプター、
`api_impl.hpp` と `api_related_impl.hpp` は入力検証・結果型・ディスパッチを担う。
`api_python.hpp` と `api_c.cpp` は型変換、FortranモジュールはC ABIへの委譲のみを行う。
逆解析と長周期 Taylor サロゲートは逆解析パッケージのアダプターを介して同じ数値実装を利用する。
Full AD Hessianの作業次元はスレッドごとに保持し、異なるモデルの並列計算で
干渉しないようにした。数式・演算順序は従来と同じで、並列テストは直列結果との
一致を確認する。

[READMEの検証手順](../README.md#verification)に従い、専用環境にビルド用・テスト用依存パッケージを入れ、Fortranを有効にしたターゲットを `build-native` にビルドする。以下でAPIの検証だけを実行できる。

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests/python -p 'test_public_api*.py' -v
python -m unittest discover -s tests/python -p test_model_arrays.py -v
(cd build-native && ctest -R 'public_api|fortran_' --output-on-failure)
```

ネイティブ検証にはCMakeで `DISPER26_BUILD_TESTS=ON`、
`DISPER26_BUILD_FORTRAN=ON` を指定し、対応するテストターゲットをビルドする。

導入時の検証（2026-09-24）では公開Pythonテスト26件、ネイティブCTest7件、
既存の `test_wrapper_full.py`、`test_romberg_level_root_trace.py`、
`test_variational_root_trace_consistency.py` が成功した。速度座標・対数座標のHessianは
有限差分で、EFT/Q・重力補正は既存経路で照合した。wheelを別ディレクトリに
インストールしたPREM計算と、本書のPython/C++例も確認した。
大規模な性能測定はこのAPI導入の検証には含めていない。

2026-09-25の追跡統合時には、公開API・追跡・カーネル/Hessian・低水準曲線・根列再利用を
含むPython 64件と、直近のC++曲線/C API/並列APIの3件が成功した。
この追加検証にはFortran再ビルドと大規模PREM性能試験は含まない。

## 容量再利用コンテキスト（2026-09-26）

`context = disper26.create_dispersion_context()` を作り、`compute_dispersion` または
`compute_dispersion_curve` の `context=context` に渡すと作業配列の容量を再利用する。
毎回モデルから内容を作り直し、前回の根・メッシュ・予測を再利用しない。
省略時の呼び出し方と数値・診断の契約は同じ。返却済み結果は独立に所有される。
同じコンテキストの同時呼び出しは直列化する。並列計算にはworkerごとに作成する。
失敗後も再利用でき、破棄すると保持容量を解放する。

C++にはmove可能・copy不可の `DispersionContext` と追加オーバーロードを用意する。
Cは `d26_api_dispersion_context` と生成・retain/release・`*_with_context` を使う。
Fortranは `dispersion_context` と省略可能な `context` 引数を使い、代入は所有権を共有する。
既存C構造体や呼び出しシグネチャは変更しない。未生成のcontextはエラーにする。

```python
context = d26.create_dispersion_context()
first = d26.compute_dispersion_curve(model, periods=[8., 10.], context=context)
second = d26.compute_dispersion_curve(model, periods=[12., 14.], context=context)
# 2回目の計算後も、コンテキストを破棄した後も first を参照できる。
```

保持するのは各GRIDレベルの層配列、補間用配列とRomberg作業表の容量である。
大きなモデルを計算した後も容量を保持するため、不要になればcontextを破棄する。
C/C++でも実行中のcontextの破棄やmoveは行わない。Fortranの代入によるコピーは
同じcontextを共有し、独立した作業領域には別々の生成呼び出しを使う。
