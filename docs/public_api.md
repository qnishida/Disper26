# 共通公開API

通常の順計算は Python の `disper26`、C++ の `disper26::api`、Fortran の
`disper26_api` から利用する。三つの入口は同じC++実装に委譲する。
従来の `disper26py` と Fortran `disper26` は互換APIとして引き続き利用できる。
順に実行するPythonの例は [練習例](../examples/README.md)、
`disper26py` による検算・詳細診断は [低水準API利用ガイド](lowlevel_api.md) を参照。

## 海底重力オプションの削除

未検証の追加 traction ジャンプを導入する `gravity_seabed_scale` は削除した。
低水準 Python の `create_fluid_parameters` の `seabed_scale` 引数も削除したため、
旧呼び出しでは該当キーワードまたは第6位置引数を除く。
`Material` の同名キーワード・第10位置引数、物性への同名フィールド設定も除く。
海底では法線変位と Lagrangian 法線 traction の連続条件を使用する。
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
kernels = d26.compute_kernels(solution, basis="isotropic")
hessian = d26.compute_hessian(solution, basis="isotropic")
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
重複する二つの深度は不連続面の上下を表す。最終ノードが半無限層となる。
曲線は周期を内部で整列・重複除去して計算し、入力順に戻す。`mode` は各周期の
位相速度順の番号であり、固有関数によるモード同一性の追跡ではない。

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
`mode_order` で `verify_mode=False` は拒否する。`level_mode_verified` は実際に
確認した根だけtrueとなる。端点の確認だけでは途中のbranch jumpを排除できない。

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
curve = disper26.compute_dispersion_curve(
    model, periods=periods,
    options=disper26.Options(method="continuation", verify_mode=False, c_guess=3.0))
last_mode = disper26.compute_mode_count(curve.solutions[-1])
first_coarse_mode = disper26.compute_mode_count(curve.solutions[0], level=0)
```

曲線は既定で最長周期から最短周期へ内部計算するため、`c_guess` は最長周期の初期推定となる。
従来の計算方向を指定するには `Options(start="short")` を使う。
返却順は入力順なので、上の `[-1]` は入力の最後の地点を意味する。
kernel/Hessianなど微分計算が要求する保存根の番号確認は従来通り別途行う。
確認付きautoのレベル1以降では、同一周期の粗い根を使う。群速度予測を利用し得るのは
レベル0であり、cのみを要求したautoでも初回に群速度を計算する。群速度による一次予測は
`dc/dω=(c/ω)(1-c/U)` を使い、前の角周波数からの変化が25%以内の場合に候補とする。
条件に応じて履歴や粗いレベルの情報へ切り替える。
`"group_velocity"` はAD評価を許可し、出力にUを要求した場合はその評価を共用する。
未要求の公開群速度は引き続きNone。局所追跡に失敗した場合、`continuation` は
予測速度の±20%内の根を探索して最も近いものを選び、根がなければ例外を返す。
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
未確認解から感度・Hessian・周波数曲率を求める前には保存済みの各レベル根を再確認する。
不一致なら `mode_mismatch` を返し、解を別の根へ差し替えない。
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
これらの追跡機能はmain統合済みで、統合判断と
今後の最適化は別に記録する。

### 開始方向・初期速度範囲・確認頻度

| Options | 指定内容 | 省略時 |
|---|---|---|
| `start` | `"long"`: 最長周期から、`"short"`: 最短周期から | `"long"` |
| `c_range` | 開始点の位相速度範囲 `(lo, hi)` [km/s] | なし |
| `verify` | `"none"`: 通常の番号確認を省略、`"endpoints"`: 有効区間の開始・終了、`"every"`: 毎周期 | 上記のmethod別の動作 |
| `on_violation` | `"fallback"`: 指定番号を再探索、`"error"`: 例外、`"missing"`: 欠損値 | 新しい確認・範囲設定では `"fallback"` |

```python
options = disper26.Options(
    start="long", c_range=(2.8, 4.2),
    verify="endpoints", on_violation="fallback")
curve = disper26.compute_dispersion_curve(model, periods=[2., 5., 10., 20.], options=options)
```

この例は20→10→5→2秒で計算し、結果は指定した2・5・10・20秒の順に返す。
重複周期は一度だけ計算する。`c_range` は最初に解を得られる周期の各生成GRIDレベルに
適用し、その開始点では `c_guess` を使わない。高次モードが最長の要求周期に存在しない場合、
欠損周期を飛ばした最初の有効周期に範囲を適用する。
範囲の両端は有限・正で `lo < hi` が必要。上端を半無限媒質S波速度の0.9995倍で
切り詰め、空になった範囲は入力エラーとする。Rayleighは鉛直、Loveは水平S波速度を使う。
確認付きの初期範囲が要求番号を1つだけ挟み、永年関数の有限な符号変化も得られれば
広い区間から直接Brent法を使う。接する根や残差の問題には従来の狭い区間での探索を使う。

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
欠損解のモード数・感度・ヘシアンなどの派生計算は `missing_solution` エラーを返す。


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

## 微分の座標系と対応範囲

`basis` は明示指定する。モデルの補間方式とは独立である。

| basis | 各層・ノードの軸順 |
|---|---|
| `stiffness` | A, C, F, L, N, Rho |
| `isotropic` | Vs, Vp, Rho（等方モデルに限定） |
| `velocity_vti` | rho, vph, vpv, vsh, vsv, eta |
| `log_velocity_vti` | 上記6量の自然対数（能動座標は正値に限定） |

半無限層は `entity="halfspace"` で区別する。速度・対数座標のHessianには
`Jᵀ H J` に加え、物性写像の2階微分と勾配の項も含める。
`compute_hessian_reduced(solution, projection)` は剛性軸×縮約軸の**線形**写像を受け取る。
非線形写像の曲率項はこの関数の対象外。フルHessianを生成する代替処理は行わない。

| 操作 | 対応 |
|---|---|
| 分散、曲線、カーネル | Layer-cake、剛性GRID、速度GRID |
| フルHessian | Layer-cake、剛性GRID |
| 縮約Hessian | 剛性GRID |
| 界面感度 | Layer-cake、flatの剛性GRID |
| 界面Hessian行 | flatの剛性GRID |
| 固有関数・積分・周波数曲率 | 解を保持する各モデル。GRIDの波動場と積分は採用メッシュ上の量 |
| 重力補正 | 非Cowling、flat LayerCakeのRayleigh。`region="fluid"/"solid"/"all"` |

GRIDの周波数曲率は固有のRomberg停止条件を使うため、戻り値の位相速度が元の
解と異なる場合がある。固有関数・積分の採用メッシュ上の値とは区別する。

流体を含むLoveのモード探索は対象外。実際のモデルについて
`get_capabilities(model, wave=...)` が対応可否と理由を返す。非対応機能は
`Error(code="unsupported")` とし、別のモデルへ置き換えない。
Cowling重力を設定したモデルの物性・界面微分は、既存AD経路が重力項を保持しない
ため共通APIでは拒否する。通常の用途では非重力の解と別途のKg補正を使う。

標準経路は `Options(method="standard")`。明示的な `ad`、`variational`、
`variational_ad` の可否は演算ごとに異なる。GRIDの固定レベルFull ADなどの
検算専用機能は従来の低水準APIで利用する。

## 変換とPREM

球面モデルは `Geometry("spherical_ba97")` または
`Geometry("spherical_experimental_density", flattening_exponent=...)` で表す。
GRIDの球面補間と、ノードを先に平坦化してからflatで補間するモデルは同一ではない。
`earth_flattening_transformation` は新しいモデルと変換履歴を返し、二重変換を拒否する。
球面LayerCakeは明示的な平坦化が必要。
LayerCakeの半無限層は、その上端深度で物性を平坦化した均質半無限層とする。

`apply_mineos_physical_dispersion` は半無限層を含むQ配列を受け取り、補正周期と
基準周期を保存する。LayerCakeでは層数+1、GRIDではノード数。補正済みモデルの
再補正・別周期での利用を拒否する。周期列を計算するときは元モデルから各周期用の
モデルを作る。

`load_prem_spherical` / `load_prem_flat` はパッケージ同梱データを使用し、逆解析や
SciPyをimportしない。海水層は地殻物性で置き換える `crust_fill` に固定し、
打切り深度・最大間隔・平坦化指数は引数で指定できる。

重力補正は既存の採用済み摂動式を使う。既定は層内40分割、角周波数の相対差分幅
0.005で、これらの値を変える場合は対象モデルで収束を確認する。結果は元の解を
変更せず、補正前の速度と `delta_c` / `delta_u` を別フィールドで返す。

## C++・C・Fortran

C++は `#include <api.hpp>` とCMake target `disper26::api` を使用する。
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
    auto kernels = d26::compute_kernels(solution, "isotropic");
    return kernels.parameters.empty();
}
```

Cは `api_c.h` の `d26_api_*` を使用する。`d26_api_create_model` は深度列と
「成分×点」の列優先物性行列を受け取る。`parameterization` と成分順は
`isotropic`: Vs,Vp,Rho、`velocity_vti`: rho,vph,vpv,vsh,vsv,eta、
`stiffness`: A,C,F,L,N,Rho。LayerCakeの補間引数はNULLとする。結果フィールドを列挙し、必要サイズを
照会してからバッファへコピーできる。複素数は実部・虚部の別配列。エラー文字列は
呼び出し側のバッファに入り、グローバル状態を使わない。戻り値の成功と数値収束は
別であり、成功時にも結果の診断を確認する。

Fortranは `use disper26_api` と `disper26_fortran` targetを使用する。
モデル・解・結果は派生型で、各演算に `status` を渡す。C++/C/Pythonの添字は0始まり、
Fortranの配列・界面・ノード添字は1始まり、モードだけは共通で0始まり。
構築例は `call create_model(depths, "layer_cake", model_out, status, vs=vs, vp=vp, rho=rho)`。
GRIDでは `interpolation="stiffness"` または `"velocity"` を追加する。
使用例と代入・寿命の検証は `tests/test_api_fortran.f90` を参照。

## 実装と互換性

`api_backend.hpp` は既存Pythonバインディングから移設した計算アダプター、
`api_impl.hpp` と `api_related_impl.hpp` は入力検証・結果型・ディスパッチを担う。
`api_python.hpp` と `api_c.cpp` は型変換、FortranモジュールはC ABIへの委譲のみを行う。
既存逆解析・Plan Lは独立した用途別パッケージとして残し、既存アダプター経由で
同じ数値実装を利用する。逆解析のパラメータ化やキャッシュ契約は変更していない。
Full AD Hessianの作業次元はスレッドごとに保持し、異なるモデルの並列計算で
干渉しないようにした。数式・演算順序は従来と同じで、並列テストは直列結果との
一致を確認する。

検証コマンド（[インストール手順](install.md)で用意した専用環境内で実行）:

```bash
python setup.py build_ext --inplace
python -m unittest discover -s tests/python -p 'test_public_api*.py' -v
python -m unittest discover -s tests/python -p test_model_arrays.py -v
ctest --test-dir build -R 'public_api|fortran_' --output-on-failure
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
条件・精度差・ログを参照。
この追加検証にはFortran再ビルドと大規模PREM性能試験は含まない。


## 容量再利用コンテキスト（2026-09-26）

`context = d26.create_dispersion_context()` を作り、`compute_dispersion` または
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
# first remains usable after the second calculation or context destruction.
```

保持するのは各GRIDレベルの層配列、補間用配列とRomberg作業表の容量である。
大きなモデルを計算した後も容量を保持するため、不要になればcontextを破棄する。
C/C++でも実行中のcontextの破棄やmoveは行わない。Fortranの代入によるコピーは
同じcontextを共有し、独立した作業領域には別々の生成呼び出しを使う。


### Signed Rayleigh ellipticity

`hv_ratio` is the signed ratio epsilon = y3/y1, and `d_HV` and the
`hv_ratio` gradient contain derivatives of that signed quantity. This is a
semantic change from the former amplitude-only value; API names and shapes
are unchanged, including deprecated aliases and the C/Fortran interfaces.
With z upward, exp(i(k*x-omega*t)), u_z=y1*exp(...) and
u_x=i*y3*exp(...), epsilon < 0 is retrograde and epsilon > 0 is prograde
for propagation in positive x. The eigenfunction arrays `ur` and `uz`
store y3 and y1, respectively; the horizontal phase factor i is implicit.

Both layer-cake (`layer_cake`) and GRID models evaluate the solid free surface,
or the solid side of the seabed when surface fluid layers are present.
The common expression is -Y12/Y14, obtained from zero shear traction.
This also corrects the former seabed amplitude, which used a free-surface
normal-traction relation under fluid loading. GRID values and gradients are
extrapolated while signed. A uniform layer perturbation corresponds to
varying both endpoints of its constant GRID segment; sum their node
sensitivities to compare with the layer sensitivity.

For an amplitude observable use E = abs(epsilon). Away from epsilon=0,
dE/dp = sign(epsilon)*d(epsilon)/dp, not abs(d(epsilon)/dp).
Across a smooth zero crossing predict abs(epsilon + gradient @ delta_p).
The signed ratio is differentiable at a horizontal-displacement zero;
the amplitude and normalized sensitivity (p/epsilon)*d(epsilon)/dp are not
regular there. A vertical-displacement zero is a pole of the ratio: values
and sensitivities are undefined (NaN), not clipped or replaced by zero.
The existing Love-wave and unrequested-output availability rules still apply.

The existing high-level restriction on material/interface derivatives with
Cowling gravity remains in force. Low-level signed-ellipticity checks do not
validate every derivative offered by that high-level interface.
