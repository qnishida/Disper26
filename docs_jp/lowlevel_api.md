# 低水準APIによる検算・詳細診断（disper26py）

[English edition](../docs/lowlevel_api.md)

モデル名は `layer_cake` を正式名とする。対応する `*_pancake` は legacy alias として
同じ引数・既定値・計算結果を維持する。新規コードでは `*_layer_cake` を使う。


mode countを直接調べる、手法別の出力を比較する、レベル別の根を明示的に渡す場合に使います。通常の計算手順は [練習例](../example_jp/README.md) にあります。ここでは `d26.Model` や `ModeSolution` の代わりに、`disper26py.PhysicalParameters`、厚さ・深度配列、角周波数・位相速度などを各関数へ渡します。

§1–8のPythonブロックは、練習例とは独立したPythonセッションで上から順に実行できます。入力の準備も本書で完結します。

例の `alpha_v`・`alpha_h` はP波速度、`beta_v`・`beta_h` はS波速度で、添字 `v` / `h` は鉛直／水平方向を表します。`depth`・`rho`・`eta` を含め、モデル入力の変数は層・ノードごとの配列です。これらは例の変数名であり、APIの引数名は変更しません。単位は深さ・厚さがkm、速度がkm/s、密度がg/cm³、弾性定数がGPaです。

| 操作 | 周期・周波数の指定 |
|---|---|
| LayerCakeの根探索・単一周期の値・感度・ヘシアン | `omega` [rad/s] |
| Layer-cake / GRIDの `compute_dispersion_curve_*` | `periods` [s] |
| `compute_dispersion_romberg*`、`compute_kernels_ad_grid`、GRID界面微分 | `period` [s] |
| `compute_hessian_variational_ad_*_grid` | `freq` [Hz] |

`omega = 2π / period`、`freq = 1 / period` と変換します。

## 目次

1. [入力の準備](#lowlevel-model)
2. [根探索とモードカウント](#lowlevel-roots)
3. [低水準バッチ分散曲線](#lowlevel-curves)
4. [群速度・H/V・固有関数・積分](#lowlevel-observables)
5. [感度カーネル](#lowlevel-kernels)
6. [位相速度ヘシアン](#lowlevel-hessian)
7. [界面深さ感度とヘシアン](#lowlevel-interface)
8. [地球平坦化とQ補正](#lowlevel-eft-q)

<a id="lowlevel-model"></a>

## 1. 入力の準備

```python
import math
import numpy as np
import disper26py as dp

# 1. 等方性媒質: create_isotropic_parameters(vs, vp, rho)
layer_solid = dp.create_isotropic_parameters(3.2, 5.8, 2.6)
half_space = dp.create_isotropic_parameters(4.5, 8.0, 3.3)

# 2. 流体/海水層: create_fluid_parameters(vp, rho)
ocean_water = dp.create_fluid_parameters(1.5, 1.0)

# 3. VTI弾性層: create_vti_layer_from_depth(A, C, F, L, N, rho, top_depth, bottom_depth)
# 戻り値: (PhysicalParameters, thickness)
vti_layer, thick = dp.create_vti_layer_from_depth(
    20.0, 20.0, 8.0, 4.0, 4.0, 2.5, 0.0, 10.0
)

# LayerCakeの入力一式（half_spaceは有限層に含めない）
layers = [layer_solid]
thicknesses = [15.0]

# GRIDの入力一式。低水準の形式で用意する
depth = [0.0, 10.0, 30.0, 30.0, 100.0]
rho = [2.5, 2.7, 2.9, 3.3, 3.4]
alpha_v = [5.5, 6.0, 6.5, 8.0, 8.1]
beta_v = [3.2, 3.5, 3.8, 4.5, 4.6]
is_disc = [False, False, True, False, False]  # 30 kmの浅い側をTrueにする
grid_params = [dp.create_isotropic_parameters(b, a, r)
               for b, a, r in zip(beta_v, alpha_v, rho)]
# 速度補間API用のVTI配列。この例は等方性なので水平・鉛直速度は等しい
alpha_h = list(alpha_v)
beta_h = list(beta_v)
eta = [1.0] * len(depth)
```

---

<a id="lowlevel-roots"></a>

## 2. 根探索とモードカウント

```python
period_layer_cake = 5.0  # s
omega = 2.0 * math.pi / period_layer_cake

# 根探索の末尾4引数は c_min, c_max, n_search, wave_type (位置引数)
roots_rayleigh = dp.search_roots(
    layers, thicknesses, half_space, omega, 2.0, 4.4, 200, "Rayleigh"
)
roots_love = dp.search_roots(
    layers, thicknesses, half_space, omega, 3.2, 4.4, 200, "Love"
)
if not roots_rayleigh or not roots_love:
    raise RuntimeError("根が見つかりません。モデル・探索範囲を確認してください。")
c_rayleigh = roots_rayleigh[0]
c_love = roots_love[0]

# この単純モデルの基本モードを、根の両側のカウントの跳びで確認
delta_rel = 3e-4  # 既存の分岐確認で使用する相対窓幅
for wave, cp in (("Rayleigh", c_rayleigh), ("Love", c_love)):
    counts = [dp.compute_mode_count(
        layers, thicknesses, half_space, omega, cp * (1.0 + sign * delta_rel), wave
    ) for sign in (-1, 1)]
    assert counts == [0, 1], (wave, counts)

# 旧実装の連続カウントは診断用。整数部をモード番号として使わない
cont_mode = dp.compute_continuous_mode_count(
    layers, thicknesses, half_space, omega, c_rayleigh, "Rayleigh"
)
```

根そのものの整数カウントは丸めに左右されます。近接した根では上の固定窓に複数の根が入ることがあるため、窓の調整と分岐確認が必要です。`search_roots` の符号変化スキャンには取り逃しの可能性もあります。分岐を指定したGRID計算には§5の `branch` または [高度な利用](../example_jp/README.md#advanced) の `ForwardEngine` を使い、診断は [モードカウント](../docs/mode_counting.md) を参照してください。

---

<a id="lowlevel-curves"></a>

## 3. 低水準バッチ分散曲線の計算

この節は `disper26py` の既存バッチです。通常の計算は [練習例](../example_jp/README.md) の共通APIを使います。
両者は数値カーネルを共有しますが、探索・予測の制御は同一ではありません。
低水準バッチには共通APIの `predictor` / `verify_mode` を渡せません。

複数周期の位相速度はC++バッチAPIで計算できます。周期列は正・有限で、重複のない昇順または降順にします。GILを解放するため独立したモデル等をPythonスレッドで並列化できますが、一つの曲線内の追跡は逐次処理です。

```python
periods = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]  # 単調な周期列

# Layer-cake モデルの一括計算
layer_cake_curve = dp.compute_dispersion_curve_layer_cake(
    layers, thicknesses, half_space,
    periods=periods, wave_type="Rayleigh", branch=0,
    c_min=2.0, c_max=4.4, verify_branch=True
)
c_layer_cake = layer_cake_curve["phase_velocities"]

# GRID モデル (速度基底): §1で構築した入力を使う
grid_curve = dp.compute_dispersion_curve_romberg_grid_velocity_vti(
    depth, rho, alpha_h, alpha_v, beta_h, beta_v, eta,
    is_disc, periods, wave_type="Rayleigh", c_guess=3.5, branch=0
)
c_grid_velocity = grid_curve["phase_velocities"]
print("GRID fallback回数:", grid_curve["fallback_count"])
```

LayerCakeの `branch=0` は初回・再探索時の速度順の基本モードを選び、`verify_branch=True` は予測した根の整数カウントも確認します。GRIDの `branch=0` は各レベルでモード順を指定します。既定の `branch=None` は近い根を追跡する経路で、分岐を指定した場合とは処理時間も異なります。

戻り値には `relative_errors`、`converged_levels`、`fallback_mask` もあります。fallbackは予測から通常探索へ戻ったことを示し、それ自体は失敗ではありません。LayerCakeのレベルは `-1`、相対誤差は `NaN` です。バッチと単点計算の整合性・fallbackの回帰確認は [test_phase_velocity_curves.py](../tests/python/test_phase_velocity_curves.py) を参照してください。

---

<a id="lowlevel-observables"></a>

## 4. 群速度・H/V比・固有関数・エネルギー積分

```python
# 1. 群速度 U (Full AD による厳密微分)
U = dp.compute_group_velocity_ad(
    layers, thicknesses, half_space, omega, c_rayleigh, "Rayleigh"
)

# 2. H/V比 (符号付き楕円率 y3/y1)
# ※流体層が存在する場合は、自動的に海底面 (固体表面) で評価されます
hv = dp.compute_hv_ratio(
    layers, thicknesses, half_space, omega, c_rayleigh, "Rayleigh"
)

# 3. 固有関数 (Rayleigh: ur, uz, szr, szz / Love: uy, tyz)
# 分割数は位置引数で渡す
ef_rayleigh = dp.compute_eigenfunctions_rayleigh(
    layers, thicknesses, half_space, omega, c_rayleigh, 50
)

# 4. 解析的エネルギー積分 (固体モデル)
I1, I2, I3, c_energy = dp.compute_energy_integrals_analytical_rayleigh(
    ef_rayleigh, layers, thicknesses, half_space, omega, c_rayleigh
)
```

---

### 符号付きRayleigh楕円率

`hv_ratio` は符号付きの比 ε = y3/y1 で、`d_HV` と `hv_ratio` の勾配はその微分です。
従来の振幅比から意味が変わりました。C・Fortranと旧名を含め、同じ規約を使います。
zを上向き、波の位相を exp(i(kx−ωt)) とすると、固有関数配列 `uz` と `ur` は
それぞれ y1 と y3 を格納し、水平変位の複素振幅は i·y3 です。
正のx方向に伝播するとき、ε < 0 は逆行、ε > 0 は順行を表します。

Layer-cakeとGRIDでは、固体自由表面、または海底の固体側で評価します。
せん断応力ゼロの条件から、固体側の小行列式を用いて ε = −Y12/Y14 と表せます。
従来の海底振幅には自由表面の法線応力条件が使われていました。
GRIDでは値と勾配を符号付きで外挿します。一定物性の区間を一様に摂動した場合の層感度は、
その区間の両端ノードの感度を足して比較できます。

振幅比 E = |ε| の微分は、ε ≠ 0 では sign(ε)·dε/dp です。
水平変位がゼロを横切るときは |ε + 勾配 · Δp| から振幅を予測できます。
滑らかなゼロ交差では符号付き比を微分できますが、振幅の微分と正規化感度
(p/ε)·dε/dp は特異になります。鉛直変位がゼロの点は比の極で、値と感度は未定義です。 低水準の比はこの極で NaN を返します。

<a id="lowlevel-kernels"></a>

## 5. 感度カーネル (1階微分: dc/dp, dU/dp, d(H/V)/dp)

### Layer-cake モデル

位相速度・群速度・H/Vの感度をFull ADで計算します。LoveのH/Vは定義されず、対応する値・感度は `NaN` になります。

```python
kernels = dp.compute_kernels_ad_layer_cake(
    layers, thicknesses, half_space, omega, c_rayleigh,
    wave_type="Rayleigh",
    sensitivity_mode="Isotropic",  # または "VTI"
    compute_thickness=False        # 層厚さ感度も求める場合は True
)

for k in kernels:
    print(f"Layer {k.layer_index} [{k.parameter_name}]: dc/dp={k.d_c:.6e}, dU/dp={k.d_U:.6e}, d(HV)/dp={k.d_HV:.6e}")
```

`layer_index` は有限層の0始まりの番号で、半無限媒質は `-1` です。このLayerCake APIの物性成分の順序は `Isotropic` が `(Vs,Vp,Rho)`、`VTI` が `(Rho,A,C,F,L,N)` です。感度の単位は出力量の単位を入力パラメータの単位で割ったものです。

### GRID モデル (Romberg 加速)

ここから§7までは**弾性定数線形補間のモデル**を使います。§3の速度補間の曲線とは区別し、このモデルの10秒・Rayleigh基本モードの根を求め直します。レベル別の根を共有して、感度とヘシアンの分岐をそろえます。

```python
period_grid = 10.0
derivative_level = 5
trace = dp.compute_dispersion_romberg_levels(
    depth, grid_params, is_disc, period=period_grid,
    wave_type="Rayleigh", c_guess=3.5, branch=0,
    max_steps=derivative_level, minimum_trace_level=derivative_level,
)
grid_roots = trace["phase_velocity_roots"]
accepted_level = trace["phase_accepted_level"]

c_grid, U_grid, hv_grid, level, rel_err, kernels_grid = dp.compute_kernels_ad_grid(
    depth, grid_params, is_disc,
    period=period_grid, wave_type="Rayleigh", c_guess=trace["phase_velocity"],
    sensitivity_mode="Isotropic", max_steps=derivative_level,
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
print("GRID c, level, rel_err:", c_grid, level, rel_err)
```

GRIDの `layer_index` はノード番号です。物性感度のほか、不連続面深さの `Depth` 成分が含まれます。`dU/dp` の既定経路は `variational_frequency_ad` です。要求しなかった感度成分はゼロなので、ゼロを必ずしも物理的な無感度と解釈しないでください。
界面深さ感度を有限差分で確認する際は、同じ深さにある上下のノードを一緒に動かします。[Love波の群速度の回帰テスト](../tests/python/test_u_kernel_frequency_variational_grid.py)に例があります。

`max_steps` / `max_levels` は細分化の上限です。返された `rel_err` とレベルを確認し、要求精度を満たすか判断してください。この5レベル例は実行方法を示すもので、すべての量が指定許容値まで収束したという保証ではありません。

---

<a id="lowlevel-hessian"></a>

## 6. 位相速度ヘシアン (2階微分: d²c/dp²)

主要経路は**変分法 + 1階AD**です。速度と精度はモデル・積分法・バックエンドに依存します。比較条件は[実装マップ](sensitivity_implementation_map.md) を参照してください。Full ADの `compute_hessian_ad_layer_cake` は独立した検算に利用できます。

### 1. Layer-cake モデルのヘシアン

```python
# Rayleigh波
res_rayleigh = dp.compute_hessian_variational_ad_rayleigh_layer_cake(
    layers, thicknesses, half_space, omega, c_rayleigh,
    sensitivity_mode="Isotropic",
    use_analytical=True  # 閉形式の積分。固体上の連続した表面流体層にも対応
)
H = res_rayleigh["hessian"]   # 2次元 numpy array
g = res_rayleigh["gradient"]  # 1次元 numpy array

# Love波
res_love = dp.compute_hessian_variational_ad_love_layer_cake(
    layers, thicknesses, half_space, omega, c_love,
    sensitivity_mode="Isotropic", use_analytical=True,
)
```

### 2. GRID モデルのヘシアン (ノードパラメータ感度)

```python
res_grid = dp.compute_hessian_variational_ad_rayleigh_grid(
    depth, grid_params, is_disc,
    freq=1.0 / period_grid, c_guess=c_grid,
    sensitivity_mode="Isotropic", max_levels=derivative_level,
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
H_grid = res_grid["hessian"]
parameter_labels = list(zip(res_grid["layer_indices"], res_grid["parameter_names"]))
print("GRID Hessian:", H_grid.shape, parameter_labels)
```

この5ノード・等方性例の物性ヘシアンは15×15です。界面深さは含まれず、§7で別に扱います。配列の対応には必ず `parameter_names` と `layer_indices` を使ってください。API間で同じ列順とは限りません。

固体媒質上に流体層が連続している場合、Rayleigh波のLayer-cake弱形式では解析積分も使えます。低水準の既定値 `use_analytical=False` は台形積分です。GRIDヘシアンでは流体を波動場に含めますが、**流体ノードは既知・固定**とし、対応する勾配・ヘシアン行列要素はゼロです。海底は重複深さの不連続面として指定します。

---

<a id="lowlevel-interface"></a>

## 7. 界面深さ感度とヘシアン (Kd)

LayerCakeの $K_d$ は各境界のカーネルです。幾何パラメータ $q$ に対する感度は $dc/dq=\sum_b [K_d]_b\,dz_b/dq$ で、層厚変更と一つの界面だけの移動では動く境界が異なります。GRIDの界面APIは指定した不連続面の深さ微分を組み立てます。

```python
# Layer-cake 界面感度 Kd
kd_rayleigh = dp.compute_interface_kernels_variational_rayleigh_layer_cake(
    layers, thicknesses, half_space, omega, c_rayleigh
)

# 30 kmの不連続面: 不連続面番号は0、浅い側のノード番号は2
discontinuity_index = 0
grad_z = dp.compute_interface_gradient_variational_grid(
    depth, grid_params, is_disc,
    period=period_grid, wave_type="Rayleigh", c_root=c_grid,
    discontinuity_index=discontinuity_index, max_level=derivative_level,
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
row_z = dp.compute_interface_hessian_row_variational_ad_grid_extrapolated(
    depth, grid_params, is_disc,
    period=period_grid, wave_type="Rayleigh", c_root=c_grid,
    discontinuity_index=discontinuity_index, max_level=derivative_level,
    sensitivity_mode="Isotropic",
    phase_velocity_roots=grid_roots, phase_accepted_level=accepted_level,
)
```

この例の `row_z` は、各ノードの `(Vs,Vp,Rho)` 順の15成分と、不連続面深さの1成分からなる16要素です。最後は $d^2c/dz^2$ です。物性ヘシアンとは出力形状が異なります。

---

<a id="lowlevel-eft-q"></a>

## 8. 地球平坦化変換 (EFT) と Mineos 減衰 Q 補正

以下は**変換方法の独立した例**です。同じ数値を球面モデルの入力例として扱い、§1–7の平面モデルやその根は上書きしません。

```python
r_earth = 6371.0

# 1. Layer-cake モデルの平坦化 (Bhattacharya & Arora 1997, デフォルト p=-2)
flat_layers, flat_thick = dp.earth_flattening_transformation_layers(
    layers, thicknesses, planet_radius=r_earth
)

# 2. GRID モデルの平坦化
flat_depths, flat_params, flat_disc = dp.earth_flattening_transformation_grid(
    depth, grid_params, is_disc, planet_radius=r_earth
)

# 3. 元の参照周期モデルを、対象周期20秒に補正
corrected_layers = dp.apply_mineos_physical_dispersion_model(
    layers, qkappas=[57823.0], qmus=[600.0], tref_s=1.0, period_s=20.0
)
```

LayerCakeの平坦化関数は渡した有限層だけを変換し、`half_space` を自動変換しません。上の出力だけでは半無限媒質まで平坦化した一式にはなりません。GRID関数は最深ノードも変換します。

GRIDのノードを先に平坦化して `grid_policy="flat"` で補間するモデルと、球面ノードを渡して `grid_policy="spherical_ba97"` で細分点ごとに補間・変換するモデルは異なります。後者では事前のノード平坦化を重ねないでください。モデル空間の扱いは [実装マップ](sensitivity_implementation_map.md) を参照してください。

Q補正も渡したリストのみが対象です。半無限媒質には `apply_mineos_physical_dispersion` を別途適用します。Qのリスト長は物性リストに合わせ、各対象周期の補正は元の参照周期モデルから行います。この補正は実弾性定数の物理分散で、複素減衰波動場の計算ではありません。補正後の分散を求める場合は、その対象周期で根を計算し直します。

---
