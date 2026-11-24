# Grid Mode WF Hessian — Romberg加速（台形積分ベース）分析

[English edition](../docs/grid_wf_hessian.md)

> 現行の GRID 変分 Hessian は Rayleigh 波で台形積分、Love 波で解析積分を使う。
> 以下には両波で台形積分を検討した時点の設計も含む。
> [感度・Hessian実装マップ](sensitivity_implementation_map.md) と [共通API](public_api.md) を参照。

## 改訂履歴
- 初版: 解析積分ベースの案A/B/Cを検討
- 改訂: エネルギー積分の収束次数分析を反映。台形積分 + Romberg加速を主戦略として採用。

---

## 1. 収束次数と積分手法の関係

### 1.1 Richardson外挿の前提

Romberg加速（Richardson外挿）は誤差の偶数べき展開を前提とする:
```
f(h) = f(0) + C₂h² + C₄h⁴ + ...   ← これが成り立つとき比率4の外挿が有効
```

表面波の layer-cake 近似の位相速度はmidpoint ruleの超収束（superconvergence）により
この構造を満たすことが確認されている（比率 ≈ 4.0、[Romberg 外挿の収束検証](../docs/Romberg.md#convergence-verification)参照）。

### 1.2 エネルギー積分の手法と収束次数

WFの変分勾配は `g_i = c³/(2ω²I₃) · (∂I₂/∂pᵢ − ω²∂I₁/∂pᵢ)` を通じてエネルギー積分に依存する。

| 積分手法 | 層内評価 | O(h²)の偶数べき構造 | Romberg外挿 |
|----------|---------|-------------------|------------|
| **解析積分** | 均質層として閉形式 ∫exp(μᵢz)exp(μⱼz)dz | ❌ 崩壊 | ❌ 外挿が悪化 |
| **台形積分** | 境界点の状態ベクトルを使用 (f₀+f₁)/2·h | ✅ 維持 | ✅ 有効 |

**解析積分が収束次数を崩す理由（近似の不整合）**:
- 伝播行列は midpointパラメータで境界点を O(h²) 精度で計算
- 解析積分は均質副層の閉形式で「正確すぎる」積分を実行
- → 伝播(O(h²))と積分(副層内正確)の近似レベルが不整合
- → layer-cake 近似の誤差が偶数べき展開に従わず、比率=4の外挿が不適切になる

**台形積分が収束次数を維持する理由（近似の一貫性）**:
- 伝播行列：境界点 y₀, y₁ を O(h²) で計算
- 台形積分：同じ境界点を使い ∫f dz ≈ (f₀+f₁)/2·h → O(h²)
- → 両者が同じ近似レベル → Euler-Maclaurin構造が保存 → O(h²)外挿が有効

**過去の経験との対応**: エネルギー積分を解析解→台形に変更したところ
Romberg収束が改善されたのは、この「近似の一貫性」が回復したためである。

### 1.3 設計時点の積分経路

| 関数 | 積分手法 | 対応波動 | Romberg外挿との相性 |
|------|---------|---------|-------------------|
| `computeLoveAnalyticalIntegrals` | 解析 | Love | ❌ |
| `computeRayleighEnergyIntegrals_Analytical` | 解析 | Rayleigh | ❌ |
| `computeRayleighEnergyIntegrals_CT` (`useAnalytical=false`) | **台形** | Rayleigh | ✅ 既存 |
| Love台形積分 | **台形** | Love | 設計時は未実装 |

`computeVariationalGradientRayleigh_T<T>` は `useAnalytical` フラグで分岐済み。

---

## 2. Grid Mode WF Hessian の設計

### 2.1 数学的構造（再整理）

**設計時のWF Hessian（layer-cake モデル）:**
- パラメータ: 各均質層の物性 p₁,...,p_L（L = 層数）
- Hessian行列サイズ: (L×npl)²

**Grid mode WF Hessian（ノードベース + Romberg）:**
- パラメータ: グリッドノードの物性 q₁,...,q_M（M = ノード数 ≪ L）
- 補間: p_layer = f(q_nodes)（Mesher線形補間、ADグラフ内）
- Hessian行列サイズ: (M×npl)²  ← 大幅に小さい
- Romberg外挿により離散化誤差を O(h²) → O(h⁴) → O(h⁶) と系統的改善

### 2.2 主戦略: 台形積分 + Romberg Hessian

```
for k = 0, 1, ..., K:
    layers_k, thick_k = mesher.get_refined_layers_ad<adual>(k, node_params_dual)
    H_k = compute_WF_Hessian_trapezoidal(layers_k, thick_k, hs, ω, c)
    # useAnalytical=false（台形積分）で N-pass Hessian計算
H_final = neville_extrapolation_matrix(H_0, ..., H_K)  # 要素ごと外挿
```

外挿が有効な根拠:
- 台形積分 + 伝播行列 → 一貫した O(h²) 近似 → Hessian要素も O(h²) 収束
- K=2-3レベルの計算 + 外挿 ≈ K=5-6レベル相当の精度
  （位相速度で確認済みの比率≈4がHessianでも成立する見込み）

**実装イメージ（Romberg Hessianループ）:**
```cpp
std::vector<Eigen::MatrixXd> H_table;

for (int k = 0; k <= K; ++k) {
    // Mesherで2^k副層を生成（ADグラフ内）
    auto [layers_k, thick_k] = mesher.get_refined_layers_ad<adual>(k, node_dual);
    auto hs_k = mesher.get_halfspace_ad<adual>(node_dual);

    // WF Hessian（台形積分、useAnalytical=false）
    auto H_k = compute_node_wf_hessian_trapezoidal(
        layers_k, thick_k, hs_k, omega, c_root, g0_node, mode);
    H_table.push_back(H_k);

    // 収束チェック（案B用adaptive stopping）
    if (k >= 1) {
        auto err = (H_table.back() - H_table[H_table.size()-2]).norm()
                   / H_table.back().norm();
        if (err < tol || err > prev_err) break;  // adaptive stopping
        prev_err = err;
    }
}

// Richardson外挿（要素ごと）
H_final = neville_extrapolation_matrix(H_table);
H_final = (H_final + H_final.transpose()) * 0.5;  // 対称化
```

### 2.3 Two-phase案（シンプル版）

Romberg外挿が不要な場合の代替:

```
Phase 1: 既存Rombergで位相速度を収束 → converged_level
Phase 2: converged_levelの副層でWF Hessian（台形積分）を1回計算
```

- 外挿なし → 収束次数の問題を完全回避
- 解析積分 or 台形積分どちらでも使用可能（外挿しないため）
- 精度は converged_level での layer_cake近似誤差 O(h²) に依存
- **設計時のこの案では `useAnalytical=true` も使用可能**

---

## 3. 実装上の技術的課題

### 3.1 Rayleigh波: 台形積分パスは既存

`computeRayleighEnergyIntegrals_CT` (`useAnalytical=false`) が台形積分を実装済み。
`computeVariationalGradientRayleigh_T<T>` は `useAnalytical` フラグで切替可能。

**必要な変更**: WF Hessian関数 `computePhaseVelocityHessianWeakFormRayleigh` を
ノードベースパラメータ対応に拡張するのみ。

### 3.2 Love波で設計時に必要だった台形積分

設計時には `computeLoveAnalyticalIntegrals`（解析積分）のみが利用できた。

以下は設計時のシグネチャ。実装は `LoveEnergyTerms<T>` を返し、分割数も引数に取る。

```cpp
template <typename T>
LoveEnergyResult<T> computeLoveEnergyIntegrals_Trap(
    const std::vector<LoveLayerState<T>>& states,  // 境界点の状態ベクトル
    const std::vector<PhysicalParameters<T>>& layers,
    const std::vector<T>& thicknesses,
    const PhysicalParameters<T>& halfSpace,
    T omega, T cp
) {
    // 台形則: ∫u²dz ≈ Σ (u[j]² + u[j+1]²)/2 * dz
    // 境界点は propagateLoveEigenstates が提供するが、
    // 現状は各層の境界1点のみ → numSubLayers対応が必要
}
```

`propagateLoveEigenstates` に `numSubLayers` 引数を追加し、
各層を副層分割してstate listを生成する変更が必要。

### 3.3 IFTのノードベース対応

設計時のWF Hessianは層ベースの `g0[j] = ∂c/∂p_layer_j` をIFTのseedに使用。

ノードベースでは `g0[j] = ∂c/∂q_node_j` が必要:
```cpp
// Phase 2 冒頭で、ノードベース勾配を計算
// 方法1: WF層勾配 g_layer を計算し、補間転置でノード勾配に変換
auto g_layer = compute_wf_gradient_double(layers_k, thick_k, ...);  // useAnalytical任意
Eigen::VectorXd g0_node = interpolation_transpose(g_layer, mesher, k);

// 方法2: 既存Rombergカーネル（AutoDiffScalar）から取得（ADフレームワーク不一致に注意）
```

方法1を推奨。補間重みは定数（パラメータ非依存）なので変換は厳密。

### 3.4 autodiff::dual と Mesher の互換性

`get_refined_layers_ad<T>` の `T h = T(sub_thickness)` コンストラクタが
`autodiff::dual` で動作するか確認が必要（現行は `Eigen::AutoDiffScalar` で実績あり）。

`autodiff::dual` は `double` からの暗黙変換が可能なため、動作する見込みが高い。

### 3.5 ADフレームワーク統一

| コンポーネント | 設計時のAD型 | Grid Hessianでの想定 |
|-------------|---------|-------------------|
| Rombergカーネル（dc/dq） | `Eigen::AutoDiffScalar` | 変更なし |
| WF Hessian | `autodiff::dual` | `autodiff::dual` で統一 |
| Mesher補間 | テンプレート対応 | `autodiff::dual` で動作確認要 |

---

## 4. 計算コスト比較（改訂版）

設計時の表では、等方性の5ノードに各3個のパラメータ（計15個）を置き、細分する区間数を5として概算した。5ノードだけからなる連続GRIDの区間数は4なので、実際のモデルで見積もる際はその区間数を用いる。

| 手法 | 積分 | Romberg | パラメータ数 | N-pass | 副層数/セグメント | 計算量の目安 |
|------|------|---------|------------|-------|----------------|------------|
| 設計時のWF layer_cake（10層） | 解析 | なし | 30 | 30 | — | 30×10=300 |
| Grid Two-phase（案） | 解析/台形 | Phase1のみ | 15 | 15 | 16 (k=4) | 15×80=1200 |
| Grid Romberg Hessian | **台形** | **Hessian** | 15 | 15×(K+1) | 1,2,4,8(K=3) | 15×75=1125 |
| Grid Romberg Hessian | **台形** | **Hessian** | 15 | 15×(K+1) | 1,2,4,8,16(K=4) | 15×155=2325 |

**注**: k=0,…,K のK+1レベルで、各レベルの15パス×そのレベルの副層数を合計した設計時の概算。K=3で外挿すると
単一の高レベル計算（k=5–6）相当の精度を低コストで得られると設計時には見込んだ。

---

## 5. 結論と推奨実装戦略

### 5.1 推奨アプローチ

**主戦略: Grid Romberg WF Hessian（台形積分 + Richardson外挿）**

```
for k = 0, ..., K:  // K = 2-4
    副層生成（Mesher, 2^k副層/セグメント）
    WF Hessian計算（台形積分, N-pass, ノードベース）→ H_k
Adaptive stopping チェック
H_final = neville_extrapolation_matrix(H_0, ..., H_K)
H_final = symmetrize(H_final)
```

- 台形積分が O(h²) 収束を保証 → Richardson外挿が有効
- 設計時にはK=2–3で目標精度に達すると予想（実測は§6.3）
- Adaptive stopping（H_kのノルム誤差増加で停止）で丸め誤差域を回避

**代替案: Two-phase（シンプル版）**

Romberg外挿をHessianに適用しない。converged_levelのみでWF Hessianを1回計算。
外挿なしのため積分手法（解析/台形）を問わず安定。精度は副層数依存。

### 5.2 実装優先順位

| 優先度 | 作業 | 波動 | 状況 |
|--------|------|------|------|
| 1 | ノードベースWF Hessian（台形, Rayleigh） | Rayleigh | 台形パス既存、接続が必要 |
| 2 | Romberg外挿ループの追加 | Rayleigh | 新規実装（小規模） |
| 3 | Love台形積分の実装 | Love | 新規実装（中規模） |
| 4 | ノードベースWF Hessian（台形, Love） | Love | 3完了後 |
| 5 | Pythonバインディング | 両波動 | 新規実装 |
| 6 | テスト（有限差分 / FD vs Romberg Hessian比較） | 両波動 | 新規 |

### 5.3 勾配（感度カーネル）との対応表（改訂）

| 項目 | 既存Rombergカーネル | Grid Romberg WF Hessian（提案） |
|------|-------------------|-------------------------------|
| Phase 1 | Romberg位相速度収束 | 同一（変更なし） |
| 積分手法 | 伝播行列のみ（積分なし） | **台形積分**（O(h²)維持） |
| AD型 | AutoDiffScalar（1次, 全変数同時） | autodiff::dual（1次, N-pass） |
| ノード→層変換 | Mesher補間（ADグラフ内） | 同一（補間転置でノード勾配） |
| IFTのseed | -dF/dp / dF_dc | g0_node[j] = ∂c/∂q_node_j |
| Richardson外挿 | 位相速度のみ | **Hessian要素に適用** |
| 出力 | dc/dq ベクトル (M×npl) | d²c/dq²行列 (M×npl)² |

---

## 6. 実装結果とレビュー

### 6.1 実装の概要

計画に基づいて以下を実装した（commit `0b37bfe`）:

| ファイル | 変更内容 |
|---------|---------|
| `src/sensitivity_utils.hpp` | `computeLoveEnergyIntegrals_Trap<T>` 追加、`computeVariationalGradientLove_T` に `useAnalytical` フラグ追加 |
| `src/romberg_engine.hpp` | `Mesher::apply_sublayer_grad_to_node()` 追加 |
| `src/grid_hessian.hpp` *(新規)* | `computePhaseVelocityHessianWeakFormRayleigh_Grid` / `Love_Grid` |
| `src/bindings.cpp` | 2つの新規Pythonバインディング |
| 当時のGRID Hessian検証（研究アーカイブ） | 9テスト（精度・対称性・Romberg収束）。当時のスクリプトは未同梱 |

### 6.2 計画からの主な変更点

#### (1) `propagateLoveEigenstates` の拡張は不要と判明

- **計画 §3.2**: 「`propagateLoveEigenstates` に `numSubLayers` 引数を追加し、各層を副層分割する変更が必要」と記載。
- **実際**: Mesherが既に2^k副層を生成するため不要。
  台形積分（`computeLoveEnergyIntegrals_Trap`）は各層の**境界2点**（既存の `propagateLoveEigenstates` が提供）のみで計算できる。
  各「入力層」はMesherが生成した薄い副層なので、境界状態で十分な精度が得られる。

#### (2) 内部計算をVTI（弾性定数）パラメータ化で統一

- **計画**: 等方性パラメータ（Vs, Vp, ρ）で直接N-passを実施することを暗黙的に想定。
- **実際**: 内部計算はすべてVTI（弾性定数 ρ, A, C, F, L, N）で実施し、最後にヤコビアンを用いて等方性に変換する設計を採用。

  **変更の理由**: Mesherの線形補間が弾性定数に対して完全に線形だが、等方性パラメータ（特にVsやVp）に対しては非線形になる。VTI内部計算では補間のチェーンルール `g_node = J^T · g_sub` が**厳密に成立**する。

  **等方性への変換**:
  ```
  H_iso = J_ip^T · H_vti · J_ip + 2次微分補正項
  ```
  `J_ip` の行は各ノードの (ρ, A, C, F, L, N)、列は (Vs, Vp, ρ) で、等方性パラメータからVTI剛性への写像のヤコビアンである。物性写像の2次微分補正は同一ノード内の対角ブロックだけに寄与する。

#### (3) 独立ファイル `src/grid_hessian.hpp` の作成

- **計画**: `sensitivity_utils.hpp` への追記を想定。
- **実際**: 独立した `src/grid_hessian.hpp` に分離。
  依存関係（`disper26.cpp`, `sensitivity_utils.hpp`, `romberg_engine.hpp`）をすべて先にincludeする前提で動作する。

#### (4) ノードベース勾配へのチェーンルール

- **計画 §3.3**: 「方法1: WF層勾配 g_layer を計算し、補間転置でノード勾配に変換」を推奨と記載。
- **実際**: 同じ方針を実装。`Mesher::apply_sublayer_grad_to_node()` として `romberg_engine.hpp` に追加。
  補間重みは定数（パラメータ非依存）なので変換は厳密。

### 6.3 収束精度の実測（改訂版）

当時の中心差分（ε=0.01〜0.05）との比較で報告された相対誤差を示す。

| Romberg Level | 勾配相対誤差 | Hessian相対誤差 |
|:---:|:---:|:---:|
| 3 | 5.8e-6 | 4.6e-5 |
| 4 | 2.8e-7 | 1.8e-5 |
| **5** | **2.1e-9** | **2.2e-7** |
| 6 | 1.4e-10 | 5.9e-8 |

この比較では、勾配の相対誤差が約2e-9に達したのはレベル5だった。
モデルと差分参照の記録、および結果を解釈する際の制約は[収束実験](grid_wf_hessian_convergence.md)に示す。
