# R1-LY-Complementary 定点混沌生成器模块文档

> **项目**: 高维混沌生成器在轻量级密码学中的应用  
> **文件**: `R1_LY_Complementary.py`  
> **状态**: 进行中（核心算法实现完成，密码学集成待扩展）  
> **版本**: 0.9.0-draft

---

## 1. 模块概述

### 1.1 定位与目标

本模块是高维混沌密码系统的**核心熵源层**，负责：

1. **参数合成**：基于 R1-LY 互补奇偶结构，生成满足锥场条件的整数参数集
2. **定点实现**：在 $Z_{2^{16}}$ 环上实现严格定点运算，适配无 FPU 的 16 位 MCU
3. **动力学增强**：通过位级循环移位（Bit-Rotation）缓解有限精度退化
4. **可验证性**：提供 Lyapunov 指数谱计算与混沌性判定工具

### 1.2 模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                  R1_LY_Complementary 模块架构                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐                │
│  │ 参数合成层        │───→│ 映射实现层        │                │
│  │ Complementary    │    │ HighDimChaosMap  │                │
│  │ ChaosParams      │    │ (基础定点映射)    │                │
│  └──────────────────┘    └────────┬─────────┘                │
│                                     │                         │
│                                     ↓                         │
│                          ┌──────────────────┐              │
│                          │ 动力学增强层        │              │
│                          │ HighDimChaosMap    │              │
│                          │ Enhanced (位旋转)  │              │
│                          └────────┬─────────┘              │
│                                   │                         │
│                                   ↓                         │
│                          ┌──────────────────┐              │
│                          │ 分析与验证层      │              │
│                          │ • Lyapunov 谱计算  │              │
│                          │ • 混沌性判定        │              │
│                          │ • NIST 序列生成    │              │
│                          └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心类详解

### 2.1 ComplementaryChaosParams —— 参数合成器

**职责**: 根据混沌理论约束合成整数参数集。

#### 构造函数

```python
class ComplementaryChaosParams:
    def __init__(
        self, 
        n: int,           # 系统维度，≥3
        A_odd: int,       # 奇数基数，奇数且≥5
        B_even: int,      # 偶数基数，偶数且>2(n-1)
        eta: float = 0.3,  # 反馈强度系数
        kappa: float = 0.8,
        tau: float = 0.5   # 无包裹预算阈值
    )
```

#### 关键约束

| 参数 | 数学约束 | 工程意义 |
|------|---------|---------|
| `n` | $n \geq 3$ | 最小可实现混沌的维度 |
| `A_odd` | 奇数, $\geq 5$ | 控制扩张因子序列 $a_{j1}$ |
| `B_even` | 偶数, $> 2(n-1)$ | 控制耦合因子序列 $a_{jn}$ |
| `tau` | $(0,1)$ | 防止定点溢出，确保 $|a_{j1}\Delta_{n1} + a_{jn}\Delta_{s0}| < 1$ |

#### 合成方法

```python
def synthesize(self, max_iter: int = 100) -> Dict
```

**返回字典结构**:

```python
{
    'a_j1': np.ndarray[int64],   # 奇数递增序列 [A_odd, A_odd+2, ...]
    'a_jn': np.ndarray[int64],   # 偶数递减序列 [B_even, B_even-2, ...]
    'b': int,                    # 线性反馈系数 $b = \max(2, \lfloor \eta A_{min} \rfloor)$
    'c': float,                  # 辅助量 $c = \sqrt{b^2 + 32}$
    'A_min': int,                # 最小扩张因子 $\min(a_{j1})$
    'beta': float,               # 耦合强度 $\beta = \|a_{jn}\|_2$
    'discriminant': float        # 锥场判别式 $\Delta = (A_{min}-1)^2 - 4\beta c$
}
```

**锥场条件**（确保混沌性）:

$$
\Delta = (A_{min} - 1)^2 - 4\beta c > 0
$$

若不满足，类自动调整 `A_odd`、`eta` 或 `B_even` 自适应重试。

---

### 2.2 HighDimChaosMap —— 基础定点映射

**职责**: 实现 R1-LY 混沌映射的严格定点迭代。

#### 构造函数

```python
class HighDimChaosMap:
    def __init__(
        self, 
        params: Dict,           # 合成器输出的参数字典
        word_size: int = 16,    # 位宽（默认 16-bit）
        seed: int = None        # 随机种子（可选）
    )
```

#### 内部状态结构

状态向量维度为 $2n-2$，分为两部分：

- **x 部分** (`x_idx`): 索引 $[0, n-2]$，对应物理状态 $x_1, ..., x_{n-1}$
- **s 部分** (`s_idx`): 索引 $[n-1, 2n-3]$，对应延迟链 $s_0, ..., s_{n-2}$（其中 $s_0 = x_n$）

#### 核心迭代方程

**数学形式**（定点版本）:

$$
\begin{aligned}
x_n^{(t+1)} &= \left[ b \cdot x_{n-2}^{(t)} + \mathcal{L}(x_{n-1}^{(t)}, s_0^{(t)}) \right] \bmod 2^w \\
x_j^{(t+1)} &= \left[ a_{j1} \cdot x_j^{(t)} + a_{jn} \cdot s_{n-1-j}^{(t)} \right] \bmod 2^w \\
s_j^{(t+1)} &= s_{j-1}^{(t)} \quad (\text{延迟移位})
\end{aligned}
$$

其中 $\mathcal{L}(\cdot)$ 为定点 Logistic 核心：

```python
def _logistic_core(self, x_nm1: int, s0: int) -> int:
    mask = self.mod - 1          # 0xFFFF for 16-bit

    # 合并输入
    z = (x_nm1 + s0) & mask      # z ∈ [0, 2^w)

    # 定点近似: (~z) ≈ (2^w - 1 - z) ≈ (2^w - z)
    inv_z = (~z) & mask          # "1-z" 的定点表示

    # 计算 z*(1-z)，使用 32/64 位中间寄存器
    temp = z * inv_z

    # 等效 4*z*(1-z)/2^w，纯右移无除法
    result = temp >> (self.word_size - 2)

    return result & mask
```

#### 主要接口方法

```python
def iterate(self, steps: int = 1) -> np.ndarray
"""
执行多步迭代。

Args:
    steps: 迭代步数

Returns:
    轨迹数组，shape (steps, n)，每行为 [x_1, ..., x_n]
"""

def next(self) -> np.ndarray
"""单步迭代，返回 shape (n,) 的物理状态。"""

def set_state_from_physical(self, physical_state: np.ndarray) -> None
"""
从物理状态 [x_1, ..., x_n] 设置完整内部状态。
用于敏感性分析和可重复实验。

Args:
    physical_state: shape (n,) 的整数数组
"""

def get_physical_state(self) -> np.ndarray
"""获取当前物理状态 [x_1, ..., x_n]。"""

def reset(self, seed: int = None) -> None
"""重置状态，可指定新随机种子。"""
```

---

### 2.3 HighDimChaosMapEnhanced —— 位级动力学增强

**职责**: 通过跨维度位级循环移位增强混沌行为，继承 `HighDimChaosMap`。

#### 增强机制

在每次基础迭代后，执行以下位级操作：

1. **打包 (Pack)**: 将 n 个 w-bit 坐标拼接为 nw-bit 大整数
   ```
   Z = x_0 | (x_1 << w) | (x_2 << 2w) | ... | (x_{n-1} << (n-1)w)
   ```

2. **循环移位 (ROL)**: 在 nw-bit 宽度上做左循环移位 $k_t$ 位
   ```
   Z' = ((Z << k) | (Z >> (nw - k))) & ((1 << nw) - 1)
   ```

3. **解包 (Unpack)**: 将 $Z'$ 切分回 n 个 w-bit 坐标

4. **步长更新**: $k_{t+1} = 1 + (k_t \mod (w-1))$，确保周期性变化

#### 数学效果

- 打破维度间独立性，引入跨维度耦合
- 延缓有限精度导致的轨道退化
- 增加有效熵率，适用于密码学应用

---

## 3. 分析与验证工具

### 3.1 Lyapunov 指数计算

```python
def compute_lyapunov_exponents(
    map_instance: HighDimChaosMap,
    steps: int = 2000,          # 总迭代步数
    transient: int = 500,       # 暂态消除步数
    ortho_interval: int = 5     # QR 正交化间隔
) -> np.ndarray  # shape (n,)
```

**算法**: QR 分解法

1. 消除暂态过程（前 `transient` 步不计入统计）
2. 在归一化空间 $[0,1)^n$ 计算 Jacobian $J(x)$
3. 演化切向量基 $Q = J \cdot Q$
4. 每 `ortho_interval` 步执行 QR 分解，记录对角线拉伸因子 $R_{ii}$
5. 计算 LE 谱：$\lambda_i = \frac{1}{\tau} \langle \ln R_{ii} \rangle$

**Jacobian 结构**:

```python
def _jacobian_normalized(x: np.ndarray, params: Dict, n: int) -> np.ndarray:
    J = np.zeros((n, n), dtype=np.float64)

    # 前 n-1 行: 扩张因子 (对角) + 耦合因子 (末列)
    for j in range(n - 1):
        J[j, j] = params['a_j1'][j]      # 扩张
        J[j, -1] = params['a_jn'][j]     # 耦合到 x_n

    # 最后一行: Logistic 核心导数
    z = x[-2] + x[-1]  # 归一化空间
    if 0 < z < 1:
        d_logistic = 4.0 * (1.0 - 2.0 * z)
    else:
        d_logistic = 0.0

    J[-1, -2] = params['b']      # 线性反馈
    J[-1, -1] = d_logistic       # Logistic 导数

    return J
```

### 3.2 混沌性验证

```python
def verify_chaos(map_instance: HighDimChaosMap, steps: int = 2000) -> Dict
```

**判定标准**:

1. **正 Lyapunov 指数**: 所有 $\lambda_i > 0.01$
2. **状态方差**: 归一化状态方差 $> 0.05$（非平凡吸引子）
3. **自相关衰减**: $x_n$ 的自相关函数快速衰减至 0

**返回报告结构**:

```python
{
    'params': map_instance.params,      # 参数备份
    'lyap_exps': np.ndarray,            # LE 谱
    'is_chaotic': bool,                 # 综合判定结果
    'variance': np.ndarray,             # 各维度方差
    'min_le': float,                    # 最小 LE
    'max_le': float                     # 最大 LE
}
```

### 3.3 NIST 测试序列生成

```python
def generate_binary_sequence(
    map_instance: HighDimChaosMap,
    length: int = 1_000_000
) -> bytes
```

**生成流程**:

1. 迭代生成 $\lceil length/16 \rceil + 1000$ 个样本（跳过前 1000 步暂态）
2. 提取每个 $x_n$ 的最低 16 位
3. 打包为字节流，可直接用于 NIST SP 800-22 统计测试套件

---

## 4. 使用示例

### 示例 1: 完整验证流程

```python
from R1_LY_Complementary import (
    ComplementaryChaosParams, 
    HighDimChaosMap, 
    verify_chaos,
    generate_binary_sequence,
    run_complete_verification
)

# 1. 参数合成 (n=4, 16-bit)
synth = ComplementaryChaosParams(n=4, A_odd=371, B_even=50, eta=0.3)
params = synth.synthesize()

print(f"参数合成成功: a_j1={params['a_j1']}, a_jn={params['a_jn']}")
print(f"锥场判别式: {params['discriminant']:.3f} (>0 表示满足条件)")

# 2. 初始化混沌映射
chaos_map = HighDimChaosMap(params, word_size=16, seed=42)

# 3. 混沌性验证
result = verify_chaos(chaos_map, steps=3000)

print(f"Lyapunov 指数谱: {result['lyap_exps']}")
print(f"最小 LE: {result['min_le']:.6f}")
print(f"混沌判定: {'通过' if result['is_chaotic'] else '未通过'}")

# 4. 生成 NIST 测试序列（若通过混沌判定）
if result["is_chaotic"]:
    test_map = HighDimChaosMap(result["params"], word_size=16, seed=123)
    binary_seq = generate_binary_sequence(test_map, length=1_000_000)

    with open("nist_test_sequence.bin", "wb") as f:
        f.write(binary_seq)
    print("NIST 测试序列已保存至 nist_test_sequence.bin")
```

### 示例 2: 使用增强版映射

```python
from R1_LY_Complementary import HighDimChaosMapEnhanced

# 使用位级动力学增强版本
enhanced_map = HighDimChaosMapEnhanced(params, word_size=16, seed=42)

# 生成更长轨迹用于密码学测试
trajectory = enhanced_map.iterate(steps=10000)

# trajectory shape: (10000, 4)
# 可用于后续 S-box 生成、密钥流生成等密码学应用
```

### 示例 3: 可重复敏感性分析

```python
# 设置特定初始条件进行敏感性分析
initial_state = np.array([12345, 6789, 11111, 22222], dtype=np.int64)

map1 = HighDimChaosMap(params, word_size=16)
map1.set_state_from_physical(initial_state)

traj1 = map1.iterate(100)

# 微小扰动
perturbed = initial_state.copy()
perturbed[0] += 1  # 仅改变 1 LSB

map2 = HighDimChaosMap(params, word_size=16)
map2.set_state_from_physical(perturbed)

traj2 = map2.iterate(100)

# 计算轨迹分歧
divergence = np.abs(traj1.astype(float) - traj2.astype(float)) / (1 << 16)
```

---

## 5. 实验输出









---

## 6. 项目进度状态

### 6.1 已实现功能 ✅

- [x] 互补奇偶参数合成与锥场条件验证
- [x] 严格定点混沌映射（无除法，纯位运算）
- [x] 位级动力学增强（Bit-Rotation 跨维度耦合）
- [x] Lyapunov 指数谱计算（QR 分解法）
- [x] 混沌性自动判定（LE + 方差 + 自相关）
- [x] NIST 测试序列生成
- [x] 实验可视化输出（EPS/PDF）

### 6.2 进行中 / 待扩展 🚧

根据项目整体进度（README 描述），以下部分仍在开发：

- [ ] **GIFT-64 集成接口**: 将混沌生成器嵌入块密码轮函数
- [ ] **16 位 MCU 移植**: MSP430/STM32L0 的 C 语言实现
- [ ] **FPGA 硬件验证**: Verilog 位精确对齐（与 MATLAB/Simulink 对比）

---

## 7. 依赖与兼容性

### 7.1 Python 依赖

```
numpy >= 1.20      # 数组运算与随机数生成
matplotlib >= 3.3  # 可视化输出（可选，仅当需要图表时）
```

### 7.2 运行环境

- **开发/仿真**: Python 3.8+，64 位系统
- **目标部署**: 16 位 MCU（MSP430, STM32L0 系列）
- **FPGA 验证**: MATLAB/Simulink + HDL Coder 工作流程

### 7.3 硬件移植注意事项

当前实现使用 `np.int64` 中间变量防止 Python 整数溢出。移植到 C/Verilog 时：

1. **乘法位宽**: $a_{j1} \cdot x_j$ 需要 $2w$ 位中间寄存器（16-bit → 32-bit）
2. **Logistic 核心**: `temp >> (w-2)` 操作在 Verilog 中为组合逻辑
3. **延迟链**: 使用 FIFO 或移位寄存器实现 $s_j$ 状态更新
4. **位旋转**: 在 $nw$ 位宽上进行，需要 n 级联寄存器或 barrel shifter

---

## 8. API 速查表

### 8.1 类初始化

| 类 | 必需参数 | 可选参数 | 默认值 |
|-----|---------|---------|--------|
| `ComplementaryChaosParams` | `n`, `A_odd`, `B_even` | `eta`, `kappa`, `tau` | 0.3, 0.8, 0.5 |
| `HighDimChaosMap` | `params` | `word_size`, `seed` | 16, None |
| `HighDimChaosMapEnhanced` | `params` | `word_size`, `seed` | 16, None |

### 8.2 常用方法

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `synthesize()` | - | `Dict` | 生成参数 |
| `iterate(steps)` | `int` | `np.ndarray (steps, n)` | 生成轨迹 |
| `next()` | - | `np.ndarray (n,)` | 单步迭代 |
| `set_state_from_physical()` | `np.ndarray (n,)` | - | 设置初始条件 |
| `reset()` | `int` (可选) | - | 重置状态 |
| `compute_lyapunov_exponents()` | `HighDimChaosMap` | `np.ndarray (n,)` | LE 谱 |
| `verify_chaos()` | `HighDimChaosMap` | `Dict` | 综合验证 |
| `generate_binary_sequence()` | `HighDimChaosMap`, `int` | `bytes` | NIST 序列 |

---

## 9. 参考文献与理论依据

1. **R1-LY 结构**: 基于时间延迟混沌映射的 n 维扩展，引入互补奇偶参数对 $(a_{j1}, a_{jn})$
2. **定点运算**: $Z_{2^w}$ 环上的模运算，确保无 FPU 硬件友好性
3. **混沌验证**: Oseledets 乘法遍历定理，通过 QR 分解计算 Lyapunov 指数谱
4. **有限精度处理**: 位级循环移位技术，缓解数字化导致的熵损失

---

*文档生成时间: 2026-03-18*  
*对应代码版本: R1_LY_Complementary.py (开发中)*