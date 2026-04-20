# 数学符号和维度约定

本文档定义了项目中使用的数学符号、维度约定和公式表示法。

## 全局维度约定

| 符号 | 含义 | 默认值 |
|------|------|--------|
| `d_model` | 模型维度（embedding 维度） | 4 |
| `seq_len` | 序列长度（token 数量） | 3 |
| `batch_size` | 批次大小 | 1 |
| `num_heads` | 注意力头数量 | 2 |
| `d_k` | Key 维度（每个 head） | 2 |
| `d_v` | Value 维度（每个 head） | 2 |
| `d_ff` | FFN 隐藏层维度 | 16 |

## Transformer 相关符号

### Self-Attention

| 符号 | 维度 | 含义 |
|------|------|------|
| `X` | `(seq_len, d_model)` | 输入序列 |
| `Q` | `(seq_len, d_k)` | Query 矩阵 |
| `K` | `(seq_len, d_k)` | Key 矩阵 |
| `V` | `(seq_len, d_v)` | Value 矩阵 |
| `W_Q` | `(d_model, d_k)` | Query 权重矩阵 |
| `W_K` | `(d_model, d_k)` | Key 权重矩阵 |
| `W_V` | `(d_model, d_v)` | Value 权重矩阵 |
| `Score` | `(seq_len, seq_len)` | 注意力分数矩阵 |
| `Attention` | `(seq_len, seq_len)` | 注意力权重矩阵（Softmax 后） |
| `Output` | `(seq_len, d_v)` | 输出矩阵 |

**核心公式：**

```
Q = X · W_Q
K = X · W_K
V = X · W_V

Score = Q · K^T / √d_k

Attention = Softmax(Score)

Output = Attention · V
```

### Multi-Head Attention

| 符号 | 维度 | 含义 |
|------|------|------|
| `head_i` | `(seq_len, d_v)` | 第 i 个注意力头的输出 |
| `MultiHead` | `(seq_len, num_heads × d_v)` | 所有头的拼接 |
| `W_O` | `(num_heads × d_v, d_model)` | 输出投影矩阵 |
| `Output` | `(seq_len, d_model)` | 最终输出 |

**核心公式：**

```
head_i = Attention(Q_i, K_i, V_i)

MultiHead = Concat(head_1, head_2, ..., head_h)

Output = MultiHead · W_O
```

### Positional Encoding

| 符号 | 维度 | 含义 |
|------|------|------|
| `PE` | `(seq_len, d_model)` | 位置编码矩阵 |
| `pos` | 标量 | 位置索引（0 到 seq_len-1） |
| `i` | 标量 | 维度索引（0 到 d_model-1） |

**核心公式：**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))

PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Feed-Forward Network

| 符号 | 维度 | 含义 |
|------|------|------|
| `X` | `(seq_len, d_model)` | 输入 |
| `W_1` | `(d_model, d_ff)` | 第一层权重 |
| `b_1` | `(d_ff,)` | 第一层偏置 |
| `W_2` | `(d_ff, d_model)` | 第二层权重 |
| `b_2` | `(d_model,)` | 第二层偏置 |
| `Output` | `(seq_len, d_model)` | 输出 |

**核心公式：**

```
FFN(X) = ReLU(X · W_1 + b_1) · W_2 + b_2
```

### Layer Normalization

| 符号 | 维度 | 含义 |
|------|------|------|
| `x` | `(d_model,)` | 输入向量 |
| `μ` | 标量 | 均值 |
| `σ` | 标量 | 标准差 |
| `γ` | `(d_model,)` | 缩放参数 |
| `β` | `(d_model,)` | 偏移参数 |
| `ε` | 标量 | 防止除零的小常数（1e-5） |

**核心公式：**

```
μ = (1/d_model) · Σ x_i

σ² = (1/d_model) · Σ (x_i - μ)²

LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

## 神经网络基础符号

### 前向传播

| 符号 | 维度 | 含义 |
|------|------|------|
| `x` | `(n_in,)` | 输入向量 |
| `W` | `(n_in, n_out)` | 权重矩阵 |
| `b` | `(n_out,)` | 偏置向量 |
| `z` | `(n_out,)` | 线性输出（激活前） |
| `a` | `(n_out,)` | 激活输出 |

**核心公式：**

```
z = W^T · x + b

a = σ(z)  # σ 为激活函数
```

### 反向传播

| 符号 | 维度 | 含义 |
|------|------|------|
| `L` | 标量 | 损失函数 |
| `∂L/∂W` | `(n_in, n_out)` | 权重梯度 |
| `∂L/∂b` | `(n_out,)` | 偏置梯度 |
| `∂L/∂a` | `(n_out,)` | 激活输出梯度 |
| `∂L/∂z` | `(n_out,)` | 线性输出梯度 |

**核心公式（链式法则）：**

```
∂L/∂z = ∂L/∂a · ∂a/∂z

∂L/∂W = x · (∂L/∂z)^T

∂L/∂b = ∂L/∂z

∂L/∂x = W · ∂L/∂z
```

### 激活函数

| 函数 | 公式 | 导数 |
|------|------|------|
| ReLU | `max(0, x)` | `x > 0 ? 1 : 0` |
| Sigmoid | `1 / (1 + e^(-x))` | `σ(x) · (1 - σ(x))` |
| Tanh | `(e^x - e^(-x)) / (e^x + e^(-x))` | `1 - tanh²(x)` |

## 强化学习符号

### MDP (Markov Decision Process)

| 符号 | 含义 |
|------|------|
| `S` | 状态空间 |
| `A` | 动作空间 |
| `s_t` | 时刻 t 的状态 |
| `a_t` | 时刻 t 的动作 |
| `r_t` | 时刻 t 的奖励 |
| `P(s'|s,a)` | 状态转移概率 |
| `R(s,a)` | 奖励函数 |
| `γ` | 折扣因子（0.9） |

### Q-Learning

| 符号 | 维度 | 含义 |
|------|------|------|
| `Q(s,a)` | 标量 | 状态-动作价值函数 |
| `Q_table` | `(|S|, |A|)` | Q 值表 |
| `α` | 标量 | 学习率（0.1） |
| `γ` | 标量 | 折扣因子（0.9） |
| `ε` | 标量 | 探索率 |

**核心公式（Q-Learning 更新）：**

```
Q(s_t, a_t) ← Q(s_t, a_t) + α · [r_t + γ · max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
```

### Policy Gradient

| 符号 | 含义 |
|------|------|
| `π(a|s)` | 策略函数（给定状态 s 选择动作 a 的概率） |
| `θ` | 策略网络参数 |
| `J(θ)` | 目标函数（期望回报） |
| `∇_θ J(θ)` | 策略梯度 |
| `G_t` | 从时刻 t 开始的累积回报 |

**核心公式（REINFORCE）：**

```
G_t = Σ_{k=0}^{T-t} γ^k · r_{t+k}

∇_θ J(θ) = E[Σ_t ∇_θ log π(a_t|s_t) · G_t]
```

### DQN (Deep Q-Network)

| 符号 | 含义 |
|------|------|
| `Q(s,a;θ)` | 用神经网络参数化的 Q 函数 |
| `θ` | 在线网络参数 |
| `θ^-` | 目标网络参数 |
| `D` | 经验回放缓冲区 |
| `(s,a,r,s')` | 经验元组 |

**核心公式（DQN 损失）：**

```
y = r + γ · max_a' Q(s', a'; θ^-)

Loss = E[(y - Q(s, a; θ))²]
```

## 颜色编码约定

| 元素 | 颜色代码 | 用途 |
|------|----------|------|
| Query (Q) | `#3B82F6` | 蓝色系 |
| Key (K) | `#8B5CF6` | 紫色系 |
| Value (V) | `#10B981` | 绿色系 |
| Score/Attention | `#F97316` | 橙红色系 |
| Output | `#06B6D4` | 青色系 |
| FFN/隐藏层 | `#F59E0B` | 琥珀色系 |
| 正值 | `#10B981` | 亮绿色 |
| 负值 | `#EF4444` | 红色 |
| 零值 | `#374151` | 深灰色 |

## 矩阵维度标注规范

在可视化中，矩阵维度标注格式为：`(行数 × 列数)`

示例：
- `X (3 × 4)` - 3 个 token，每个 4 维
- `W_Q (4 × 2)` - 权重矩阵，输入 4 维，输出 2 维
- `Q (3 × 2)` - Query 矩阵，3 个 token，每个 2 维

## 数值格式约定

- 小数保留 2-3 位：`0.123`
- 科学计数法用于极小值：`1.23e-5`
- 百分比用于概率：`45.6%`
- 矩阵元素对齐显示

## LaTeX 渲染示例

### Attention 公式

```latex
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Multi-Head Attention 公式

```latex
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

### Q-Learning 更新公式

```latex
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)\right]
```

### Policy Gradient 公式

```latex
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi(a_t|s_t) G_t\right]
```

---

**注意事项：**

1. 所有维度标注必须与实际计算一致
2. 矩阵乘法前需检查维度兼容性
3. 颜色编码在所有模块中保持一致
4. 公式渲染使用 KaTeX，确保语法正确
5. 数值例子使用 `GLOBAL_CONFIG` 中定义的固定值
