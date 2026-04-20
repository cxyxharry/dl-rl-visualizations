# 模块状态清单

本文档记录当前 13 个模块的真实状态，包括分类、导出类名、生命周期方法覆盖情况、主要问题与改进方向。

状态记录时间点：基于仓库当前代码快照（不含本轮之前未合入的改动）。

## 1. 状态定义

| 状态 | 含义 |
|---|---|
| ✅ 可用 | 能正常加载、交互流畅、无明显 bug、教学内容基本完整 |
| ⚠️ 需补强 | 基本可用，但教学内容、交互一致性或视觉风格需要打磨 |
| 🛠 需重构 | 存在结构性问题（重复逻辑、无法清理、耦合全局状态等），建议在 Phase 1/3 统一处理 |

注：**当前 13 个模块全部没有实现 `cleanup()` 方法**（见 §3 全局问题），因此严格来说没有任何模块完全符合未来的统一协议。本表中的「可用/需补强/需重构」是相对评分，而不是「已达标」。

## 2. 模块总表

| 分类 | 模块 ID | 导出类名 | `init` | `reset` | `play/pause` | `stepFwd/Back` | `cleanup` | `getMeta` | `getState` | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | `attention` | `AttentionVisualization` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| Transformer | `multi-head-attention` | `MultiHeadAttention` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| Transformer | `positional-encoding` | `PositionalEncodingVisualization` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| Transformer | `ffn` | `FFNVisualization` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| Transformer | `layer-norm` | `LayerNormVisualization` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| Transformer | `transformer-encoder` | `TransformerEncoder` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| Transformer | `transformer-decoder` | `TransformerDecoder` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| 神经网络 | `forward-backward` | `ForwardBackward` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| 神经网络 | `activations` | `ActivationsVisualization` | ✅ | ✅ | 空实现 | ✅ | ❌ | ❌ | ❌ | ⚠️ 需补强 |
| 强化学习 | `rl-mdp` | `RLMDPTutorial` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🛠 需重构 |
| 强化学习 | `q-learning` | `QLearning` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🛠 需重构 |
| 强化学习 | `policy-gradient` | `PolicyGradient` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🛠 需重构 |
| 强化学习 | `dqn` | `DQN` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🛠 需重构 |

## 3. 全局性问题（影响所有模块）

这些问题不是单个模块的锅，而是当前架构层的结构性缺口，建议在 Phase 1 统一处理：

### 3.1 没有任何模块实现 `cleanup()`

- `js/main.js` 的 `cleanupModule()` 已经在调用 `moduleInstance.cleanup()`，但**所有模块都没有提供该方法**
- 每个模块内部都有 `this.timer`（来自 `setTimeout` 的自动播放循环）。切换模块时这个 timer 不会被清掉
- 实际表现：如果在自动播放状态下切到另一个模块，原模块的 `render()` 会被定时器继续触发，可能把内容写回已经被替换的容器，或者报错
- 改进方向：每个模块补 `cleanup() { clearTimeout(this.timer); this.isPlaying = false; this.container = null; }`，并移除注册在 `document` 上的事件

### 3.2 每个模块都把实例挂到 `window._xxxInstance`

- 原因：渲染大量使用字符串模板 + `onclick="window._xxxInstance.method()"`
- 这些全局变量：`window._attnInstance`、`window._actInstance`、`window._dqnInstance` 等
- 副作用：
  - 切换模块时旧实例仍挂在 `window` 上，GC 不会回收
  - 和 `AppState.moduleInstance` 重复维护，容易分叉
- 改进方向（Phase 1）：改为在容器上用事件委托（`container.addEventListener('click', ...)`），或通过 `data-action` 属性统一分发

### 3.3 全局控制栏已被禁用

- `index.html` 包含底部控制栏（播放/暂停/单步/重置 + 速度/步骤指示）
- 但 `main.js` 的 `showModuleView()` 强制 `display:none`，注释写明「每个模块内嵌自己的播放/单步/重置控件，底部全局控件保持隐藏以避免重复」
- 实际：全局控制栏相关代码（`play()` / `pause()` / `step()` / `reset()` / `animate()` / `updateStepIndicator()`）都不会被触发
- 改进方向（Phase 1）：按 `PROJECT_UPGRADE_PLAN.md` 方案 A，去掉模块内嵌控制栏，改为全局控制栏统一驱动

### 3.4 导出类命名不一致

| 命名风格 | 模块 |
|---|---|
| `XxxVisualization` | attention、positional-encoding、ffn、layer-norm、activations |
| 只用主体名 | MultiHeadAttention、TransformerEncoder、TransformerDecoder、ForwardBackward、QLearning、PolicyGradient、DQN |
| 带 `Tutorial` 后缀 | RLMDPTutorial |

`main.js` 的 `getModuleInfo()` 通过硬编码 `className` 字段兼容了这种差异，但会让后续贡献者困惑。Phase 1 建议统一。

### 3.5 `main.js` 里调用的 `instance.step(n)` 没有任何模块实现

- `main.js` 的 `step()` / `animate()` 调用 `AppState.moduleInstance.step(AppState.currentStep)`
- 但所有模块都只实现 `stepForward()` / `stepBack()`，没有 `step(n)`
- 由于全局控制栏被隐藏，这段代码实际不会执行 —— 属于「看起来能用但其实死掉」的路径
- 改进方向：Phase 1 统一控制系统时，要么补 `goTo(n)` 统一 API（目前只有 `attention.js` 有），要么删除全局控制栏的 `step()`/`animate()`

### 3.6 `js/utils.js` 的公共工具复用率低

- `utils.js` 提供了 `matrixMultiply` / `matrixTranspose` / `softmax` / `layerNorm` / `generatePositionalEncoding` / `renderLatex` 等
- 但模块内部普遍重新实现了一份私有的 `_matmul` / `_transpose` / `_softmax`（见 `attention.js:65-84` 等）
- 改进方向（Phase 1）：统一通过 `Utils.*` 调用，减少重复

## 4. 分模块详情

### 4.1 Transformer 系列

#### `attention` — Self-Attention
- **状态**：⚠️ 需补强
- **交互亮点**：可点击任一 Score/Attention/Output 单元格查看计算细节；支持 causal mask 切换；支持 `goTo(i)` 跳步（其他模块没有）
- **主要问题**：
  - 8 步流程在小屏幕上信息密度过高
  - 缺少与 Multi-Head 的对比入口
- **改进方向**（Phase 3）：增加缩放因子作用对比；与 MHA 共享同一套 token/权重

#### `multi-head-attention` — Multi-Head Attention
- **状态**：⚠️ 需补强
- **主要问题**：
  - 与 `attention.js` 使用不同的 token 例子和权重，无法直接对照
  - head 数、d_model 等参数硬编码在构造函数中
- **改进方向**（Phase 3）：统一 token/数值；增加 head specialization 切换

#### `positional-encoding` — Positional Encoding
- **状态**：⚠️ 需补强
- **主要问题**：仅覆盖 sinusoidal，未包含 RoPE / ALiBi
- **改进方向**（Phase 3 / Phase 4）：增加 RoPE 与 ALiBi 对照子模块

#### `ffn` — Feed-Forward Network
- **状态**：⚠️ 需补强
- **主要问题**：结构最简单，但未展示「position-wise」的含义
- **改进方向**（Phase 3）：加一个「同一个 FFN 分别作用于每个 token」的动画演示

#### `layer-norm` — Layer Norm & Residuals
- **状态**：⚠️ 需补强
- **主要问题**：未与 BatchNorm 做对照
- **改进方向**（Phase 3）：增加 LN vs BN 对照面板

#### `transformer-encoder` — Transformer Encoder
- **状态**：⚠️ 需补强
- **主要问题**：数据流图与子模块之间没有深链接
- **改进方向**（Phase 3）：点击子块跳转到对应子模块（Attention / FFN / LayerNorm）

#### `transformer-decoder` — Transformer Decoder
- **状态**：⚠️ 需补强
- **主要问题**：
  - Masked MHA 的 mask 可视化不够显眼
  - Cross-Attention 的 K/V 来源解释较弱
- **改进方向**（Phase 3）：凸显 mask 的三角形区域；Cross-Attention 加「K/V 来自 Encoder」的明确箭头

### 4.2 神经网络基础

#### `forward-backward` — 前向/反向传播
- **状态**：⚠️ 需补强
- **主要问题**：参数更新前后对比不够直接
- **改进方向**（Phase 3）：增加「更新前 vs 更新后」的权重对比视图

#### `activations` — 激活函数对比
- **状态**：⚠️ 需补强
- **特殊点**：`play()` 与 `pause()` 是空实现（激活函数本身就是静态函数，没有播放概念）
- **主要问题**：悬停交互 OK，但缺少「梯度消失/爆炸」的直观演示
- **改进方向**（Phase 3）：增加「深层网络中梯度逐层变化」的小动画

### 4.3 强化学习

RL 线 4 个模块都标记为 🛠 需重构，主要原因是：

1. **没有统一的 seeded random**：每次重置产生不同轨迹，无法复现课堂案例
2. **环境设定互不一致**：MDP 用一个 grid，Q-Learning 用另一个，DQN 又换一个
3. **缺少 episode / update / convergence 指标面板**：用户看不到「学得怎么样」

#### `rl-mdp` — MDP 状态图
- **改进方向**：增加 Policy Iteration；统一 RL 线的 grid 世界

#### `q-learning` — Q-Learning
- **改进方向**：增加 SARSA 对照；展示 Q-table 收敛曲线

#### `policy-gradient` — Policy Gradient
- **改进方向**：增加 baseline / advantage 版本；轨迹采样过程的复现

#### `dqn` — DQN
- **改进方向**：增加 Double DQN / Dueling DQN / PER 的概念预告

## 5. 优先级建议

按照 `PROJECT_UPGRADE_PLAN.md` 的阶段划分：

| 优先级 | 工作 | 涉及模块 |
|---|---|---|
| P0（Phase 1） | 补 `cleanup()`、统一控制栏、移除 `window._xxxInstance` | 全部 13 个 |
| P0（Phase 1） | 统一类名命名规范 | 需重命名 8 个 |
| P1（Phase 3） | 补强 Transformer 线一致性 | 7 个 Transformer 模块 |
| P1（Phase 3） | RL 线引入 seeded random + 统一环境 | 4 个 RL 模块 |
| P2（Phase 4） | 新增 SARSA / PE 对照 / 训练优化等新模块 | 新增模块 |

## 6. 本轮（Phase 0）不做什么

为了避免「顺手重构」带来的隐性破坏，以下内容**本轮明确不做**：

- 不改任何模块的业务逻辑、渲染代码、数值或动画
- 不补 `cleanup()`、不统一类名、不动 `window._xxxInstance`
- 不抽离公共组件、不改 `utils.js`
- 仅修正文档以对齐当前代码实际状态

上述工作统一留到 Phase 1 推进。
