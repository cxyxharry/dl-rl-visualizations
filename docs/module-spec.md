# 模块开发规范

本文档定义**所有可视化模块应当遵循的统一协议**，包括：模块接口、生命周期、控制方法、清理要求、接入流程、命名与样式约定。

> 本文档的目标读者：想要新增一个可视化模块、或重构现有模块的开发者。
>
> **注意**：现有 13 个模块尚未完全符合本规范，详见 [module-status.md](module-status.md)。本规范是目标态，配合 Phase 1 架构整顿推进。新增模块请直接按照本规范编写。

## 1. 模块接口

### 1.1 基本形态

每个模块是一个 ES6 class，导出到 `window` 上，供 `js/main.js` 动态加载。

```js
// js/visualizations/my-module.js
class MyModule {
  constructor() {
    // 仅做纯状态初始化，不要 DOM 访问、不要 setTimeout、不要事件绑定
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.timer = null;
    this.speed = 1;

    this.STEPS = [
      { short: '① ...', long: '步骤说明' },
      // ...
    ];
  }

  // --- 生命周期 ---
  init(container)  { /* 见 §2 */ }
  cleanup()        { /* 见 §4 */ }

  // --- 渲染 ---
  render()         { /* 见 §3 */ }

  // --- 控制 ---
  play()           { /* 见 §3 */ }
  pause()          { /* 见 §3 */ }
  stepForward()    { /* 见 §3 */ }
  stepBack()       { /* 见 §3 */ }
  reset()          { /* 见 §3 */ }
  goTo(i)          { /* 见 §3 */ }
  setSpeed(s)      { /* 见 §3 */ }

  // --- 元信息 ---
  getMeta()        { /* 见 §5 */ }
  getState()       { /* 见 §5 */ }
}

window.MyModule = MyModule;
```

### 1.2 方法总表

| 方法 | 必须 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `constructor()` | ✅ | — | — | 纯状态初始化，**无副作用** |
| `init(container)` | ✅ | `HTMLElement` | `this` | 绑定容器、首次计算、首次渲染 |
| `cleanup()` | ✅ | — | — | 清理 timer / 事件 / DOM 引用 |
| `render()` | ✅ | — | — | 将当前状态渲染到 `this.container` |
| `reset()` | ✅ | — | — | 回到初始状态并重绘 |
| `play()` | ✅ | — | — | 开始自动播放（无「步」的模块可空实现） |
| `pause()` | ✅ | — | — | 停止自动播放 |
| `stepForward()` | ✅ | — | — | 前进一步 |
| `stepBack()` | ✅ | — | — | 后退一步 |
| `goTo(i)` | ⭐ 建议 | `number` | — | 跳到第 i 步 |
| `setSpeed(s)` | ⭐ 建议 | `number` | — | 设置播放速度（1.0 = 默认） |
| `getMeta()` | ⭐ 建议 | — | `object` | 返回标题/分类/总步数等元信息 |
| `getState()` | ⭐ 建议 | — | `object` | 返回当前步/播放状态等运行时状态 |

✅ = 必须实现；⭐ = Phase 1 后强烈建议。

## 2. 生命周期

### 2.1 完整生命周期

```
new ModuleClass()
      │
      ▼
  init(container)    ← 由 main.js 调用，传入 #module-content 容器
      │
      ▼
  render()            ← init() 的最后一步应主动调用一次
      │
      ▼
  ┌─ play / pause / stepForward / stepBack / reset / goTo  （任意次）
  │
  ▼
  cleanup()           ← 由 main.js 在切换模块或返回首页前调用
```

### 2.2 `init(container)` 规范

- 保存 `this.container = container`
- 重置 `currentStep`、`isPlaying`、`timer` 等运行时状态
- 执行一次完整计算（矩阵乘法 / 位置编码 / Q-table 初值等）
- 最后调用一次 `this.render()`
- 返回 `this`，以便 `main.js` 缓存实例引用

```js
init(container) {
  this.container = container;
  this.currentStep = 0;
  this.isPlaying = false;
  this.timer = null;
  this._compute();     // 重算派生量
  this.render();
  return this;
}
```

### 2.3 `constructor` 与 `init` 的分工

- **`constructor` 不接触 DOM**：这样同一个类能被多次 `new`，便于测试
- **`init` 才接触 DOM**：`main.js` 会在同一实例上多次调用 `init()` 吗？当前不会，但为了不破坏未来「同页多实例」的可能性，请严格保持这个分工

## 3. 控制方法

### 3.1 `play()` / `pause()`

- 通过 `setTimeout` 或 `setInterval` 驱动自动播放
- **timer 必须挂在 `this.timer` 上**（让 `cleanup()` 可以找到它）
- 播放每一「步」= 调 `stepForward()` + 调 `render()`
- 到达最后一步时自动 `pause()`，不要循环播放
- 没有步进概念的模块（如 `activations` 激活函数对比）可以将 `play()` / `pause()` 实现为空函数，但仍须定义

```js
play() {
  this.isPlaying = true;
  this._auto();
}

pause() {
  this.isPlaying = false;
  clearTimeout(this.timer);
  this.timer = null;
  this.render();
}

_auto() {
  if (!this.isPlaying) return;
  if (this.currentStep >= this.STEPS.length - 1) { this.pause(); return; }
  this.currentStep++;
  this.render();
  this.timer = setTimeout(() => this._auto(), 1200 / this.speed);
}
```

### 3.2 `stepForward()` / `stepBack()` / `goTo(i)`

- 边界裁剪（`Math.max(0, ...)`、`Math.min(STEPS.length - 1, ...)`），不要越界
- 每次都必须调用 `render()`
- **不要在 step 方法里重新做全量计算** —— 计算结果应该在 `init()` 里一次算好缓存在 `this` 上，step 只切换「展示到哪一步」

### 3.3 `reset()`

- 先 `pause()`（清 timer）
- 把所有运行时状态恢复到 `init()` 之后的状态
- 重新 `render()`

### 3.4 `setSpeed(s)`

- `this.speed = Number(s) || 1`
- 不直接改变 timer 间隔，下一次 `_auto` 时自然生效

## 4. cleanup 规范

**这是目前所有现存模块都缺的能力**。新模块必须实现，老模块改造时必须补上。

### 4.1 必须做的事

```js
cleanup() {
  // 1. 停止所有定时器
  if (this.timer) { clearTimeout(this.timer); this.timer = null; }
  this.isPlaying = false;

  // 2. 解绑挂在 document / window 上的事件
  //    （挂在 container 内部的事件会随容器 innerHTML 被替换而自动消失）
  if (this._onKeydown) {
    document.removeEventListener('keydown', this._onKeydown);
    this._onKeydown = null;
  }

  // 3. 断开 D3 / ResizeObserver / MutationObserver 等长期订阅
  if (this._resizeObserver) {
    this._resizeObserver.disconnect();
    this._resizeObserver = null;
  }

  // 4. 释放对 DOM 的强引用
  this.container = null;

  // 5. 清理挂在 window 上的实例指针（如果沿用旧风格的话）
  //    新代码建议改用事件委托，不要再挂 window._xxxInstance
}
```

### 4.2 常见漏网之鱼

| 类型 | 例子 | 清理方式 |
|---|---|---|
| `setTimeout` | `this.timer = setTimeout(...)` | `clearTimeout(this.timer)` |
| `setInterval` | `this._tick = setInterval(...)` | `clearInterval(this._tick)` |
| `requestAnimationFrame` | `this._raf = requestAnimationFrame(...)` | `cancelAnimationFrame(this._raf)` |
| `document.addEventListener` | 键盘、全局点击 | `removeEventListener`，**注意必须是同一个函数引用**，所以要预先 `this._onKey = this._handleKey.bind(this)` |
| D3 `.transition()` | 长时间过渡动画 | 切模块时会自动中断，一般不用管；但要避免过渡回调里访问 `this.container`，清理顺序上先 `this.container = null`，然后在回调里 `if (!this.container) return;` |
| `window._xxxInstance` | 旧风格全局指针 | Phase 1 起不允许再用；如果暂时还在用，`cleanup()` 里 `delete window._xxxInstance` |

### 4.3 自检清单

在 `cleanup()` 返回后，用户切到另一个模块，**应当观察到**：

- 控制台没有报错（尤其是 `Cannot read property 'xxx' of null`）
- 旧模块的 `render()` 不再被触发
- 浏览器 Performance / Memory 面板里老的闭包能被 GC

## 5. 元信息接口

### 5.1 `getMeta()`

返回模块的静态元信息：

```js
getMeta() {
  return {
    id: 'self-attention',            // 与 main.js 中的 moduleName 一致
    title: 'Self-Attention',
    category: 'transformer',         // 'transformer' | 'neural' | 'rl'
    difficulty: 'hard',              // 'easy' | 'medium' | 'hard'
    totalSteps: this.STEPS.length,
    tags: ['Transformer', '矩阵运算'],
    prerequisites: [],               // 先修模块 id
    estimatedMinutes: 10
  };
}
```

### 5.2 `getState()`

返回运行时状态（用于未来的全局控制栏 / 进度同步 / 深链接恢复）：

```js
getState() {
  return {
    currentStep: this.currentStep,
    totalSteps: this.STEPS.length,
    isPlaying: this.isPlaying,
    speed: this.speed
  };
}
```

## 6. 新模块接入流程

### 6.1 文件与命名

1. 在 `js/visualizations/` 新建一个文件，**文件名用 kebab-case**，例如 `sarsa.js`、`rope.js`、`double-dqn.js`
2. 文件名与 `main.js` 中 `getModuleInfo()` 里登记的 `moduleName` 一一对应
3. 导出的 class 名用 PascalCase，末尾统一加 `Visualization` 后缀（Phase 1 后推行）：
   - ✅ `SarsaVisualization`、`RopeVisualization`、`DoubleDqnVisualization`
   - ❌ `Sarsa`、`RoPE`、`DoubleDQN`（这种风格是历史包袱，详见 [module-status.md §3.4](module-status.md)）

### 6.2 代码模板

参照 §1.1 的骨架。

### 6.3 在 `main.js` 注册

在 `js/main.js` 的 `getModuleInfo()` 里加一行：

```js
'sarsa': {
  title: 'SARSA',
  icon: '🧭',
  description: 'On-policy TD 控制',
  className: 'SarsaVisualization'
},
```

### 6.4 在 `index.html` 增加入口

两处都要加：

- 侧边栏：对应 `nav-section` 下加一个 `<li><a href="#sarsa" data-module="sarsa">SARSA</a></li>`
- 首页卡片区：加一个 `<div class="module-card" data-module="sarsa">...</div>`

### 6.5 本地自测清单

- [ ] 首页能看到新卡片，点击能进入模块
- [ ] 侧边栏能点到，URL hash 变为 `#sarsa`
- [ ] 直接访问 `http://localhost:8000/#sarsa` 能正确进入
- [ ] 「播放 → 到最后一步 → 自动暂停」无报错
- [ ] 「播放中切到别的模块」无残留 timer，控制台无报错
- [ ] 「刷新页面后回到同一 hash」能恢复到该模块
- [ ] `Esc` 能返回首页，`Space` `←` `→` `R` 四个快捷键能用
- [ ] 浏览器 DevTools → Elements，确认切模块后旧模块的 DOM 已清空
- [ ] 至少在 1440px 与 768px 两个宽度下能看

## 7. 命名与样式规范

### 7.1 命名

| 场景 | 约定 | 例子 |
|---|---|---|
| 模块 ID / 文件名 / URL hash | kebab-case | `multi-head-attention` |
| 导出类名 | PascalCase + `Visualization` 后缀 | `MultiHeadAttentionVisualization` |
| 内部私有方法 | 下划线前缀 | `_compute()`、`_auto()`、`_softmaxRow()` |
| 步骤数组 | `this.STEPS`，每项 `{ short, long }` | `{ short: '② Q/K/V', long: '三路投影' }` |
| 颜色 | 优先使用 `Utils.COLORS` 与 `css/style.css` 中的 CSS 变量 | `var(--q-color)` |

### 7.2 颜色编码（全局统一）

| 语义 | 颜色 | 取值 | CSS 变量 | JS 常量 |
|---|---|---|---|---|
| Query | 蓝 | `#3B82F6` | `--q-color` | `Utils.COLORS.Q` |
| Key | 紫 | `#8B5CF6` | `--k-color` | `Utils.COLORS.K` |
| Value | 绿 | `#10B981` | `--v-color` | `Utils.COLORS.V` |
| Score / Attention | 橙 | `#F97316` | `--score-color` | `Utils.COLORS.Score` |
| Output | 青 | `#06B6D4` | `--output-color` | `Utils.COLORS.Output` |
| FFN / 隐藏层 | 琥珀 | `#F59E0B` | `--ffn-color` | `Utils.COLORS.FFN` |

新模块**禁止私自引入新颜色代表 Q/K/V/Score**；扩展颜色请先在 `style.css` 的 `:root` 里加变量。

### 7.3 公共数据与工具

- 用 `Utils.GLOBAL_CONFIG` 提供的基础尺寸（`d_model=4, seq_len=3`）作为教学数值例子的默认值，保证跨模块可对照
- 矩阵运算**优先用 `Utils.matrixMultiply` / `Utils.matrixTranspose` / `Utils.softmax` / `Utils.layerNorm`**，不要再自己写 `_matmul`
- 数字显示用 `Utils.formatNumber(x, 2)`，保留 2 位小数
- 公式渲染用 `Utils.renderLatex(latex, el, displayMode)`

### 7.4 样式

- 所有颜色、间距、字体**必须通过 CSS 变量**引用，不要写死十六进制（除非是 heatmap 这种算法生成的色）
- 模块内部样式尽量写在 class 选择器里（`.ffn-step`、`.ffn-matrix`），避免 id 选择器
- 尽量避免在 JS 中拼 `style="..."` 字符串；无法避免时（如 heatmap 色值），控制在最小范围

### 7.5 渲染风格

- 优先用 `this.container.innerHTML = ...` 批量重绘（这是现有模块的一致风格，便于 `cleanup` 时容器清空）
- D3 的 `.select()` / `.selectAll()` 必须在 `this.container` 范围内，**不要 `d3.select('body')` 这种全局选择**
- 事件绑定**尽量用事件委托**：`this.container.addEventListener('click', e => ...)`，不要在每个按钮上写 `onclick="window._xxxInstance.method()"`（这是当前的历史风格，Phase 1 后逐步淘汰）

## 8. 不要做的事

- ❌ 不要在 `constructor` 里碰 DOM
- ❌ 不要在模块里直接操作 `document.title`、`window.location`（这是 `main.js` 的职责）
- ❌ 不要引入新的 CDN 依赖（本项目保持 D3 + KaTeX 两个依赖；如确需新增，先提议再加）
- ❌ 不要在模块里监听 `hashchange` / `popstate`（这是 `main.js` 的职责）
- ❌ 不要写入 `localStorage` / `sessionStorage`（Phase 1 会有统一的 `store` 层）
- ❌ 不要新增 `window.*` 全局变量（旧的 `window._xxxInstance` 属于遗留债）
- ❌ 不要调用其他模块的类或方法（模块之间互不引用；共享逻辑一律放在 `utils.js`）

## 9. 与 `PROJECT_UPGRADE_PLAN.md` 的关系

- 本文档是 Phase 0 产出物之一（模块开发规范）
- Phase 1（架构整顿）会提供更完善的共享组件层和公共工具层，届时本文档会同步更新：
  - 共享控制栏 API
  - 共享矩阵/公式组件
  - 统一的 seeded random
  - 统一的路由与深链接恢复机制
- 在 Phase 1 落地前，新模块至少要做到本文档 §1 - §7 所列要求
