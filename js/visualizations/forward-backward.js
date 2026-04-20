// 神经网络前向 / 反向传播可视化
// 架构: Input(2) → Linear(W1,b1) → ReLU → Linear(W2,b2) → Softmax → Cross-Entropy
// 展示: 5 步前向 + 6 步反向 + 梯度下降更新
class ForwardBackward {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;
    this.lr = 0.1; // 学习率

    // 固定输入 / 标签
    this.x = [1.0, 0.5];                 // shape (2,)
    this.y_true = [0, 1];                // one-hot, 正确类别 = 1
    this.correctClass = 1;

    // 第一层: W1 (2×3), b1 (3,)
    this.W1 = [
      [0.5, -0.3, 0.8],
      [0.2,  0.7, -0.4]
    ];
    this.b1 = [0.1, 0.0, -0.1];

    // 第二层: W2 (3×2), b2 (2,)
    this.W2 = [
      [0.3, -0.5],
      [0.8,  0.2],
      [-0.4, 0.6]
    ];
    this.b2 = [0.0, 0.0];

    // 步骤元数据: 5 forward + 6 backward = 11 + 1 final update view
    this.STEPS = [
      { id: 0,  dir: 'F', title: '输入 x',                     short: 'x',             formula: 'x ∈ ℝ²' },
      { id: 1,  dir: 'F', title: 'z₁ = x·W₁ + b₁',             short: 'z₁',            formula: 'z₁ = xW₁ + b₁' },
      { id: 2,  dir: 'F', title: 'h = ReLU(z₁)',               short: 'h',             formula: 'h = max(0, z₁)' },
      { id: 3,  dir: 'F', title: 'z₂ = h·W₂ + b₂',             short: 'z₂',            formula: 'z₂ = hW₂ + b₂' },
      { id: 4,  dir: 'F', title: 'ŷ = softmax(z₂),  L = CE',   short: 'ŷ, L',          formula: 'ŷ = softmax(z₂),  L = -log ŷ[y]' },
      { id: 5,  dir: 'B', title: '∂L/∂z₂ = ŷ - y',             short: '∂L/∂z₂',        formula: '∂L/∂z₂ = ŷ - y' },
      { id: 6,  dir: 'B', title: '∂L/∂W₂ = hᵀ · ∂L/∂z₂',       short: '∂L/∂W₂',        formula: '∂L/∂W₂ = hᵀ · ∂L/∂z₂' },
      { id: 7,  dir: 'B', title: '∂L/∂b₂ = ∂L/∂z₂',            short: '∂L/∂b₂',        formula: '∂L/∂b₂ = ∂L/∂z₂' },
      { id: 8,  dir: 'B', title: '∂L/∂h = ∂L/∂z₂ · W₂ᵀ',       short: '∂L/∂h',         formula: '∂L/∂h = ∂L/∂z₂ · W₂ᵀ' },
      { id: 9,  dir: 'B', title: '∂L/∂z₁ = ∂L/∂h ⊙ ReLU′(z₁)', short: '∂L/∂z₁',        formula: "∂L/∂z₁ = ∂L/∂h ⊙ ReLU'(z₁)" },
      { id: 10, dir: 'B', title: '∂L/∂W₁ = xᵀ · ∂L/∂z₁,  ∂L/∂b₁ = ∂L/∂z₁', short: '∂L/∂W₁,b₁', formula: '∂L/∂W₁ = xᵀ · ∂L/∂z₁' },
      { id: 11, dir: 'U', title: '梯度下降更新：W ← W - η · ∂L/∂W', short: 'update', formula: 'θ ← θ - η · ∂L/∂θ' }
    ];
  }

  // ============================================================
  // 生命周期 + 控制
  // ============================================================
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this._computeAll();
    this.render();
    return this;
  }
  reset()        { this.currentStep = 0; this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  play()         { this.isPlaying = true; this._auto(); this.render(); }
  pause()        { this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  setSpeed(s)    { this.speed = Number(s) || 1; }
  stepForward()  { if (this.currentStep < this.STEPS.length - 1) { this.currentStep++; this.render(); } }
  stepBack()     { if (this.currentStep > 0) { this.currentStep--; this.render(); } }
  goTo(i)        { this.currentStep = Math.max(0, Math.min(i, this.STEPS.length - 1)); this.render(); }
  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this.render();
      this.timer = setTimeout(() => this._auto(), 1400 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  // ============================================================
  // 数值计算（前向 + 反向 + 更新后参数）
  // ============================================================
  _computeAll() {
    const x = this.x, W1 = this.W1, b1 = this.b1, W2 = this.W2, b2 = this.b2, y = this.y_true;

    // 前向
    this.z1 = [0, 1, 2].map(j => x[0] * W1[0][j] + x[1] * W1[1][j] + b1[j]);
    this.h  = this.z1.map(v => Math.max(0, v));
    this.reluMask = this.z1.map(v => (v > 0 ? 1 : 0));
    this.z2 = [0, 1].map(j => this.h[0] * W2[0][j] + this.h[1] * W2[1][j] + this.h[2] * W2[2][j] + b2[j]);

    // softmax（数值稳定）
    const m = Math.max(...this.z2);
    const ez = this.z2.map(v => Math.exp(v - m));
    const S = ez.reduce((a, b) => a + b, 0);
    this.y_pred = ez.map(v => v / S);
    this.loss = -Math.log(this.y_pred[this.correctClass] + 1e-12);

    // 反向
    this.dL_dz2 = this.y_pred.map((v, i) => v - y[i]);                                    // (2,)
    this.dL_dW2 = [0, 1, 2].map(i => [0, 1].map(j => this.h[i] * this.dL_dz2[j]));        // (3×2)
    this.dL_db2 = this.dL_dz2.slice();                                                    // (2,)
    this.dL_dh  = [0, 1, 2].map(i => W2[i][0] * this.dL_dz2[0] + W2[i][1] * this.dL_dz2[1]); // (3,)
    this.dL_dz1 = this.dL_dh.map((v, i) => v * this.reluMask[i]);                         // (3,)
    this.dL_dW1 = [0, 1].map(i => [0, 1, 2].map(j => x[i] * this.dL_dz1[j]));             // (2×3)
    this.dL_db1 = this.dL_dz1.slice();                                                    // (3,)

    // 更新后的参数
    this.W1_new = this.W1.map((row, i) => row.map((v, j) => v - this.lr * this.dL_dW1[i][j]));
    this.b1_new = this.b1.map((v, i) => v - this.lr * this.dL_db1[i]);
    this.W2_new = this.W2.map((row, i) => row.map((v, j) => v - this.lr * this.dL_dW2[i][j]));
    this.b2_new = this.b2.map((v, i) => v - this.lr * this.dL_db2[i]);
  }

  // ============================================================
  // 渲染辅助
  // ============================================================
  // 定宽数字：正数前补一个空格，用于计算图里的列对齐（共享：见 js/utils/format.js）
  _fmt(v, d = 3) { return window.Utils.fmtSigned(v, d); }

  // 单个彩色数值 cell（背景强度随绝对值）
  _cell(v, color, opts = {}) {
    const d = opts.d ?? 3;
    const minw = opts.minw ?? 58;
    const hl = opts.hl === true;
    const intensity = Math.min(Math.abs(v) * 0.35, 0.55);
    const bg = v === 0 ? '#111'
      : (v > 0 ? 'rgba(16,185,129,' + (0.12 + intensity) + ')'
               : 'rgba(239,68,68,'  + (0.12 + intensity) + ')');
    const border = hl ? '2px solid #F97316' : '1px solid ' + color;
    return '<div style="min-width:' + minw + 'px;padding:6px 8px;margin:2px;background:' + bg +
           ';border:' + border + ';border-radius:4px;color:' + color +
           ';font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:center">' +
           Number(v).toFixed(d) + '</div>';
  }

  _hidden(minw = 58) {
    return '<div style="min-width:' + minw + 'px;padding:6px 8px;margin:2px;background:#0a0a0a;border:1px dashed #333;border-radius:4px;color:#444;font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:center">?</div>';
  }

  // 向量一行
  _vecRow(label, vec, color, visible = true, labelColor) {
    const cells = visible
      ? vec.map(v => this._cell(v, color)).join('')
      : vec.map(() => this._hidden()).join('');
    return '<div style="display:flex;align-items:center;gap:10px;margin:5px 0;flex-wrap:wrap">' +
      '<span style="color:' + (labelColor || color) + ';min-width:130px;font-family:JetBrains Mono,monospace;font-size:0.78rem">' + label + '</span>' +
      '<div style="display:flex;flex-wrap:wrap">' + cells + '</div>' +
      '</div>';
  }

  // 矩阵（带可选行列标签 + 标题）
  _matrix(mat, color, opts = {}) {
    const d = opts.d ?? 3;
    const rowLabels = opts.rowLabels || null;
    const colLabels = opts.colLabels || null;
    let html = '<table style="border-collapse:separate;border-spacing:3px;font-family:JetBrains Mono,monospace;font-size:0.74rem">';
    if (colLabels) {
      html += '<tr><td></td>' + colLabels.map(c => '<td style="color:#888;padding:0 4px;text-align:center;font-size:0.7rem">' + c + '</td>').join('') + '</tr>';
    }
    for (let i = 0; i < mat.length; i++) {
      html += '<tr>';
      if (rowLabels) html += '<td style="color:#888;padding:0 4px;text-align:right;font-size:0.7rem">' + rowLabels[i] + '</td>';
      for (let j = 0; j < mat[i].length; j++) {
        const v = mat[i][j];
        const intensity = Math.min(Math.abs(v) * 0.35, 0.55);
        const bg = v === 0 ? '#111'
          : (v > 0 ? 'rgba(16,185,129,' + (0.12 + intensity) + ')'
                   : 'rgba(239,68,68,'  + (0.12 + intensity) + ')');
        html += '<td style="background:' + bg + ';border:1px solid ' + color + ';padding:5px 7px;color:' + color +
                ';text-align:center;border-radius:3px;min-width:46px">' + Number(v).toFixed(d) + '</td>';
      }
      html += '</tr>';
    }
    html += '</table>';
    return html;
  }

  // 矩阵 + 标题
  _matrixBlock(title, mat, color, opts = {}) {
    return '<div style="display:inline-block;vertical-align:top;margin:4px 10px 4px 0">' +
           '<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:' + color + ';margin-bottom:3px">' + title + '</div>' +
           this._matrix(mat, color, opts) +
           '</div>';
  }

  // 计算图节点（SVG）
  _computationGraph(activeStep) {
    const dir = this.STEPS[activeStep].dir;
    const nodes = [
      { id: 'x',  x: 60,  y: 80,  label: 'x',  sub: '(2,)',     color: '#3B82F6' },
      { id: 'z1', x: 200, y: 80,  label: 'z₁', sub: '(3,)',     color: '#F59E0B' },
      { id: 'h',  x: 340, y: 80,  label: 'h',  sub: 'ReLU(3,)', color: '#10B981' },
      { id: 'z2', x: 480, y: 80,  label: 'z₂', sub: '(2,)',     color: '#F59E0B' },
      { id: 'y',  x: 620, y: 80,  label: 'ŷ',  sub: 'softmax',  color: '#06B6D4' },
      { id: 'L',  x: 760, y: 80,  label: 'L',  sub: 'CE',       color: '#EF4444' }
    ];
    // 每条边属于哪一步
    const edges = [
      { from: 'x',  to: 'z1', op: '·W₁+b₁',  stepF: 1, stepB: [10] },
      { from: 'z1', to: 'h',  op: 'ReLU',    stepF: 2, stepB: [9]  },
      { from: 'h',  to: 'z2', op: '·W₂+b₂',  stepF: 3, stepB: [6, 7, 8] },
      { from: 'z2', to: 'y',  op: 'softmax', stepF: 4, stepB: [5]  },
      { from: 'y',  to: 'L',  op: 'CE',      stepF: 4, stepB: [5]  }
    ];
    const nodeById = {};
    nodes.forEach(n => nodeById[n.id] = n);

    // 哪些节点高亮（对应当前步骤）
    const activeNodes = (() => {
      switch (activeStep) {
        case 0:  return ['x'];
        case 1:  return ['x', 'z1'];
        case 2:  return ['z1', 'h'];
        case 3:  return ['h', 'z2'];
        case 4:  return ['z2', 'y', 'L'];
        case 5:  return ['y', 'L', 'z2'];
        case 6:  return ['h', 'z2'];
        case 7:  return ['z2'];
        case 8:  return ['z2', 'h'];
        case 9:  return ['z1', 'h'];
        case 10: return ['x', 'z1'];
        case 11: return ['x', 'z1', 'h', 'z2', 'y', 'L'];
        default: return [];
      }
    })();

    const W = 820, H = 170;
    let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" style="max-width:100%;display:block">';
    // defs
    svg += '<defs>' +
      '<marker id="fb-arrF" viewBox="0 -5 10 10" refX="8" refY="0" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,-5L10,0L0,5" fill="#60a5fa"/></marker>' +
      '<marker id="fb-arrB" viewBox="0 -5 10 10" refX="8" refY="0" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,-5L10,0L0,5" fill="#f97316"/></marker>' +
      '<marker id="fb-arrFdim" viewBox="0 -5 10 10" refX="8" refY="0" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,-5L10,0L0,5" fill="#1e3a5f"/></marker>' +
      '<marker id="fb-arrBdim" viewBox="0 -5 10 10" refX="8" refY="0" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,-5L10,0L0,5" fill="#3a2418"/></marker>' +
      '</defs>';

    // 边：前向（蓝）在上方；反向（橙）在下方弯
    edges.forEach(e => {
      const a = nodeById[e.from], b = nodeById[e.to];
      const activeF = (activeStep === e.stepF);
      const activeB = e.stepB.indexOf(activeStep) >= 0;

      // 前向箭头（直线）
      const colF = activeF ? '#60a5fa' : '#1e3a5f';
      const wF = activeF ? 2.5 : 1.2;
      const markerF = activeF ? 'fb-arrF' : 'fb-arrFdim';
      svg += '<line x1="' + (a.x + 22) + '" y1="' + (a.y - 6) + '" x2="' + (b.x - 22) + '" y2="' + (b.y - 6) +
             '" stroke="' + colF + '" stroke-width="' + wF + '" marker-end="url(#' + markerF + ')"/>';
      svg += '<text x="' + ((a.x + b.x) / 2) + '" y="' + (a.y - 14) + '" text-anchor="middle" font-size="10" fill="' +
             (activeF ? '#93c5fd' : '#3b4b6a') + '" font-family="JetBrains Mono,monospace">' + e.op + '</text>';

      // 反向箭头（曲线，从 to → from）
      const colB = activeB ? '#f97316' : '#3a2418';
      const wB = activeB ? 2.5 : 1.2;
      const markerB = activeB ? 'fb-arrB' : 'fb-arrBdim';
      const midX = (a.x + b.x) / 2;
      const curveY = a.y + 34;
      svg += '<path d="M ' + (b.x - 18) + ' ' + (b.y + 10) + ' Q ' + midX + ' ' + curveY +
             ' ' + (a.x + 18) + ' ' + (a.y + 10) + '" stroke="' + colB + '" stroke-width="' + wB +
             '" fill="none" marker-end="url(#' + markerB + ')"/>';
    });

    // 节点
    nodes.forEach(n => {
      const isActive = activeNodes.indexOf(n.id) >= 0;
      const r = isActive ? 26 : 22;
      const fill = isActive ? n.color : '#0d1117';
      const stroke = n.color;
      const sw = isActive ? 3 : 1.5;
      const textCol = isActive ? '#0d1117' : n.color;
      svg += '<circle cx="' + n.x + '" cy="' + n.y + '" r="' + r + '" fill="' + fill +
             '" stroke="' + stroke + '" stroke-width="' + sw + '"/>';
      svg += '<text x="' + n.x + '" y="' + (n.y + 5) + '" text-anchor="middle" font-size="16" font-weight="700" fill="' + textCol +
             '" font-family="JetBrains Mono,monospace">' + n.label + '</text>';
      svg += '<text x="' + n.x + '" y="' + (n.y + 48) + '" text-anchor="middle" font-size="10" fill="#888" font-family="JetBrains Mono,monospace">' + n.sub + '</text>';
    });

    // 方向标识
    svg += '<text x="12" y="24" font-size="11" fill="#60a5fa" font-family="Inter,sans-serif">→ 前向 (forward)</text>';
    svg += '<text x="12" y="162" font-size="11" fill="#f97316" font-family="Inter,sans-serif">← 反向 (backward)</text>';

    svg += '</svg>';

    // 当前方向徽章
    const badge = dir === 'F'
      ? '<span class="fb-badge fb-badge-fwd">FORWARD · 第 ' + (activeStep + 1) + ' / 5 步</span>'
      : dir === 'B'
        ? '<span class="fb-badge fb-badge-bwd">BACKWARD · 第 ' + (activeStep - 4) + ' / 6 步</span>'
        : '<span class="fb-badge fb-badge-upd">UPDATE · 梯度下降</span>';

    return '<div style="margin-bottom:8px">' + badge + '</div>' + svg;
  }

  // ============================================================
  // 每一步的详细面板
  // ============================================================
  _detailPanel(st) {
    const x = this.x, y = this.y_true;

    if (st === 0) {
      return '<div class="formula-box">x = [' + this.x.join(', ') + '],   y_true = [' + this.y_true.join(', ') + ']   (one-hot，正确类别 = ' + this.correctClass + ')</div>' +
        this._vecRow('x', this.x, '#3B82F6') +
        '<div class="edu-text">输入是一个 2 维特征向量；标签 y 用 one-hot 表示正确类别位置。我们的目标：训练网络让 ŷ[' + this.correctClass + '] 尽量接近 1。</div>';
    }

    if (st === 1) {
      // 手工展开 z1[0]
      const terms = [
        this._fmt(x[0], 1) + '×' + this._fmt(this.W1[0][0], 2),
        this._fmt(x[1], 1) + '×' + this._fmt(this.W1[1][0], 2),
        '+ b₁[0](' + this._fmt(this.b1[0], 2) + ')'
      ];
      return '<div class="formula-box">z₁ = x · W₁ + b₁    shape: (1×2)·(2×3) + (3,) = (3,)</div>' +
        '<div class="formula-box" style="font-size:0.76rem">z₁[0] = ' + terms[0] + ' + ' + terms[1] + ' ' + terms[2] +
        ' = <span style="color:#F59E0B">' + this._fmt(this.z1[0]) + '</span></div>' +
        this._matrixBlock('W₁ (2×3)', this.W1, '#3B82F6', { rowLabels: ['x₀', 'x₁'], colLabels: ['j=0', 'j=1', 'j=2'] }) +
        this._matrixBlock('b₁ (3,)', [this.b1], '#3B82F6', { colLabels: ['0', '1', '2'] }) +
        this._vecRow('z₁ (3,)', this.z1, '#F59E0B') +
        '<div class="edu-text">线性层把输入从 2 维"升"到 3 维。每个 z₁[j] 都是输入向量与 W₁ 的第 j 列做点积再加上偏置。</div>';
    }

    if (st === 2) {
      const masked = this.z1.map((v, i) => (v > 0 ? 'z₁[' + i + ']=' + this._fmt(v, 2) + ' > 0 → h[' + i + ']=' + this._fmt(v, 2)
                                                 : 'z₁[' + i + ']=' + this._fmt(v, 2) + ' ≤ 0 → h[' + i + ']=0 （被 ReLU 抑制）')).join('<br>');
      return '<div class="formula-box">h = ReLU(z₁) = max(0, z₁)</div>' +
        '<div class="edu-text" style="font-family:JetBrains Mono,monospace;font-size:0.76rem">' + masked + '</div>' +
        this._vecRow('z₁', this.z1, '#F59E0B') +
        this._vecRow('h = ReLU(z₁)', this.h, '#10B981') +
        '<div class="edu-text">ReLU 把负数"切掉"，引入非线性。注意：被切掉的神经元在反向传播时梯度为 0——这就是"死 ReLU"问题的根源。</div>';
    }

    if (st === 3) {
      const exp0 = this.h.map((hv, i) => this._fmt(hv, 2) + '×' + this._fmt(this.W2[i][0], 2)).join(' + ') + ' + ' + this._fmt(this.b2[0], 2);
      return '<div class="formula-box">z₂ = h · W₂ + b₂    shape: (1×3)·(3×2) + (2,) = (2,)</div>' +
        '<div class="formula-box" style="font-size:0.76rem">z₂[0] = ' + exp0 + ' = <span style="color:#F59E0B">' + this._fmt(this.z2[0]) + '</span></div>' +
        this._matrixBlock('W₂ (3×2)', this.W2, '#3B82F6', { rowLabels: ['h₀', 'h₁', 'h₂'], colLabels: ['class0', 'class1'] }) +
        this._matrixBlock('b₂ (2,)', [this.b2], '#3B82F6', { colLabels: ['0', '1'] }) +
        this._vecRow('z₂ (logits)', this.z2, '#F59E0B') +
        '<div class="edu-text">第二层把 3 维隐藏表示投影成 2 个 logits（每个类别一个）。尚未归一化成概率。</div>';
    }

    if (st === 4) {
      const sumExp = Math.exp(this.z2[0]) + Math.exp(this.z2[1]);
      return '<div class="formula-box">ŷᵢ = e^{z₂ᵢ} / Σⱼ e^{z₂ⱼ}</div>' +
        '<div class="formula-box" style="font-size:0.76rem">Σ e^{z₂} = e^{' + this._fmt(this.z2[0], 2) + '} + e^{' + this._fmt(this.z2[1], 2) + '} = ' + this._fmt(sumExp, 3) + '</div>' +
        this._vecRow('z₂ (logits)', this.z2, '#F59E0B') +
        this._vecRow('ŷ (probs)', this.y_pred, '#06B6D4') +
        '<div class="formula-box" style="color:#f87171">L = -log ŷ[' + this.correctClass + '] = -log(' + this._fmt(this.y_pred[this.correctClass], 4) + ') = <b>' + this._fmt(this.loss, 4) + '</b></div>' +
        '<div class="edu-text">softmax 把任意实数向量变成概率分布（元素非负、总和为 1）。交叉熵 L 只关心"正确类别的预测概率"：ŷ[y] 越接近 1，L 越接近 0。</div>';
    }

    if (st === 5) {
      return '<div class="formula-box">∂L/∂z₂ = ŷ - y   （softmax + 交叉熵的神奇合并）</div>' +
        this._vecRow('ŷ', this.y_pred, '#06B6D4') +
        this._vecRow('y (one-hot)', this.y_true, '#888') +
        this._vecRow('∂L/∂z₂ = ŷ - y', this.dL_dz2, '#F97316') +
        '<div class="edu-text">如果分开求 ∂L/∂ŷ 和 ∂ŷ/∂z₂ 会得到一大坨式子。合并后居然只剩 <b>ŷ - y</b>！正类方向梯度为负（要"推高"），其它类方向梯度为正（要"压低"）。</div>';
    }

    if (st === 6) {
      return '<div class="formula-box">∂L/∂W₂ = hᵀ · ∂L/∂z₂    shape: (3×1)·(1×2) = (3×2)   (外积)</div>' +
        this._matrixBlock('h (列向量 3×1)', this.h.map(v => [v]), '#10B981') +
        '<div style="display:inline-block;vertical-align:top;font-size:1.3rem;color:#666;padding:20px 8px 0">×</div>' +
        this._matrixBlock('∂L/∂z₂ (行向量 1×2)', [this.dL_dz2], '#F97316') +
        '<div style="display:inline-block;vertical-align:top;font-size:1.3rem;color:#666;padding:20px 8px 0">=</div>' +
        this._matrixBlock('∂L/∂W₂ (3×2)', this.dL_dW2, '#F97316', { rowLabels: ['h₀', 'h₁', 'h₂'], colLabels: ['c0', 'c1'] }) +
        '<div class="edu-text">外积的直观解释：每个权重 W₂[i,j] 的梯度 = 它连接的前端激活值 h[i] × 后端梯度 ∂L/∂z₂[j]。激活越大的神经元，它对应的权重"感受"的梯度越强。</div>';
    }

    if (st === 7) {
      return '<div class="formula-box">∂L/∂b₂ = ∂L/∂z₂  （bias 的梯度就是它所处层的梯度本身）</div>' +
        this._vecRow('∂L/∂b₂', this.dL_db2, '#F97316') +
        '<div class="edu-text">因为 z₂ = h·W₂ + b₂，所以 ∂z₂/∂b₂ = I。链式法则乘以恒等矩阵，结果不变。</div>';
    }

    if (st === 8) {
      return '<div class="formula-box">∂L/∂h = ∂L/∂z₂ · W₂ᵀ    shape: (1×2)·(2×3) = (1×3)</div>' +
        this._matrixBlock('∂L/∂z₂ (1×2)', [this.dL_dz2], '#F97316') +
        '<div style="display:inline-block;vertical-align:top;font-size:1.3rem;color:#666;padding:20px 8px 0">·</div>' +
        this._matrixBlock('W₂ᵀ (2×3)', [0, 1].map(j => [0, 1, 2].map(i => this.W2[i][j])), '#3B82F6') +
        this._vecRow('∂L/∂h', this.dL_dh, '#F97316') +
        '<div class="edu-text">梯度继续往回传。W₂ᵀ 把"输出侧的责任"按权重分发回每个隐藏单元。</div>';
    }

    if (st === 9) {
      const lines = this.z1.map((z, i) => 'z₁[' + i + ']=' + this._fmt(z, 2) + (z > 0 ? " → ReLU'=1" : " → ReLU'=0 （梯度被阻断）") +
        '，  ∂L/∂h[' + i + ']=' + this._fmt(this.dL_dh[i]) + ' × ' + (z > 0 ? '1' : '0') + ' = ' + this._fmt(this.dL_dz1[i])).join('<br>');
      return '<div class="formula-box">ReLU′(z) = 1 若 z>0 否则 0</div>' +
        '<div class="formula-box">∂L/∂z₁ = ∂L/∂h ⊙ ReLU′(z₁)   （⊙ = 按元素乘）</div>' +
        '<div class="edu-text" style="font-family:JetBrains Mono,monospace;font-size:0.76rem">' + lines + '</div>' +
        this._vecRow('ReLU′(z₁)', this.reluMask, '#666') +
        this._vecRow('∂L/∂h', this.dL_dh, '#F97316') +
        this._vecRow('∂L/∂z₁', this.dL_dz1, '#F97316') +
        '<div class="edu-text">凡是前向时被 ReLU 切掉的位置，反向时梯度也同样为 0——这就是"死神经元"：一旦输入持续为负，这个神经元再也学不动了。</div>';
    }

    if (st === 10) {
      return '<div class="formula-box">∂L/∂W₁ = xᵀ · ∂L/∂z₁    shape: (2×1)·(1×3) = (2×3)</div>' +
        '<div class="formula-box">∂L/∂b₁ = ∂L/∂z₁</div>' +
        this._matrixBlock('x (2×1)', this.x.map(v => [v]), '#3B82F6') +
        '<div style="display:inline-block;vertical-align:top;font-size:1.3rem;color:#666;padding:20px 8px 0">×</div>' +
        this._matrixBlock('∂L/∂z₁ (1×3)', [this.dL_dz1], '#F97316') +
        '<div style="display:inline-block;vertical-align:top;font-size:1.3rem;color:#666;padding:20px 8px 0">=</div>' +
        this._matrixBlock('∂L/∂W₁ (2×3)', this.dL_dW1, '#F97316', { rowLabels: ['x₀', 'x₁'], colLabels: ['j=0', 'j=1', 'j=2'] }) +
        this._vecRow('∂L/∂b₁', this.dL_db1, '#F97316') +
        '<div class="edu-text">所有梯度全部算完。注意到 ∂L/∂W₁ 的第 j 列如果 ∂L/∂z₁[j]=0（被 ReLU 阻断），整列都是 0——这个神经元本轮不更新。</div>';
    }

    if (st === 11) {
      // 更新视图：before/after 对比
      return '<div class="formula-box">θ ← θ - η · ∂L/∂θ ,   学习率 η = ' + this.lr + '</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:20px;margin-top:10px">' +
          '<div>' +
            '<div style="color:#93c5fd;font-size:0.82rem;margin-bottom:6px">W₂ (更新前)</div>' +
            this._matrix(this.W2, '#3B82F6') +
          '</div>' +
          '<div style="padding:20px 4px 0;color:#666">−  η ·</div>' +
          '<div>' +
            '<div style="color:#f97316;font-size:0.82rem;margin-bottom:6px">∂L/∂W₂</div>' +
            this._matrix(this.dL_dW2, '#F97316') +
          '</div>' +
          '<div style="padding:20px 4px 0;color:#666">=</div>' +
          '<div>' +
            '<div style="color:#10B981;font-size:0.82rem;margin-bottom:6px">W₂ (更新后)</div>' +
            this._matrix(this.W2_new, '#10B981') +
          '</div>' +
        '</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:20px;margin-top:16px">' +
          '<div>' +
            '<div style="color:#93c5fd;font-size:0.82rem;margin-bottom:6px">W₁ (更新前)</div>' +
            this._matrix(this.W1, '#3B82F6') +
          '</div>' +
          '<div style="padding:20px 4px 0;color:#666">−  η ·</div>' +
          '<div>' +
            '<div style="color:#f97316;font-size:0.82rem;margin-bottom:6px">∂L/∂W₁</div>' +
            this._matrix(this.dL_dW1, '#F97316') +
          '</div>' +
          '<div style="padding:20px 4px 0;color:#666">=</div>' +
          '<div>' +
            '<div style="color:#10B981;font-size:0.82rem;margin-bottom:6px">W₁ (更新后)</div>' +
            this._matrix(this.W1_new, '#10B981') +
          '</div>' +
        '</div>' +
        '<div class="edu-text" style="margin-top:14px">梯度指向 loss 增加最快的方向；减去梯度 × 学习率，参数就往 loss 下降最快的方向挪一小步。重复成千上万次这个过程，网络就学会了。</div>';
    }

    return '';
  }

  // ============================================================
  // 链式法则侧栏：针对当前步骤展示"分解 = 各因子"
  // ============================================================
  _chainPanel(st) {
    const items = [];
    items.push({ name: '总目标', tex: '∂L/∂W₁ = ∂L/∂z₁ · ∂z₁/∂W₁', hi: (st === 10) });
    items.push({ name: '总目标', tex: '∂L/∂W₂ = ∂L/∂z₂ · ∂z₂/∂W₂', hi: (st === 6) });
    items.push({ name: '展开', tex: '∂L/∂W₁ = ((ŷ−y) · W₂ᵀ ⊙ ReLU′(z₁)) · xᵀ', hi: (st === 10) });

    const factors = [
      { k: 'ŷ - y',           v: this.dL_dz2, dim: '(2,)', step: 5, color: '#F97316' },
      { k: '· W₂ᵀ',           v: this.dL_dh,  dim: '(3,)', step: 8, color: '#F97316' },
      { k: "⊙ ReLU'(z₁)",     v: this.dL_dz1, dim: '(3,)', step: 9, color: '#F97316' },
      { k: '· xᵀ  ⇒ ∂L/∂W₁',  v: null,         dim: '(2×3)', step: 10, color: '#F97316' }
    ];

    let html = '<div style="font-weight:600;margin-bottom:8px">链式法则分解</div>';
    items.forEach(it => {
      html += '<div class="formula-box" style="' + (it.hi ? 'border:1px solid #F97316;' : '') + '">' + it.tex + '</div>';
    });

    html += '<div style="margin-top:10px;font-size:0.78rem;color:#888">各因子的当前数值：</div>';
    factors.forEach(f => {
      const hl = (st === f.step);
      const box = '<div class="chain-factor' + (hl ? ' chain-factor-active' : '') + '">' +
        '<div style="display:flex;justify-content:space-between;align-items:baseline">' +
          '<span style="font-family:JetBrains Mono,monospace;color:' + f.color + ';font-size:0.78rem">' + f.k + '</span>' +
          '<span style="font-size:0.68rem;color:#666">' + f.dim + '</span>' +
        '</div>' +
        (f.v ? '<div style="display:flex;margin-top:4px">' + f.v.map(v => this._cell(v, f.color, { minw: 48, d: 3 })).join('') + '</div>' : '') +
      '</div>';
      html += box;
    });
    return html;
  }

  // ============================================================
  // 顶部步骤点 (clickable)
  // ============================================================
  _stepDots(current) {
    return this.STEPS.map((s, i) => {
      const active = i === current;
      const done = i < current;
      const cls = ['fb-dot'];
      cls.push(s.dir === 'F' ? 'fb-dot-fwd' : s.dir === 'B' ? 'fb-dot-bwd' : 'fb-dot-upd');
      if (active) cls.push('fb-dot-active');
      else if (done) cls.push('fb-dot-done');
      const title = (i + 1) + '. ' + s.title;
      return '<div class="' + cls.join(' ') + '" title="' + title + '" onclick="window._fbInstance.goTo(' + i + ')">' +
        '<span class="fb-dot-num">' + (i + 1) + '</span>' +
        '<span class="fb-dot-lbl">' + s.short + '</span>' +
      '</div>';
    }).join('');
  }

  _progressBar(current) {
    const pct = (current / (this.STEPS.length - 1)) * 100;
    return '<div class="fb-progress"><div class="fb-progress-bar" style="width:' + pct.toFixed(1) + '%"></div></div>';
  }

  // ============================================================
  // 网络结构小面板（右侧常驻）
  // ============================================================
  _networkPanel(st) {
    const fwdVisible = {
      x:  st >= 0,
      z1: st >= 1,
      h:  st >= 2,
      z2: st >= 3,
      y:  st >= 4,
      L:  st >= 4
    };
    const bwdVisible = {
      dz2: st >= 5,
      dW2: st >= 6,
      db2: st >= 7,
      dh:  st >= 8,
      dz1: st >= 9,
      dW1: st >= 10
    };

    const row = (lbl, vec, visible, col) => {
      if (!visible) return '<div class="fb-mini-row"><span class="fb-mini-lbl" style="color:#555">' + lbl + '</span><span style="color:#444;font-family:JetBrains Mono,monospace;font-size:0.7rem">—</span></div>';
      const cells = vec.map(v => this._cell(v, col, { minw: 44, d: 2 })).join('');
      return '<div class="fb-mini-row"><span class="fb-mini-lbl" style="color:' + col + '">' + lbl + '</span><div style="display:flex">' + cells + '</div></div>';
    };

    let html = '<div style="font-weight:600;margin-bottom:8px">实时激活值 / 梯度</div>';
    html += '<div style="font-size:0.74rem;color:#93c5fd;margin:8px 0 4px">前向 ↓</div>';
    html += row('x',      this.x,              fwdVisible.x,  '#3B82F6');
    html += row('z₁',     this.z1,             fwdVisible.z1, '#F59E0B');
    html += row('h',      this.h,              fwdVisible.h,  '#10B981');
    html += row('z₂',     this.z2,             fwdVisible.z2, '#F59E0B');
    html += row('ŷ',      this.y_pred,         fwdVisible.y,  '#06B6D4');
    if (fwdVisible.L) {
      html += '<div class="fb-mini-row"><span class="fb-mini-lbl" style="color:#EF4444">L</span>' +
              '<span style="font-family:JetBrains Mono,monospace;color:#f87171;font-size:0.78rem">' + this._fmt(this.loss, 4) + '</span></div>';
    }
    html += '<div style="font-size:0.74rem;color:#f97316;margin:10px 0 4px">反向 ↑</div>';
    html += row('∂L/∂z₂', this.dL_dz2,         bwdVisible.dz2, '#F97316');
    html += row('∂L/∂b₂', this.dL_db2,         bwdVisible.db2, '#F97316');
    html += row('∂L/∂h',  this.dL_dh,          bwdVisible.dh,  '#F97316');
    html += row('∂L/∂z₁', this.dL_dz1,         bwdVisible.dz1, '#F97316');
    return html;
  }

  // ============================================================
  // 主渲染
  // ============================================================
  render() {
    if (!this.container) return;
    const st = this.currentStep;
    const stepMeta = this.STEPS[st];

    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._fbInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._fbInstance.play()">▶ 播放</button>';

    const html = '' +
      '<div class="fb-viz">' +
        '<style>' +
          '.fb-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.fb-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem;transition:all 0.15s}' +
          '.fb-viz .ctrl-btn:hover{background:#252525;border-color:#555}' +
          '.fb-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.fb-viz .speed-select{padding:6px 10px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#e5e5e5;font-size:0.85rem;margin-right:8px}' +
          '.fb-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.fb-viz .section{background:#1a1a1a;border-radius:8px;padding:16px;border:1px solid #333;margin-bottom:14px}' +
          '.fb-viz .section-title{font-weight:600;margin-bottom:10px;font-size:1rem;color:#fff}' +
          '.fb-viz .two-col{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}' +
          '.fb-viz .edu-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}' +
          '@media(max-width:1000px){.fb-viz .two-col{grid-template-columns:1fr}.fb-viz .edu-grid{grid-template-columns:1fr}}' +
          '.fb-viz .edu-text{color:#9ca3af;font-size:0.85rem;line-height:1.7;margin-top:8px}' +
          '.fb-viz .edu-card{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px}' +
          '.fb-viz .edu-card h4{margin:0 0 8px;color:#e5e5e5;font-size:0.92rem}' +
          '.fb-viz .edu-card p{margin:0;color:#9ca3af;font-size:0.82rem;line-height:1.7}' +
          '.fb-viz .fb-title{font-size:1.5rem;font-weight:600;margin-bottom:4px;color:#fff}' +
          '.fb-viz .fb-sub{color:#9ca3af;font-size:0.88rem;margin-bottom:14px}' +
          '.fb-viz .fb-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:600;letter-spacing:0.5px}' +
          '.fb-viz .fb-badge-fwd{background:#1e3a5f;color:#93c5fd;border:1px solid #3b82f6}' +
          '.fb-viz .fb-badge-bwd{background:#3a2418;color:#fdba74;border:1px solid #f97316}' +
          '.fb-viz .fb-badge-upd{background:#1a3a2e;color:#86efac;border:1px solid #10B981}' +
          '.fb-viz .fb-dots{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}' +
          '.fb-viz .fb-dot{display:flex;flex-direction:column;align-items:center;padding:5px 9px;border-radius:6px;cursor:pointer;border:1px solid #333;background:#0d1117;min-width:58px;transition:all 0.15s}' +
          '.fb-viz .fb-dot:hover{border-color:#555;transform:translateY(-1px)}' +
          '.fb-viz .fb-dot-num{font-size:0.62rem;color:#666;line-height:1}' +
          '.fb-viz .fb-dot-lbl{font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#888;margin-top:2px}' +
          '.fb-viz .fb-dot-fwd{border-color:#1e3a5f}' +
          '.fb-viz .fb-dot-bwd{border-color:#3a2418}' +
          '.fb-viz .fb-dot-upd{border-color:#1a3a2e}' +
          '.fb-viz .fb-dot-done.fb-dot-fwd{background:#132538;border-color:#3b82f6}' +
          '.fb-viz .fb-dot-done.fb-dot-fwd .fb-dot-lbl{color:#60a5fa}' +
          '.fb-viz .fb-dot-done.fb-dot-bwd{background:#2a1a10;border-color:#f97316}' +
          '.fb-viz .fb-dot-done.fb-dot-bwd .fb-dot-lbl{color:#fb923c}' +
          '.fb-viz .fb-dot-done.fb-dot-upd{background:#122921;border-color:#10b981}' +
          '.fb-viz .fb-dot-done.fb-dot-upd .fb-dot-lbl{color:#34d399}' +
          '.fb-viz .fb-dot-active{background:#3B82F6;border-color:#3B82F6;transform:translateY(-2px);box-shadow:0 4px 12px rgba(59,130,246,0.4)}' +
          '.fb-viz .fb-dot-active .fb-dot-num,.fb-viz .fb-dot-active .fb-dot-lbl{color:#fff}' +
          '.fb-viz .fb-dot-active.fb-dot-bwd{background:#F97316;border-color:#F97316;box-shadow:0 4px 12px rgba(249,115,22,0.4)}' +
          '.fb-viz .fb-dot-active.fb-dot-upd{background:#10B981;border-color:#10B981;box-shadow:0 4px 12px rgba(16,185,129,0.4)}' +
          '.fb-viz .fb-progress{height:4px;background:#1a1a1a;border-radius:2px;margin:10px 0;overflow:hidden}' +
          '.fb-viz .fb-progress-bar{height:100%;background:linear-gradient(90deg,#3B82F6 0%,#F97316 55%,#10B981 100%);transition:width 0.4s ease}' +
          '.fb-viz .fb-mini-row{display:flex;align-items:center;gap:8px;margin:3px 0}' +
          '.fb-viz .fb-mini-lbl{min-width:60px;font-family:JetBrains Mono,monospace;font-size:0.76rem}' +
          '.fb-viz .chain-factor{background:#0d1117;border:1px solid #21262d;border-radius:5px;padding:6px 8px;margin:5px 0;transition:all 0.2s}' +
          '.fb-viz .chain-factor-active{border-color:#F97316;box-shadow:0 0 0 1px rgba(249,115,22,0.3)}' +
          '.fb-viz .gotcha-list{margin:0;padding-left:18px;color:#9ca3af;font-size:0.82rem;line-height:1.7}' +
          '.fb-viz .gotcha-list li{margin:6px 0}' +
          '.fb-viz .gotcha-list b{color:#e5e5e5}' +
        '</style>' +

        // 标题
        '<div class="fb-title">神经网络 · 前向 / 反向传播</div>' +
        '<div class="fb-sub">2 层网络：Input(2) → Linear → ReLU → Linear → Softmax → CrossEntropy。逐步拆解梯度如何从 loss 顺着链式法则一路回传，并演示一次梯度下降参数更新。</div>' +

        // 控制面板
        '<div class="section">' +
          '<div class="section-title">控制</div>' +
          '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px">' +
            '<button class="ctrl-btn" onclick="window._fbInstance.stepBack()">⏮ 上一步</button>' +
            playBtn +
            '<button class="ctrl-btn" onclick="window._fbInstance.stepForward()">⏭ 下一步</button>' +
            '<button class="ctrl-btn" onclick="window._fbInstance.reset()">↻ 重置</button>' +
            '<select class="speed-select" onchange="window._fbInstance.setSpeed(this.value)">' +
              '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="3">3×</option>' +
            '</select>' +
            '<span style="color:#666;font-size:0.82rem;margin-left:8px">进度 ' + (st + 1) + ' / ' + this.STEPS.length + '</span>' +
          '</div>' +
          this._progressBar(st) +
          '<div class="fb-dots">' + this._stepDots(st) + '</div>' +
        '</div>' +

        // 计算图
        '<div class="section">' +
          '<div class="section-title">计算图（前向：蓝色直线 · 反向：橙色曲线）</div>' +
          this._computationGraph(st) +
        '</div>' +

        // 主内容双栏
        '<div class="two-col">' +
          // 左：当前步骤详情
          '<div class="section">' +
            '<div class="section-title">当前步骤 · ' + stepMeta.title + '</div>' +
            '<div class="formula-box" style="color:#93c5fd">' + stepMeta.formula + '</div>' +
            this._detailPanel(st) +
          '</div>' +
          // 右：链式法则 + 实时值
          '<div>' +
            '<div class="section">' +
              this._networkPanel(st) +
            '</div>' +
            '<div class="section">' +
              this._chainPanel(st) +
            '</div>' +
          '</div>' +
        '</div>' +

        // 教学面板
        '<div class="section">' +
          '<div class="section-title">教学要点</div>' +
          '<div class="edu-grid">' +
            '<div class="edu-card">' +
              '<h4>🔮 什么是梯度？</h4>' +
              '<p>梯度 ∂L/∂W 告诉我们：<b>loss 相对参数 W 增加最快的方向</b>。我们想让 loss <i>下降</i>，所以沿着负梯度走。学习率 η 控制步长：太大会震荡，太小会太慢。</p>' +
            '</div>' +
            '<div class="edu-card">' +
              '<h4>🔗 为什么链式法则就是全部？</h4>' +
              '<p>神经网络是函数的函数的函数…。求外层对最内层参数的导数，链式法则告诉我们：<b>沿着计算图反向走一遍，把每条边的局部导数连乘起来</b>。反向传播 = 自动化的链式法则。</p>' +
            '</div>' +
            '<div class="edu-card">' +
              '<h4>✨ softmax + CE = ŷ - y</h4>' +
              '<p>softmax 和交叉熵单独求导都很复杂，但组合起来神奇地化简为 <b>ŷ - y</b>。这也是为什么分类任务几乎总是 softmax 配交叉熵——不只是数学上漂亮，数值上也更稳定（避免了对 softmax 求导后可能溢出的中间量）。</p>' +
            '</div>' +
            '<div class="edu-card">' +
              '<h4>⚠️ 常见陷阱</h4>' +
              '<ul class="gotcha-list">' +
                '<li><b>梯度消失（Sigmoid/Tanh）</b>：这些激活的导数最大值分别只有 0.25 / 1。深层网络每传一层梯度就被乘一个 &lt; 1 的数，指数级衰减。</li>' +
                '<li><b>死 ReLU</b>：ReLU 在 z ≤ 0 时梯度为 0。若某个神经元长期输入负数，它就再也更新不了。解决：用 LeakyReLU / GELU，或小心初始化。</li>' +
                '<li><b>梯度爆炸</b>：权重过大或序列过长（RNN），梯度指数级放大。解决：梯度裁剪、合适的初始化、LayerNorm。</li>' +
                '<li><b>学习率敏感</b>：太大发散，太小停滞。Adam 等自适应优化器会给每个参数单独调步长。</li>' +
              '</ul>' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 完整总结
        '<div class="section">' +
          '<div class="section-title">完整公式总览</div>' +
          '<div class="formula-box">前向：  z₁ = xW₁+b₁ ;  h = ReLU(z₁) ;  z₂ = hW₂+b₂ ;  ŷ = softmax(z₂) ;  L = -log ŷ[y]</div>' +
          '<div class="formula-box">反向：  ∂L/∂z₂ = ŷ-y ;  ∂L/∂W₂ = hᵀ·∂L/∂z₂ ;  ∂L/∂b₂ = ∂L/∂z₂</div>' +
          '<div class="formula-box">反向：  ∂L/∂h = ∂L/∂z₂·W₂ᵀ ;  ∂L/∂z₁ = ∂L/∂h ⊙ ReLU′(z₁) ;  ∂L/∂W₁ = xᵀ·∂L/∂z₁ ;  ∂L/∂b₁ = ∂L/∂z₁</div>' +
          '<div class="formula-box">更新：  W ← W - η · ∂L/∂W ,    b ← b - η · ∂L/∂b</div>' +
        '</div>' +
      '</div>';

    this.container.innerHTML = html;
    window._fbInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._fbInstance === this) {
      try { delete window._fbInstance; } catch (e) { window._fbInstance = null; }
    }
    this.container = null;
  }
}

window.ForwardBackward = ForwardBackward;
