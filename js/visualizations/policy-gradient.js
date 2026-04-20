// Policy Gradient (REINFORCE) 可视化
// 策略网络: π_θ(a|s) = softmax(s·W + b)
// 策略梯度: ∇J(θ) = Σ_t ∇log π_θ(a_t|s_t) · G_t
// 更新: θ ← θ + α · ∇J
class PolicyGradient {
  constructor() {
    this.container = null;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // ---------- 超参数 ----------
    this.gamma = 0.9;   // 折扣因子
    this.alpha = 0.2;   // 学习率（偏大，便于观察更新效果）

    // ---------- 状态空间（2 个状态, 2 维特征） ----------
    this.states = [
      [1.0, 0.5],  // s0
      [0.3, 0.7]   // s1
    ];
    this.stateNames = ['s0', 's1'];

    // ---------- 动作空间（3 个动作） ----------
    this.actionNames = ['a0', 'a1', 'a2'];
    this.actionColors = ['#3B82F6', '#8B5CF6', '#F97316'];

    // ---------- 策略网络参数 W:(2,3), b:(3,) ----------
    this.W_init = [
      [ 0.5, -0.3,  0.8],
      [-0.2,  0.7, -0.4]
    ];
    this.b_init = [0.1, 0.0, -0.1];
    this.W = this._clone2D(this.W_init);
    this.b = this.b_init.slice();

    // 记录更新前的策略（用于 before/after 对比）
    this.W_old = this._clone2D(this.W_init);
    this.b_old = this.b_init.slice();

    // ---------- 固定轨迹 ----------
    // (s, a, r) 三元组
    this.trajectory = [
      { s: 0, a: 1, r: 1.0 },
      { s: 1, a: 0, r: 0.0 },
      { s: 0, a: 2, r: 2.0 }
    ];

    // ---------- 步骤定义 ----------
    // 0: 介绍策略网络
    // 1..T: 访问轨迹第 t 步，显示 π(a|s)、采样动作、即时回报
    // T+1: 计算全部 G_t
    // T+2: 计算 ∇log π 与每步梯度贡献
    // T+3: 汇总策略梯度 g
    // T+4: 更新参数 θ ← θ + α·g
    // T+5: 显示 before/after 策略分布对比
    const T = this.trajectory.length;
    this.STEPS = [
      '① 策略网络结构',
      '② 轨迹 t=0 — π(a|s0)',
      '③ 轨迹 t=1 — π(a|s1)',
      '④ 轨迹 t=2 — π(a|s0)',
      '⑤ 计算累积回报 G_t',
      '⑥ 计算 ∇log π(a|s)',
      '⑦ 汇总策略梯度 g',
      '⑧ 参数更新 θ ← θ + α·g',
      '⑨ 策略分布 Before / After'
    ];
    this.currentStep = 0;

    this._compute();
  }

  // ==================== 公共接口 ====================
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this.W = this._clone2D(this.W_init);
    this.b = this.b_init.slice();
    this.W_old = this._clone2D(this.W_init);
    this.b_old = this.b_init.slice();
    this._compute();
    this.render();
    return this;
  }

  reset() {
    this.isPlaying = false;
    clearTimeout(this.timer);
    this.currentStep = 0;
    this.W = this._clone2D(this.W_init);
    this.b = this.b_init.slice();
    this.W_old = this._clone2D(this.W_init);
    this.b_old = this.b_init.slice();
    this._compute();
    this.render();
  }

  play() {
    this.isPlaying = true;
    this._auto();
  }

  pause() {
    this.isPlaying = false;
    clearTimeout(this.timer);
    this.render();
  }

  setSpeed(s) { this.speed = Number(s) || 1; }

  stepForward() {
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this._applyStepEffects();
      this.render();
    }
  }

  stepBack() {
    if (this.currentStep > 0) {
      this.currentStep--;
      // 若从「更新后」退回，需要恢复策略参数
      if (this.currentStep < 7) {
        this.W = this._clone2D(this.W_init);
        this.b = this.b_init.slice();
        this._compute();
      }
      this.render();
    }
  }

  goTo(i) {
    const target = Math.max(0, Math.min(i, this.STEPS.length - 1));
    // 重置参数再前进，保证一致性
    this.W = this._clone2D(this.W_init);
    this.b = this.b_init.slice();
    this._compute();
    this.currentStep = 0;
    while (this.currentStep < target) {
      this.currentStep++;
      this._applyStepEffects();
    }
    this.render();
  }

  // ==================== 核心计算 ====================
  _clone2D(M) { return M.map(r => r.slice()); }

  _softmax(logits) {
    const mx = Math.max.apply(null, logits);
    const ex = logits.map(v => Math.exp(v - mx));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(e => e / s);
  }

  _logits(sIdx, W, b) {
    const s = this.states[sIdx];
    return [0, 1, 2].map(j => s[0] * W[0][j] + s[1] * W[1][j] + b[j]);
  }

  _pi(sIdx, W, b) {
    return this._softmax(this._logits(sIdx, W || this.W, b || this.b));
  }

  /**
   * 预计算：
   *  - G_t（每步的累积折扣回报）
   *  - ∇_θ log π(a_t|s_t)（对 W 和 b 的梯度）
   *  - 策略梯度 g = Σ G_t · ∇log π
   */
  _compute() {
    const T = this.trajectory.length;

    // G_t = Σ_{k≥t} γ^{k-t} · r_k
    this.G = [];
    for (let t = 0; t < T; t++) {
      let g = 0;
      for (let k = t; k < T; k++) {
        g += Math.pow(this.gamma, k - t) * this.trajectory[k].r;
      }
      this.G.push(g);
    }

    // ∇_W log π(a|s)[i,j] = s[i] · (1_{j=a} - π_j(s))
    // ∇_b log π(a|s)[j]   =        (1_{j=a} - π_j(s))
    this.gradLog = [];   // 每步的 {gW, gb}
    for (let t = 0; t < T; t++) {
      const { s, a } = this.trajectory[t];
      const sv = this.states[s];
      const pi = this._pi(s);
      const gW = [[0,0,0],[0,0,0]];
      const gb = [0,0,0];
      for (let j = 0; j < 3; j++) {
        const indicator = (j === a) ? 1 : 0;
        const diff = indicator - pi[j];
        gb[j] = diff;
        for (let i = 0; i < 2; i++) {
          gW[i][j] = sv[i] * diff;
        }
      }
      this.gradLog.push({ gW, gb, pi });
    }

    // 策略梯度 g = Σ_t G_t · ∇log π(a_t|s_t)
    this.gW = [[0,0,0],[0,0,0]];
    this.gb = [0,0,0];
    for (let t = 0; t < T; t++) {
      const Gt = this.G[t];
      const { gW, gb } = this.gradLog[t];
      for (let j = 0; j < 3; j++) {
        this.gb[j] += Gt * gb[j];
        for (let i = 0; i < 2; i++) {
          this.gW[i][j] += Gt * gW[i][j];
        }
      }
    }
  }

  // 应用某一步特有的状态变化（例如参数更新）
  _applyStepEffects() {
    // 进入第 7 步（下标 7 = "⑧ 参数更新"）时，应用更新
    if (this.currentStep === 7) {
      this.W_old = this._clone2D(this.W);
      this.b_old = this.b.slice();
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 3; j++) {
          this.W[i][j] += this.alpha * this.gW[i][j];
        }
      }
      for (let j = 0; j < 3; j++) {
        this.b[j] += this.alpha * this.gb[j];
      }
    }
  }

  // ==================== 播放器 ====================
  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this._applyStepEffects();
      this.render();
      this.timer = setTimeout(() => this._auto(), 1600 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  // ==================== 渲染辅助 ====================
  _fmt(x, d = 3) { return (x >= 0 ? ' ' : '') + x.toFixed(d); }

  _probBar(pi, highlightA = -1, label = '') {
    const bars = pi.map((p, j) => {
      const c = this.actionColors[j];
      const hl = j === highlightA;
      return (
        '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
          '<span style="width:22px;font-size:0.72rem;color:' + c + ';font-weight:' + (hl ? 700 : 500) + '">' + this.actionNames[j] + '</span>' +
          '<div style="flex:1;height:16px;background:#0a0a0a;border-radius:3px;overflow:hidden;border:1px solid ' + (hl ? c : '#222') + '">' +
            '<div style="width:' + (p * 100).toFixed(1) + '%;height:100%;background:' + c + ';opacity:' + (hl ? 0.95 : 0.55) + '"></div>' +
          '</div>' +
          '<span style="width:46px;text-align:right;font-size:0.72rem;font-family:JetBrains Mono,monospace;color:' + c + '">' + (p * 100).toFixed(1) + '%</span>' +
        '</div>'
      );
    }).join('');
    return (label ? '<div style="color:#58a6ff;font-size:0.78rem;margin-bottom:4px">' + label + '</div>' : '') + bars;
  }

  _matrixTable(M, label, color, cmpM) {
    // M: (rows × cols) ；cmpM 可选（若提供则显示差值颜色）
    const rows = M.length, cols = M[0].length;
    let header = '<tr><th style="padding:4px 6px;border:1px solid #222;color:#666;font-size:0.7rem;background:#0a0a0a"></th>';
    for (let j = 0; j < cols; j++) {
      header += '<th style="padding:4px 8px;border:1px solid #222;font-size:0.72rem;background:#0a0a0a;color:' + this.actionColors[j] + '">' + this.actionNames[j] + '</th>';
    }
    header += '</tr>';
    let body = '';
    const rowLabels = ['x₁', 'x₂']; // 输入特征
    for (let i = 0; i < rows; i++) {
      body += '<tr><td style="padding:4px 6px;border:1px solid #222;color:#888;font-size:0.72rem;background:#0a0a0a;font-family:JetBrains Mono,monospace">' + rowLabels[i] + '</td>';
      for (let j = 0; j < cols; j++) {
        const v = M[i][j];
        let cellColor = '#e5e5e5';
        let bg = '#0a0a0a';
        if (cmpM) {
          const diff = v - cmpM[i][j];
          const intensity = Math.min(Math.abs(diff) * 2.5, 0.55);
          if (diff > 1e-6) bg = 'rgba(16,185,129,' + (0.12 + intensity) + ')';
          else if (diff < -1e-6) bg = 'rgba(239,68,68,' + (0.12 + intensity) + ')';
        } else {
          const intensity = Math.min(Math.abs(v) * 0.35, 0.5);
          if (v > 0) bg = 'rgba(16,185,129,' + (0.08 + intensity) + ')';
          else if (v < 0) bg = 'rgba(239,68,68,' + (0.08 + intensity) + ')';
        }
        body += '<td style="padding:4px 8px;border:1px solid #222;font-family:JetBrains Mono,monospace;font-size:0.74rem;text-align:center;background:' + bg + ';color:' + cellColor + '">' + v.toFixed(3) + '</td>';
      }
      body += '</tr>';
    }
    return (
      '<div style="margin-bottom:6px;color:' + color + ';font-size:0.78rem;font-weight:600">' + label + '</div>' +
      '<table style="border-collapse:collapse">' + header + body + '</table>'
    );
  }

  _gradArrow(Gt) {
    // 根据 G_t 决定箭头方向与强度
    if (Math.abs(Gt) < 1e-6) {
      return '<span style="color:#666">· 无更新</span>';
    }
    const dir = Gt > 0 ? '↑ 提升概率' : '↓ 降低概率';
    const color = Gt > 0 ? '#4ade80' : '#f87171';
    const arrows = Math.min(Math.ceil(Math.abs(Gt) * 1.5), 5);
    const glyph = Gt > 0 ? '▲' : '▼';
    return '<span style="color:' + color + ';font-weight:600">' + glyph.repeat(arrows) + ' ' + dir + '</span>';
  }

  // ==================== 子面板 ====================
  _renderPolicyNet() {
    // 网络架构图示 + 公式
    return (
      '<div class="section">' +
        '<div class="sec-title">🧠 策略网络结构</div>' +
        '<div class="formula-box">logits = x · W + b  &nbsp;&nbsp;&nbsp;  π_θ(a|s) = softmax(logits)</div>' +
        '<div style="display:flex;align-items:center;justify-content:center;gap:22px;margin:14px 0;flex-wrap:wrap">' +
          '<div style="text-align:center">' +
            '<div style="color:#3B82F6;font-size:0.78rem;margin-bottom:4px">输入 s</div>' +
            '<div style="display:flex;flex-direction:column;gap:4px">' +
              '<div class="node node-in">x₁</div>' +
              '<div class="node node-in">x₂</div>' +
            '</div>' +
            '<div style="color:#888;font-size:0.7rem;margin-top:4px">(2,)</div>' +
          '</div>' +
          '<div style="color:#666;font-size:1.2rem">→ W (2×3) →</div>' +
          '<div style="text-align:center">' +
            '<div style="color:#F59E0B;font-size:0.78rem;margin-bottom:4px">logits</div>' +
            '<div style="display:flex;flex-direction:column;gap:4px">' +
              '<div class="node node-mid">z₀</div>' +
              '<div class="node node-mid">z₁</div>' +
              '<div class="node node-mid">z₂</div>' +
            '</div>' +
            '<div style="color:#888;font-size:0.7rem;margin-top:4px">(3,)</div>' +
          '</div>' +
          '<div style="color:#666;font-size:1rem">softmax →</div>' +
          '<div style="text-align:center">' +
            '<div style="color:#10B981;font-size:0.78rem;margin-bottom:4px">π(a|s)</div>' +
            '<div style="display:flex;flex-direction:column;gap:4px">' +
              '<div class="node node-out" style="color:' + this.actionColors[0] + '">π(a₀)</div>' +
              '<div class="node node-out" style="color:' + this.actionColors[1] + '">π(a₁)</div>' +
              '<div class="node node-out" style="color:' + this.actionColors[2] + '">π(a₂)</div>' +
            '</div>' +
            '<div style="color:#888;font-size:0.7rem;margin-top:4px">概率</div>' +
          '</div>' +
        '</div>' +
        '<div class="two-col" style="margin-top:10px">' +
          '<div>' + this._matrixTable(this.W, 'W (2×3) — 权重矩阵', '#3B82F6') + '</div>' +
          '<div>' +
            '<div style="margin-bottom:6px;color:#8B5CF6;font-size:0.78rem;font-weight:600">b (3,) — 偏置</div>' +
            '<div style="display:flex;gap:6px">' +
              this.b.map((v, j) => '<div class="mini-cell" style="border-color:' + this.actionColors[j] + ';color:' + this.actionColors[j] + '">' + v.toFixed(3) + '</div>').join('') +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  _renderCurrentStateDistributions(withHighlight = true) {
    // 两个状态的当前策略分布
    return (
      '<div class="section">' +
        '<div class="sec-title">📊 当前策略分布 π(a|s)</div>' +
        '<div class="two-col" style="gap:14px">' +
        [0, 1].map(s => {
          const pi = this._pi(s);
          return (
            '<div style="background:#0d1117;border:1px solid #222;border-radius:6px;padding:10px 12px">' +
              '<div style="color:#3B82F6;font-size:0.82rem;font-weight:600;margin-bottom:8px">' + this.stateNames[s] + ' = [' + this.states[s].map(v => v.toFixed(1)).join(', ') + ']</div>' +
              this._probBar(pi) +
            '</div>'
          );
        }).join('') +
        '</div>' +
      '</div>'
    );
  }

  _renderTrajectoryTimeline(focusT = -1) {
    const rows = this.trajectory.map((tr, t) => {
      const pi = this._pi(tr.s);
      const ac = this.actionColors[tr.a];
      const Gt = this.G[t];
      const focused = t === focusT;
      return (
        '<div class="traj-row" style="' + (focused ? 'border-color:#3B82F6;background:#0f1a2a' : '') + '" onclick="window._pgInstance.goTo(' + (1 + t) + ')">' +
          '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">' +
            '<span style="color:#666;font-size:0.72rem;font-family:JetBrains Mono,monospace">t=' + t + '</span>' +
            '<span style="color:#3B82F6;font-weight:600">' + this.stateNames[tr.s] + '</span>' +
            '<span style="color:#666">→</span>' +
            '<span style="color:' + ac + ';font-weight:600">' + this.actionNames[tr.a] + '</span>' +
            '<span style="color:' + (tr.r > 0 ? '#4ade80' : '#888') + ';font-size:0.78rem;font-family:JetBrains Mono,monospace">r=' + tr.r.toFixed(1) + '</span>' +
            '<span style="margin-left:auto;font-size:0.76rem;color:#8b949e">G<sub>' + t + '</sub> = <span style="color:#a5f3fc;font-family:JetBrains Mono,monospace">' + Gt.toFixed(3) + '</span></span>' +
          '</div>' +
          this._probBar(pi, tr.a) +
        '</div>'
      );
    }).join('');
    return (
      '<div class="section">' +
        '<div class="sec-title">🛤️ 轨迹时间线</div>' +
        '<div style="color:#888;font-size:0.76rem;margin-bottom:8px">点击任一步跳转；被采样的动作高亮。</div>' +
        rows +
      '</div>'
    );
  }

  _renderReturnsPanel() {
    const T = this.trajectory.length;
    let html = '';
    for (let t = 0; t < T; t++) {
      const terms = [];
      for (let k = t; k < T; k++) {
        const coef = Math.pow(this.gamma, k - t);
        terms.push((coef === 1 ? '' : coef.toFixed(3) + '·') + 'r<sub>' + k + '</sub>');
      }
      const numeric = [];
      for (let k = t; k < T; k++) {
        const coef = Math.pow(this.gamma, k - t);
        numeric.push((coef === 1 ? '' : coef.toFixed(3) + '·') + this.trajectory[k].r.toFixed(1));
      }
      html += '<div class="formula-box">G<sub>' + t + '</sub> = ' + terms.join(' + ') +
              ' = ' + numeric.join(' + ') +
              ' = <span style="color:' + (this.G[t] >= 0 ? '#4ade80' : '#f87171') + '">' + this.G[t].toFixed(4) + '</span></div>';
    }
    return (
      '<div class="section">' +
        '<div class="sec-title">Σ 累积折扣回报 G_t</div>' +
        '<div class="formula-box" style="color:#fca5a5">G<sub>t</sub> = Σ<sub>k≥t</sub> γ<sup>k-t</sup> · r<sub>k</sub>   &nbsp;(γ = ' + this.gamma + ')</div>' +
        html +
        '<div style="color:#888;font-size:0.78rem;margin-top:8px">G_t 衡量从时刻 t 起"这条轨迹到底赚了多少"。越大 → 该步采样的动作越值得鼓励。</div>' +
      '</div>'
    );
  }

  _renderGradLogPanel() {
    const T = this.trajectory.length;
    let html = '';
    for (let t = 0; t < T; t++) {
      const tr = this.trajectory[t];
      const pi = this.gradLog[t].pi;
      const gW = this.gradLog[t].gW;
      const gb = this.gradLog[t].gb;
      const ac = this.actionColors[tr.a];
      html += (
        '<div style="background:#0d1117;border:1px solid #222;border-radius:6px;padding:10px 12px;margin-bottom:8px">' +
          '<div style="font-size:0.82rem;margin-bottom:6px">' +
            '<span style="color:#666">t=' + t + '</span> &nbsp;|&nbsp; ' +
            '状态 <span style="color:#3B82F6">' + this.stateNames[tr.s] + '</span>  ' +
            '动作 <span style="color:' + ac + '">' + this.actionNames[tr.a] + '</span>  ' +
            'π(a|s) = <span style="color:' + ac + '">' + pi[tr.a].toFixed(3) + '</span>' +
          '</div>' +
          '<div class="formula-box" style="font-size:0.76rem">' +
            '∇<sub>b</sub> log π = 1<sub>j=' + tr.a + '</sub> − π(j|s) = [' +
            gb.map((v, j) => '<span style="color:' + this.actionColors[j] + '">' + v.toFixed(3) + '</span>').join(', ') +
            ']' +
          '</div>' +
          '<div class="formula-box" style="font-size:0.76rem">' +
            '∇<sub>W</sub> log π = sᵀ ⊗ (1<sub>j=a</sub> − π) ∈ ℝ<sup>(2×3)</sup>  ' +
            '&nbsp;&nbsp;s = [' + this.states[tr.s].map(v => v.toFixed(1)).join(', ') + ']' +
          '</div>' +
          this._matrixTable(gW, '∇_W log π(a|s) at t=' + t, '#F59E0B') +
        '</div>'
      );
    }
    return (
      '<div class="section">' +
        '<div class="sec-title">∇ 对数概率梯度 ∇ log π(a|s)</div>' +
        '<div class="formula-box" style="color:#fca5a5">∇<sub>θ</sub> log π(a|s) —— 用 log 技巧把期望下的梯度变成可采样的形式</div>' +
        html +
      '</div>'
    );
  }

  _renderPolicyGradientPanel() {
    // 展示 g = Σ G_t · ∇log π
    const T = this.trajectory.length;
    const bParts = this.gb.map((v, j) => {
      const s = this.G.map((Gt, t) => Gt.toFixed(3) + '·' + this.gradLog[t].gb[j].toFixed(3)).join(' + ');
      return (
        '<div class="formula-box" style="font-size:0.76rem">' +
          'g<sub>b</sub>[<span style="color:' + this.actionColors[j] + '">' + j + '</span>] = ' + s +
          ' = <span style="color:' + (v >= 0 ? '#4ade80' : '#f87171') + '">' + v.toFixed(4) + '</span>' +
        '</div>'
      );
    }).join('');
    return (
      '<div class="section">' +
        '<div class="sec-title">Σ 策略梯度 g = Σ ∇log π · G_t</div>' +
        '<div class="formula-box" style="color:#fca5a5">∇<sub>θ</sub>J(θ) ≈ Σ<sub>t</sub> ∇<sub>θ</sub> log π(a<sub>t</sub>|s<sub>t</sub>) · G<sub>t</sub></div>' +
        '<div class="two-col">' +
          '<div>' + this._matrixTable(this.gW, 'g_W (策略梯度对 W)', '#F59E0B') + '</div>' +
          '<div>' +
            '<div style="margin-bottom:6px;color:#F59E0B;font-size:0.78rem;font-weight:600">g_b (策略梯度对 b)</div>' +
            bParts +
          '</div>' +
        '</div>' +
        '<div style="color:#888;font-size:0.78rem;margin-top:8px">每一步的 G_t 是放大器：高回报让 ∇log π(a) 得到更大"推力"，把采样动作 a 的概率往上抬。</div>' +
      '</div>'
    );
  }

  _renderUpdatePanel() {
    // 参数更新可视化
    return (
      '<div class="section">' +
        '<div class="sec-title">θ ← θ + α · g  （参数更新）</div>' +
        '<div class="formula-box">W<sub>new</sub> = W<sub>old</sub> + ' + this.alpha + ' · g_W</div>' +
        '<div class="formula-box">b<sub>new</sub> = b<sub>old</sub> + ' + this.alpha + ' · g_b</div>' +
        '<div class="two-col" style="margin-top:10px">' +
          '<div>' + this._matrixTable(this.W_old, 'W (更新前)', '#8b949e') + '</div>' +
          '<div>' + this._matrixTable(this.W, 'W (更新后)', '#4ade80', this.W_old) + '</div>' +
        '</div>' +
        '<div style="color:#888;font-size:0.78rem;margin-top:10px">颜色说明：<span style="color:#4ade80">绿色=数值上升</span>，<span style="color:#f87171">红色=数值下降</span>。</div>' +
      '</div>'
    );
  }

  _renderBeforeAfter() {
    return (
      '<div class="section">' +
        '<div class="sec-title">🔍 策略分布 Before / After</div>' +
        '<div class="two-col" style="gap:14px">' +
          [0, 1].map(s => {
            const piOld = this._pi(s, this.W_old, this.b_old);
            const piNew = this._pi(s);
            const sv = this.states[s];
            // 找到该状态在轨迹中被执行的动作
            const actsHere = this.trajectory.filter(tr => tr.s === s);
            return (
              '<div style="background:#0d1117;border:1px solid #222;border-radius:6px;padding:12px">' +
                '<div style="color:#3B82F6;font-size:0.84rem;font-weight:600;margin-bottom:8px">' + this.stateNames[s] + ' = [' + sv.map(v => v.toFixed(1)).join(', ') + ']</div>' +
                '<div style="display:flex;flex-direction:column;gap:10px">' +
                  '<div>' + this._probBar(piOld, -1, '更新前 π_old') + '</div>' +
                  '<div>' + this._probBar(piNew, -1, '更新后 π_new') + '</div>' +
                '</div>' +
                '<div style="font-size:0.74rem;color:#8b949e;margin-top:8px">Δπ = ' +
                  piNew.map((p, j) => {
                    const d = p - piOld[j];
                    const col = d > 0 ? '#4ade80' : (d < 0 ? '#f87171' : '#888');
                    const sign = d >= 0 ? '+' : '';
                    return '<span style="color:' + this.actionColors[j] + '">' + this.actionNames[j] + '</span>=<span style="color:' + col + '">' + sign + (d * 100).toFixed(2) + '%</span>';
                  }).join(' &nbsp; ') +
                '</div>' +
                (actsHere.length > 0 ? '<div style="font-size:0.72rem;color:#888;margin-top:6px">本轨迹中采样的动作：' +
                    actsHere.map(tr => '<span style="color:' + this.actionColors[tr.a] + '">' + this.actionNames[tr.a] + '</span> (G=' + this.G[this.trajectory.indexOf(tr)].toFixed(2) + ')').join(', ') +
                  '</div>' : '') +
              '</div>'
            );
          }).join('') +
        '</div>' +
        '<div style="color:#888;font-size:0.78rem;margin-top:10px">' +
        '可以看到：<b>G_t 为正的动作</b>（例如 t=2 a2 → G=2.0）概率被<b>推高</b>；G_t=0 的步不产生更新；若某步 G_t 为负，对应动作概率会被压低。这正是策略梯度的"试错 → 强化"机制。' +
        '</div>' +
      '</div>'
    );
  }

  _renderStepDetail() {
    const st = this.currentStep;
    const T = this.trajectory.length;

    if (st === 0) {
      return this._renderPolicyNet() + this._renderCurrentStateDistributions();
    }
    if (st >= 1 && st <= T) {
      const t = st - 1;
      const tr = this.trajectory[t];
      const pi = this._pi(tr.s);
      const ac = this.actionColors[tr.a];
      const detail = (
        '<div class="section">' +
          '<div class="sec-title">🎲 访问轨迹 t=' + t + '</div>' +
          '<div class="formula-box">s = <span style="color:#3B82F6">' + this.stateNames[tr.s] + '</span> = [' + this.states[tr.s].map(v => v.toFixed(1)).join(', ') + ']</div>' +
          '<div class="formula-box">logits = s·W + b = [' + this._logits(tr.s, this.W, this.b).map(v => v.toFixed(3)).join(', ') + ']</div>' +
          '<div class="formula-box">π(·|s) = softmax(logits) = [' + pi.map(v => v.toFixed(4)).join(', ') + ']</div>' +
          '<div class="formula-box">采样动作 a = <span style="color:' + ac + '">' + this.actionNames[tr.a] + '</span>  ( π(a|s) = ' + pi[tr.a].toFixed(4) + ' )</div>' +
          '<div class="formula-box">即时奖励 r<sub>' + t + '</sub> = <span style="color:' + (tr.r > 0 ? '#4ade80' : '#f87171') + '">' + tr.r.toFixed(2) + '</span></div>' +
          '<div style="margin-top:10px">' + this._probBar(pi, tr.a, 'π(·|' + this.stateNames[tr.s] + ')') + '</div>' +
        '</div>'
      );
      return detail + this._renderTrajectoryTimeline(t);
    }
    if (st === T + 1) return this._renderReturnsPanel() + this._renderTrajectoryTimeline();
    if (st === T + 2) return this._renderGradLogPanel();
    if (st === T + 3) return this._renderPolicyGradientPanel();
    if (st === T + 4) return this._renderUpdatePanel();
    if (st === T + 5) return this._renderBeforeAfter();
    return '';
  }

  // ==================== 主渲染 ====================
  render() {
    if (!this.container) return;
    window._pgInstance = this;

    const st = this.currentStep;
    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._pgInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._pgInstance.play()">▶ 播放</button>';

    const stepBar = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff;border-color:#3B82F6' : (i < st ? 'background:#1e3a5f;color:#60a5fa' : 'background:#1a1a1a;color:#888');
      return '<div class="step-pill" onclick="window._pgInstance.goTo(' + i + ')" style="' + active + '">' + s + '</div>';
    }).join('');

    this.container.innerHTML =
      '<div class="pg-viz">' +
        '<style>' +
          '.pg-viz{font-family:Inter,sans-serif;color:#e5e5e5;background:#0a0a0a;padding:18px}' +
          '.pg-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;font-size:0.85rem}' +
          '.pg-viz .ctrl-btn:hover{background:#252525;border-color:#3B82F6}' +
          '.pg-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.pg-viz .speed-select{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 10px;border-radius:6px;font-size:0.85rem}' +
          '.pg-viz .formula-box{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.pg-viz .section{background:#111217;border-radius:8px;padding:14px 16px;border:1px solid #21262d;margin-bottom:14px}' +
          '.pg-viz .sec-title{font-weight:700;margin-bottom:10px;color:#fff;font-size:0.95rem}' +
          '.pg-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '@media(max-width:900px){.pg-viz .two-col{grid-template-columns:1fr}}' +
          '.pg-viz .edu-panel{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin-bottom:16px}' +
          '.pg-viz .edu-title{font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #21262d}' +
          '.pg-viz .edu-section{margin-bottom:14px}' +
          '.pg-viz .edu-section-title{font-size:0.84rem;font-weight:600;color:#58a6ff;margin-bottom:4px}' +
          '.pg-viz .edu-text{font-size:0.86rem;color:#8b949e;line-height:1.75}' +
          '.pg-viz .step-pill{padding:6px 10px;border:1px solid #333;border-radius:14px;font-size:0.72rem;cursor:pointer;transition:all 0.15s;white-space:nowrap}' +
          '.pg-viz .step-pill:hover{border-color:#3B82F6}' +
          '.pg-viz .node{width:40px;height:28px;display:flex;align-items:center;justify-content:center;border:1px solid #333;border-radius:5px;font-family:JetBrains Mono,monospace;font-size:0.74rem;background:#0a0a0a}' +
          '.pg-viz .node-in{color:#3B82F6}' +
          '.pg-viz .node-mid{color:#F59E0B}' +
          '.pg-viz .node-out{min-width:56px}' +
          '.pg-viz .traj-row{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:10px 12px;margin-bottom:8px;cursor:pointer;transition:all 0.15s}' +
          '.pg-viz .traj-row:hover{border-color:#3B82F6}' +
          '.pg-viz .mini-cell{min-width:54px;padding:6px 8px;border:1px solid #333;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.76rem;text-align:center;background:#0a0a0a}' +
        '</style>' +

        // 标题
        '<div style="font-size:1.5rem;font-weight:700;margin-bottom:2px">Policy Gradient (REINFORCE)</div>' +
        '<div style="color:#8b949e;margin-bottom:14px;font-size:0.88rem">用策略梯度直接优化参数化策略 π<sub>θ</sub>(a|s)，使期望累积回报最大化。</div>' +

        // 教育面板
        '<div class="edu-panel">' +
          '<div class="edu-title">📖 零基础讲解：REINFORCE 为什么这样更新？</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🎯 核心思路：直接学策略</div>' +
            '<div class="edu-text">Q-Learning 学的是"值函数"，然后贪心得到策略；REINFORCE 则直接把<b>策略</b>参数化为神经网络 π<sub>θ</sub>(a|s)，通过梯度上升让"让回报更高的动作被选中的概率"变大。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🧩 log-probability 技巧：为什么是 ∇log π</div>' +
            '<div class="edu-text">目标 J(θ) = E<sub>π</sub>[G] 的梯度没法直接对期望求。用恒等式 ∇π = π·∇log π：<br>' +
            '∇J = E[∇log π · G]。<br>这样就能<b>用蒙特卡洛采样</b>来估计梯度：跑一条轨迹，把 ∇log π(a|s)·G 累加即可。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🔁 为什么用 G_t 加权？</div>' +
            '<div class="edu-text">高 G_t 表示"这一步之后整条轨迹赚得多"——就把这一刻采样的动作概率往上推得更猛；反之，G_t 小或为负，就要把这个动作往下压。本质是<b>good action credit assignment</b>。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">⚠️ 高方差问题 → 基线 / Advantage</div>' +
            '<div class="edu-text">蒙特卡洛 G_t 方差很大。减去基线 b(s)（不影响期望）能大幅降低方差：<br>' +
            '∇J = E[∇log π · (G - b(s))]。常见选择：b(s) = V(s)，于是得到 <b>Advantage A(s,a) = G - V(s)</b>——这正是 Actor-Critic / PPO 的起点。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">📜 REINFORCE vs Actor-Critic 预告</div>' +
            '<div class="edu-text">REINFORCE：蒙特卡洛回报 → 方差高、仅能回合后更新；<br>' +
            'Actor-Critic：用 V(s) 作为基线 → 方差低、可在线更新；<br>' +
            'PPO：在 Actor-Critic 上加"信赖域"约束，避免一步更新过大导致策略崩溃。</div>' +
          '</div>' +
        '</div>' +

        // 控制栏
        '<div class="section">' +
          '<div class="sec-title">🎮 控制</div>' +
          '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">' +
            '<button class="ctrl-btn" onclick="window._pgInstance.reset()">↻ 重置</button>' +
            '<button class="ctrl-btn" onclick="window._pgInstance.stepBack()">⏮ 上一步</button>' +
            playBtn +
            '<button class="ctrl-btn" onclick="window._pgInstance.stepForward()">⏭ 下一步</button>' +
            '<select class="speed-select" onchange="window._pgInstance.setSpeed(this.value)">' +
              '<option value="0.5">0.5×</option>' +
              '<option value="1" selected>1×</option>' +
              '<option value="1.5">1.5×</option>' +
              '<option value="2">2×</option>' +
              '<option value="3">3×</option>' +
            '</select>' +
            '<span style="color:#666;font-size:0.78rem;margin-left:8px">γ=' + this.gamma + '   α=' + this.alpha + '</span>' +
          '</div>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepBar + '</div>' +
        '</div>' +

        // 总公式
        '<div class="formula-box" style="font-size:0.86rem">∇<sub>θ</sub>J(θ) ≈ Σ<sub>t</sub> ∇<sub>θ</sub> log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) · G<sub>t</sub>   ;   θ ← θ + α · ∇J</div>' +

        // 当前步骤标题
        '<div class="section">' +
          '<div class="sec-title">🧭 当前步骤：' + this.STEPS[st] + '</div>' +
          '<div style="color:#888;font-size:0.8rem">（步 ' + (st + 1) + ' / ' + this.STEPS.length + '）</div>' +
        '</div>' +

        // 当前步骤详情
        this._renderStepDetail() +

      '</div>';

    window._pgInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._pgInstance === this) {
      try { delete window._pgInstance; } catch (e) { window._pgInstance = null; }
    }
    this.container = null;
  }
}

window.PolicyGradient = PolicyGradient;
