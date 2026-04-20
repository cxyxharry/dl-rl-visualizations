// DQN (Deep Q-Network) 可视化
// Q-网络: Q(s, a) = s · W_q[:, a]    (2 状态 × 2 动作, 线性近似)
// TD 目标: y = r + γ · max_{a'} Q_target(s', a')
// 损失: L = (y - Q(s,a))²     半梯度下降更新 W_q
// 软更新: W_target ← τ·W_q + (1-τ)·W_target
class DQN {
  constructor() {
    this.container = null;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // ---------- 超参数 ----------
    this.gamma = 0.9;
    this.alpha = 0.3;     // 学习率（偏大，便于观察）
    this.tau = 0.1;       // 软更新系数（偏大，便于观察目标网络变化）

    // ---------- 状态/动作 ----------
    // 2 个状态，2 维特征；2 个动作
    this.states = [
      [1.0, 0.5],  // s0
      [0.3, 0.7]   // s1
    ];
    this.stateNames = ['s0', 's1'];
    this.actionNames = ['a0', 'a1'];
    this.actionColors = ['#3B82F6', '#F97316'];

    // ---------- Q 网络权重 W_q : (2, 2)  每个动作一列 ----------
    // 为方便线性 Q(s,a) = s · W_q[:, a] 使用 2 维特征
    this.W_q_init = [
      [ 0.5, -0.3],
      [-0.2,  0.7]
    ];
    this.W_target_init = [
      [ 0.5, -0.3],
      [-0.2,  0.7]
    ];
    this.W_q = this._clone2D(this.W_q_init);
    this.W_target = this._clone2D(this.W_target_init);

    // ---------- 经验回放缓冲区 ----------
    // (s, a, r, s', done)
    this.buffer = [
      { s: 0, a: 0, r: 0.0, s_next: 1, done: false },
      { s: 1, a: 1, r: 1.0, s_next: 0, done: false },
      { s: 0, a: 1, r: 0.5, s_next: 1, done: false },
      { s: 1, a: 0, r: 0.2, s_next: 0, done: false },
      { s: 0, a: 0, r: 2.0, s_next: 0, done: true  },
      { s: 1, a: 1, r: 1.0, s_next: 1, done: false },
      { s: 0, a: 1, r: 0.0, s_next: 0, done: false }
    ];

    // ---------- 训练状态 ----------
    this.currentSample = 0;
    this.iteration = 0;     // 训练迭代计数
    this.lossHistory = [];  // 损失曲线
    this.updateEvery = 2;   // 每 N 次 Q 更新做一次 target 软更新

    // ---------- 步骤定义 ----------
    // 0: Q / Q_target 初始化
    // 1: 从 buffer 中随机采样一条 experience
    // 2: 计算当前 Q(s,a)
    // 3: 计算 TD 目标 y
    // 4: 计算 TD 误差 δ = y - Q(s,a)，显示损失
    // 5: 执行 SGD 更新 W_q
    // 6: (每 N 步) 软更新 W_target
    // 7: 总结本轮迭代 + 进入下一轮
    this.STEPS = [
      '① 初始化 Q、Q_target',
      '② 从 Replay Buffer 采样',
      '③ 计算 Q(s,a)',
      '④ 计算 TD 目标 y',
      '⑤ 计算 TD 误差 δ 与损失',
      '⑥ SGD 更新 Q 网络',
      '⑦ 软更新 Q_target',
      '⑧ 迭代汇总 / 下一轮'
    ];
    this.currentStep = 0;

    // 当前缓存计算
    this._computed = this._compute();
  }

  // ==================== 接口 ====================
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this.iteration = 0;
    this.lossHistory = [];
    this.currentSample = 0;
    this.W_q = this._clone2D(this.W_q_init);
    this.W_target = this._clone2D(this.W_target_init);
    this._computed = this._compute();
    this.render();
    return this;
  }

  reset() {
    this.isPlaying = false;
    clearTimeout(this.timer);
    this.currentStep = 0;
    this.iteration = 0;
    this.lossHistory = [];
    this.currentSample = 0;
    this.W_q = this._clone2D(this.W_q_init);
    this.W_target = this._clone2D(this.W_target_init);
    this._computed = this._compute();
    this.render();
  }

  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  setSpeed(s) { this.speed = Number(s) || 1; }

  stepForward() {
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this._applyStepEffects();
      this.render();
    } else {
      // 已到末步 -> 进入下一轮迭代：重新采样、从步骤 1 开始
      this._nextIteration();
      this.render();
    }
  }

  stepBack() {
    // 只允许在当前迭代内退：权重已修改，不做历史回放
    if (this.currentStep > 0) {
      this.currentStep--;
      this.render();
    }
  }

  goTo(i) {
    // 跳到当前迭代的某一步（不跨迭代）
    const target = Math.max(0, Math.min(i, this.STEPS.length - 1));
    // 若往后跳，逐步 apply
    while (this.currentStep < target) {
      this.currentStep++;
      this._applyStepEffects();
    }
    while (this.currentStep > target) {
      this.currentStep--;
    }
    this.render();
  }

  // ==================== 核心计算 ====================
  _clone2D(M) { return M.map(r => r.slice()); }

  _Q(sIdx, aIdx, W) {
    const s = this.states[sIdx];
    const w = W || this.W_q;
    return s[0] * w[0][aIdx] + s[1] * w[1][aIdx];
  }

  _Qrow(sIdx, W) { return [this._Q(sIdx, 0, W), this._Q(sIdx, 1, W)]; }

  _compute() {
    const e = this.buffer[this.currentSample];
    const q_sa = this._Q(e.s, e.a, this.W_q);
    const q_next = this._Qrow(e.s_next, this.W_target);
    const maxQ = Math.max(q_next[0], q_next[1]);
    const argmax = q_next[0] >= q_next[1] ? 0 : 1;
    const y = e.done ? e.r : e.r + this.gamma * maxQ;
    const delta = y - q_sa;
    const loss = 0.5 * delta * delta;
    // 半梯度：∂L/∂W_q[:, a] = - δ · s，更新 W_q[:, a] += α · δ · s
    const s = this.states[e.s];
    const gradCol = [s[0] * delta, s[1] * delta];
    return { e, q_sa, q_next, maxQ, argmax, y, delta, loss, gradCol };
  }

  _applyStepEffects() {
    const st = this.currentStep;
    if (st === 5) {
      // SGD 更新
      const { e, gradCol } = this._computed;
      this.W_q[0][e.a] += this.alpha * gradCol[0];
      this.W_q[1][e.a] += this.alpha * gradCol[1];
      this.lossHistory.push(this._computed.loss);
      // 重新计算 q_sa（基于新权重）仅用于展示
      this._computed.q_sa_after = this._Q(e.s, e.a, this.W_q);
    } else if (st === 6) {
      // 软更新 W_target（每次都软更新；τ 足够小即可看作"周期性"）
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          this.W_target[i][j] = this.tau * this.W_q[i][j] + (1 - this.tau) * this.W_target[i][j];
        }
      }
    }
  }

  _nextIteration() {
    this.iteration++;
    this.currentSample = (this.currentSample + 1 + Math.floor(Math.random() * (this.buffer.length - 1))) % this.buffer.length;
    this._computed = this._compute();
    this.currentStep = 1;
    // 自动进入"已采样"
  }

  // ==================== 播放器 ====================
  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this._applyStepEffects();
      this.render();
      this.timer = setTimeout(() => this._auto(), 1500 / this.speed);
    } else {
      // 自动进入下一轮迭代
      this._nextIteration();
      this.render();
      this.timer = setTimeout(() => this._auto(), 1800 / this.speed);
    }
  }

  // ==================== 渲染辅助 ====================
  _qTable(W, label, color, highlight, useDiff) {
    // highlight: { s, a } 或 null
    let html = '<div style="margin-bottom:6px;color:' + color + ';font-size:0.78rem;font-weight:600">' + label + '</div>';
    html += '<table style="border-collapse:collapse">';
    html += '<tr><th style="padding:4px 8px;border:1px solid #222;color:#666;font-size:0.72rem;background:#0a0a0a"></th>';
    for (let j = 0; j < 2; j++) {
      html += '<th style="padding:4px 8px;border:1px solid #222;color:' + this.actionColors[j] + ';font-size:0.72rem;background:#0a0a0a">' + this.actionNames[j] + '</th>';
    }
    html += '</tr>';
    for (let i = 0; i < 2; i++) {
      html += '<tr><td style="padding:4px 8px;border:1px solid #222;color:#3B82F6;font-size:0.72rem;background:#0a0a0a;font-family:JetBrains Mono,monospace">' + this.stateNames[i] + '</td>';
      for (let j = 0; j < 2; j++) {
        // 实际展示 Q(s_i, a_j) = s_i · W[:, a_j]
        const q = this._Q(i, j, W);
        const isHL = highlight && highlight.s === i && highlight.a === j;
        let bg = '#0a0a0a';
        if (useDiff) {
          const qOnline = this._Q(i, j, this.W_q);
          const diff = q - qOnline;
          const intensity = Math.min(Math.abs(diff) * 2, 0.5);
          if (diff > 1e-6) bg = 'rgba(245,158,11,' + (0.1 + intensity) + ')';
          else if (diff < -1e-6) bg = 'rgba(139,92,246,' + (0.1 + intensity) + ')';
        } else {
          const intensity = Math.min(Math.abs(q) * 0.3, 0.5);
          bg = q >= 0 ? 'rgba(16,185,129,' + (0.08 + intensity) + ')' : 'rgba(239,68,68,' + (0.08 + intensity) + ')';
        }
        const border = isHL ? '2px solid #F97316' : '1px solid #222';
        const shadow = isHL ? 'box-shadow:0 0 8px rgba(249,115,22,0.5);' : '';
        html += '<td style="padding:5px 8px;border:' + border + ';' + shadow +
                'background:' + bg + ';font-family:JetBrains Mono,monospace;font-size:0.76rem;text-align:center;min-width:60px;color:#e5e5e5">' +
                q.toFixed(3) + '</td>';
      }
      html += '</tr>';
    }
    html += '</table>';
    return html;
  }

  _weightTable(W, label, color, cmpW) {
    let html = '<div style="margin-bottom:6px;color:' + color + ';font-size:0.78rem;font-weight:600">' + label + '</div>';
    html += '<table style="border-collapse:collapse">';
    html += '<tr><th style="padding:4px 8px;border:1px solid #222;color:#666;font-size:0.7rem;background:#0a0a0a"></th>';
    for (let j = 0; j < 2; j++) {
      html += '<th style="padding:4px 8px;border:1px solid #222;color:' + this.actionColors[j] + ';font-size:0.7rem;background:#0a0a0a">' + this.actionNames[j] + '</th>';
    }
    html += '</tr>';
    const rowLabels = ['x₁', 'x₂'];
    for (let i = 0; i < 2; i++) {
      html += '<tr><td style="padding:4px 8px;border:1px solid #222;color:#888;font-size:0.7rem;background:#0a0a0a;font-family:JetBrains Mono,monospace">' + rowLabels[i] + '</td>';
      for (let j = 0; j < 2; j++) {
        const v = W[i][j];
        let bg = '#0a0a0a';
        if (cmpW) {
          const diff = v - cmpW[i][j];
          const intensity = Math.min(Math.abs(diff) * 3, 0.55);
          if (diff > 1e-6) bg = 'rgba(16,185,129,' + (0.12 + intensity) + ')';
          else if (diff < -1e-6) bg = 'rgba(239,68,68,' + (0.12 + intensity) + ')';
        } else {
          const intensity = Math.min(Math.abs(v) * 0.35, 0.5);
          if (v > 0) bg = 'rgba(16,185,129,' + (0.08 + intensity) + ')';
          else if (v < 0) bg = 'rgba(239,68,68,' + (0.08 + intensity) + ')';
        }
        html += '<td style="padding:5px 8px;border:1px solid #222;background:' + bg + ';font-family:JetBrains Mono,monospace;font-size:0.74rem;text-align:center;min-width:58px;color:#e5e5e5">' + v.toFixed(3) + '</td>';
      }
      html += '</tr>';
    }
    html += '</table>';
    return html;
  }

  _renderReplayBuffer() {
    const activeIdx = this.currentSample;
    const cards = this.buffer.map((e, i) => {
      const isActive = i === activeIdx && this.currentStep >= 1;
      const bg = isActive ? '#1a2a3a' : '#0d1117';
      const border = isActive ? '2px solid #3B82F6' : '1px solid #222';
      const shadow = isActive ? 'box-shadow:0 0 12px rgba(59,130,246,0.4);' : '';
      return (
        '<div onclick="window._dqnInstance._jumpSample(' + i + ')" ' +
             'style="background:' + bg + ';border:' + border + ';' + shadow +
             'border-radius:6px;padding:8px 10px;margin-bottom:5px;cursor:pointer;transition:all 0.15s">' +
          '<div style="display:flex;gap:8px;align-items:center;font-size:0.76rem;font-family:JetBrains Mono,monospace;flex-wrap:wrap">' +
            '<span style="color:#666;font-size:0.7rem">#' + i + '</span>' +
            '<span style="color:#3B82F6;font-weight:600">' + this.stateNames[e.s] + '</span>' +
            '<span style="color:#666">·</span>' +
            '<span style="color:' + this.actionColors[e.a] + ';font-weight:600">' + this.actionNames[e.a] + '</span>' +
            '<span style="color:#666">·</span>' +
            '<span style="color:' + (e.r > 0 ? '#4ade80' : '#888') + '">r=' + e.r.toFixed(1) + '</span>' +
            '<span style="color:#666">→</span>' +
            '<span style="color:#8B5CF6;font-weight:600">' + this.stateNames[e.s_next] + '</span>' +
            (e.done ? '<span style="color:#EF4444;font-size:0.66rem;margin-left:auto;padding:1px 6px;border:1px solid #EF4444;border-radius:8px">done</span>' : '') +
          '</div>' +
        '</div>'
      );
    }).join('');
    return (
      '<div class="section">' +
        '<div class="sec-title">📦 Replay Buffer（经验回放缓冲）</div>' +
        '<div style="color:#888;font-size:0.76rem;margin-bottom:8px">共 ' + this.buffer.length + ' 条经验，蓝色边框 = 当前采样。点击任一条可选中。</div>' +
        cards +
      '</div>'
    );
  }

  _renderLossChart() {
    // 简单 SVG 折线图
    const hist = this.lossHistory;
    if (hist.length === 0) {
      return (
        '<div class="section">' +
          '<div class="sec-title">📉 损失曲线 L = ½(y − Q(s,a))²</div>' +
          '<div style="color:#666;font-size:0.8rem;text-align:center;padding:20px">（尚未开始训练）</div>' +
        '</div>'
      );
    }
    const W = 340, H = 120, pad = 22;
    const maxL = Math.max.apply(null, hist);
    const minL = 0;
    const n = hist.length;
    const xOf = i => pad + (n <= 1 ? 0 : i * (W - 2 * pad) / (n - 1));
    const yOf = v => H - pad - ((v - minL) / (maxL - minL + 1e-9)) * (H - 2 * pad);
    const pts = hist.map((v, i) => xOf(i) + ',' + yOf(v)).join(' ');
    const dots = hist.map((v, i) =>
      '<circle cx="' + xOf(i) + '" cy="' + yOf(v) + '" r="3" fill="#F97316"></circle>'
    ).join('');
    // y 轴网格
    const grid = [0, 0.25, 0.5, 0.75, 1].map(t => {
      const y = H - pad - t * (H - 2 * pad);
      const v = minL + t * (maxL - minL);
      return '<line x1="' + pad + '" x2="' + (W - pad) + '" y1="' + y + '" y2="' + y + '" stroke="#1a1a1a" stroke-width="1"/>' +
             '<text x="4" y="' + (y + 3) + '" fill="#444" font-size="9" font-family="JetBrains Mono,monospace">' + v.toFixed(2) + '</text>';
    }).join('');
    return (
      '<div class="section">' +
        '<div class="sec-title">📉 损失曲线 L = ½(y − Q(s,a))²</div>' +
        '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;background:#0d1117;border:1px solid #21262d;border-radius:6px">' +
          grid +
          '<polyline points="' + pts + '" fill="none" stroke="#F59E0B" stroke-width="1.5"/>' +
          dots +
          '<text x="' + (W - 4) + '" y="' + (H - 4) + '" text-anchor="end" fill="#666" font-size="10">iter</text>' +
        '</svg>' +
        '<div style="color:#888;font-size:0.76rem;margin-top:6px">迭代 ' + n + ' 次 / 最近损失 = <span style="color:#F59E0B">' + hist[hist.length - 1].toFixed(4) + '</span> / 最大 = ' + maxL.toFixed(4) + '</div>' +
      '</div>'
    );
  }

  _renderDriftIndicator() {
    // 计算 online 与 target 之间的 L2 距离
    let d2 = 0;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        d2 += Math.pow(this.W_q[i][j] - this.W_target[i][j], 2);
      }
    }
    const dist = Math.sqrt(d2);
    const maxDist = 1.0;
    const pct = Math.min(dist / maxDist, 1);
    const barColor = pct < 0.2 ? '#4ade80' : (pct < 0.5 ? '#F59E0B' : '#EF4444');
    return (
      '<div class="section">' +
        '<div class="sec-title">🎯 目标网络滞后指示（Target Drift）</div>' +
        '<div class="formula-box" style="color:#a5f3fc;font-size:0.78rem">‖W<sub>q</sub> − W<sub>target</sub>‖<sub>2</sub> = <span style="color:' + barColor + '">' + dist.toFixed(4) + '</span></div>' +
        '<div style="background:#0d1117;border:1px solid #21262d;border-radius:4px;height:14px;overflow:hidden;margin-top:8px">' +
          '<div style="width:' + (pct * 100).toFixed(1) + '%;height:100%;background:' + barColor + ';transition:width 0.3s"></div>' +
        '</div>' +
        '<div style="color:#888;font-size:0.74rem;margin-top:6px">绿=同步（drift 小）/ 黄=中等漂移 / 红=大漂移。τ=' + this.tau + ' 越小 → drift 越稳定，训练越稳但追赶越慢。</div>' +
      '</div>'
    );
  }

  // 跳转到某条经验（保留权重）
  _jumpSample(i) {
    this.currentSample = i;
    this._computed = this._compute();
    if (this.currentStep === 0) this.currentStep = 1;
    this.render();
  }

  // ==================== 子面板：每步详情 ====================
  _renderStepDetail() {
    const st = this.currentStep;
    const c = this._computed;
    const e = c.e;

    // 公共信息：当前正在处理哪条经验
    const expHeader = (
      '<div class="formula-box" style="font-size:0.8rem">' +
        '当前经验 #' + this.currentSample + ' : ' +
        '(s=<span style="color:#3B82F6">' + this.stateNames[e.s] + '</span>, ' +
        'a=<span style="color:' + this.actionColors[e.a] + '">' + this.actionNames[e.a] + '</span>, ' +
        'r=<span style="color:#4ade80">' + e.r.toFixed(2) + '</span>, ' +
        's\'=<span style="color:#8B5CF6">' + this.stateNames[e.s_next] + '</span>' +
        (e.done ? ', <span style="color:#EF4444">done=true</span>' : '') + ')' +
      '</div>'
    );

    if (st === 0) {
      return (
        '<div class="section">' +
          '<div class="sec-title">① 初始化网络</div>' +
          '<div class="formula-box">Q(s, a) = s · W<sub>q</sub>[:, a]  —— 用线性函数近似 Q 值（真实 DQN 用深网络，这里简化）</div>' +
          '<div class="formula-box">Q<sub>target</sub> 与 Q 初始共享权重，稍后通过<strong>软更新</strong>缓慢跟随 Q。</div>' +
          '<div class="two-col">' +
            '<div>' + this._weightTable(this.W_q, 'W_q (Online)', '#3B82F6') + '</div>' +
            '<div>' + this._weightTable(this.W_target, 'W_target (Target)', '#8B5CF6') + '</div>' +
          '</div>' +
        '</div>'
      );
    }
    if (st === 1) {
      return (
        '<div class="section">' +
          '<div class="sec-title">② 从 Replay Buffer 随机采样</div>' +
          expHeader +
          '<div style="color:#888;font-size:0.8rem;margin-top:6px">随机采样打破样本时序相关性；同一经验可被多次复用 → 提升数据效率。</div>' +
        '</div>'
      );
    }
    if (st === 2) {
      return (
        '<div class="section">' +
          '<div class="sec-title">③ 计算 Q(s, a)（Online 网络）</div>' +
          expHeader +
          '<div class="formula-box">Q(' + this.stateNames[e.s] + ', ' + this.actionNames[e.a] + ') = s · W<sub>q</sub>[:, ' + e.a + '] = ' +
            this.states[e.s][0].toFixed(1) + '×' + this.W_q[0][e.a].toFixed(3) + ' + ' +
            this.states[e.s][1].toFixed(1) + '×' + this.W_q[1][e.a].toFixed(3) +
            ' = <span style="color:#3B82F6">' + c.q_sa.toFixed(4) + '</span></div>' +
          this._qTable(this.W_q, 'Q (Online)', '#3B82F6', { s: e.s, a: e.a }, false) +
        '</div>'
      );
    }
    if (st === 3) {
      if (e.done) {
        return (
          '<div class="section">' +
            '<div class="sec-title">④ 计算 TD 目标 y</div>' +
            expHeader +
            '<div class="formula-box">该经验 <b>done=true</b> → 终态无未来折扣</div>' +
            '<div class="formula-box">y = r = <span style="color:#a5f3fc">' + c.y.toFixed(4) + '</span></div>' +
          '</div>'
        );
      }
      return (
        '<div class="section">' +
          '<div class="sec-title">④ 计算 TD 目标 y = r + γ·max Q_target(s\', ·)</div>' +
          expHeader +
          '<div class="formula-box">Q<sub>target</sub>(' + this.stateNames[e.s_next] + ', :) = [' +
            c.q_next.map((v, j) => '<span style="color:' + this.actionColors[j] + '">' + v.toFixed(4) + '</span>').join(', ') + ']</div>' +
          '<div class="formula-box">max<sub>a\'</sub> Q<sub>target</sub>(s\', a\') = <span style="color:' + this.actionColors[c.argmax] + '">' + c.maxQ.toFixed(4) + '</span>   (argmax = ' + this.actionNames[c.argmax] + ')</div>' +
          '<div class="formula-box">y = ' + e.r.toFixed(2) + ' + ' + this.gamma + ' × ' + c.maxQ.toFixed(4) + ' = <span style="color:#a5f3fc">' + c.y.toFixed(4) + '</span></div>' +
          this._qTable(this.W_target, 'Q_target（取自 Target 网络）', '#8B5CF6', { s: e.s_next, a: c.argmax }, false) +
        '</div>'
      );
    }
    if (st === 4) {
      const deltaColor = c.delta >= 0 ? '#4ade80' : '#f87171';
      return (
        '<div class="section">' +
          '<div class="sec-title">⑤ TD 误差 δ = y − Q(s,a)  &  损失 L = ½δ²</div>' +
          expHeader +
          '<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:10px 0">' +
            '<div style="flex:1;min-width:160px;background:#0d1117;border:1px solid #8B5CF6;border-radius:6px;padding:10px 12px;text-align:center">' +
              '<div style="color:#8B5CF6;font-size:0.78rem;margin-bottom:4px">目标 y</div>' +
              '<div style="font-family:JetBrains Mono,monospace;font-size:1.2rem;color:#a5f3fc">' + c.y.toFixed(4) + '</div>' +
            '</div>' +
            '<div style="font-size:1.3rem;color:' + deltaColor + '">' + (c.delta >= 0 ? '↑' : '↓') + '</div>' +
            '<div style="flex:1;min-width:160px;background:#0d1117;border:1px solid #3B82F6;border-radius:6px;padding:10px 12px;text-align:center">' +
              '<div style="color:#3B82F6;font-size:0.78rem;margin-bottom:4px">估计 Q(s,a)</div>' +
              '<div style="font-family:JetBrains Mono,monospace;font-size:1.2rem;color:#93c5fd">' + c.q_sa.toFixed(4) + '</div>' +
            '</div>' +
          '</div>' +
          '<div class="formula-box">δ = y − Q(s,a) = ' + c.y.toFixed(4) + ' − ' + c.q_sa.toFixed(4) + ' = <span style="color:' + deltaColor + '">' + c.delta.toFixed(4) + '</span></div>' +
          '<div class="formula-box">L = ½ δ² = <span style="color:#F59E0B">' + c.loss.toFixed(5) + '</span></div>' +
          '<div style="color:#888;font-size:0.78rem;margin-top:6px">δ 正 → 当前 Q 低估，应上调；δ 负 → 当前 Q 高估，应下调。</div>' +
        '</div>'
      );
    }
    if (st === 5) {
      const W_before = this._clone2D(this.W_q);
      // 此时本步的更新已经应用 => W_q 就是新值
      const s = this.states[e.s];
      return (
        '<div class="section">' +
          '<div class="sec-title">⑥ SGD 更新 W_q（半梯度下降）</div>' +
          expHeader +
          '<div class="formula-box">∂L / ∂W<sub>q</sub>[:, a] = − δ · s  （半梯度：对 y 不求导）</div>' +
          '<div class="formula-box">W<sub>q</sub>[:, ' + e.a + '] ← W<sub>q</sub>[:, ' + e.a + '] + α · δ · s</div>' +
          '<div class="formula-box">更新向量 α·δ·s = ' + this.alpha + ' × ' + c.delta.toFixed(4) + ' × [' + s.map(v => v.toFixed(1)).join(', ') +
            '] = [<span style="color:' + (c.gradCol[0] * this.alpha >= 0 ? '#4ade80' : '#f87171') + '">' + (this.alpha * c.gradCol[0]).toFixed(4) + '</span>, <span style="color:' + (c.gradCol[1] * this.alpha >= 0 ? '#4ade80' : '#f87171') + '">' + (this.alpha * c.gradCol[1]).toFixed(4) + '</span>]</div>' +
          this._weightTable(this.W_q, 'W_q 更新后（绿=增大，红=减小）', '#4ade80', this.W_q_init) +
          '<div class="formula-box" style="margin-top:10px">更新后 Q(' + this.stateNames[e.s] + ',' + this.actionNames[e.a] + ') = <span style="color:#93c5fd">' + this._Q(e.s, e.a, this.W_q).toFixed(4) + '</span>  (原 ' + c.q_sa.toFixed(4) + ' → 向 y=' + c.y.toFixed(4) + ' 靠拢)</div>' +
        '</div>'
      );
    }
    if (st === 6) {
      return (
        '<div class="section">' +
          '<div class="sec-title">⑦ 软更新 Q_target</div>' +
          '<div class="formula-box">W<sub>target</sub> ← τ · W<sub>q</sub> + (1 − τ) · W<sub>target</sub>   (τ = ' + this.tau + ')</div>' +
          '<div class="two-col">' +
            '<div>' + this._weightTable(this.W_q, 'W_q (Online 最新)', '#3B82F6') + '</div>' +
            '<div>' + this._weightTable(this.W_target, 'W_target (缓慢跟随)', '#8B5CF6') + '</div>' +
          '</div>' +
          '<div style="color:#888;font-size:0.78rem;margin-top:8px">τ 越小 → target 变化越慢 → TD 目标 y 越稳定 → 训练越稳但收敛越慢。Hard update 则是每 C 步直接拷贝。</div>' +
        '</div>'
      );
    }
    if (st === 7) {
      return (
        '<div class="section">' +
          '<div class="sec-title">⑧ 迭代汇总</div>' +
          '<div class="formula-box">迭代 #' + (this.iteration + 1) + ' 完成 · 本步损失 L = <span style="color:#F59E0B">' + (this.lossHistory.length > 0 ? this.lossHistory[this.lossHistory.length - 1].toFixed(5) : '-') + '</span></div>' +
          '<div class="formula-box">历史损失共 ' + this.lossHistory.length + ' 次 · 平均 = <span style="color:#F59E0B">' + (this.lossHistory.length > 0 ? (this.lossHistory.reduce((a, b) => a + b, 0) / this.lossHistory.length).toFixed(5) : '-') + '</span></div>' +
          '<div style="color:#888;font-size:0.8rem;margin-top:8px">点击 ⏭ 进入下一轮：重新采样 → 计算 y → 更新 Q → 软更新 target。</div>' +
        '</div>'
      );
    }
    return '';
  }

  // ==================== 主渲染 ====================
  render() {
    if (!this.container) return;
    window._dqnInstance = this;

    const st = this.currentStep;
    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._dqnInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._dqnInstance.play()">▶ 播放</button>';

    const stepBar = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff;border-color:#3B82F6' : (i < st ? 'background:#1e3a5f;color:#60a5fa' : 'background:#1a1a1a;color:#888');
      return '<div class="step-pill" onclick="window._dqnInstance.goTo(' + i + ')" style="' + active + '">' + s + '</div>';
    }).join('');

    this.container.innerHTML =
      '<div class="dqn-viz">' +
        '<style>' +
          '.dqn-viz{font-family:Inter,sans-serif;color:#e5e5e5;background:#0a0a0a;padding:18px}' +
          '.dqn-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;font-size:0.85rem}' +
          '.dqn-viz .ctrl-btn:hover{background:#252525;border-color:#3B82F6}' +
          '.dqn-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.dqn-viz .speed-select{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 10px;border-radius:6px;font-size:0.85rem}' +
          '.dqn-viz .formula-box{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.dqn-viz .section{background:#111217;border-radius:8px;padding:14px 16px;border:1px solid #21262d;margin-bottom:14px}' +
          '.dqn-viz .sec-title{font-weight:700;margin-bottom:10px;color:#fff;font-size:0.95rem}' +
          '.dqn-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '@media(max-width:900px){.dqn-viz .two-col{grid-template-columns:1fr}}' +
          '.dqn-viz .edu-panel{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin-bottom:16px}' +
          '.dqn-viz .edu-title{font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #21262d}' +
          '.dqn-viz .edu-section{margin-bottom:14px}' +
          '.dqn-viz .edu-section-title{font-size:0.84rem;font-weight:600;color:#58a6ff;margin-bottom:4px}' +
          '.dqn-viz .edu-text{font-size:0.86rem;color:#8b949e;line-height:1.75}' +
          '.dqn-viz .step-pill{padding:6px 10px;border:1px solid #333;border-radius:14px;font-size:0.72rem;cursor:pointer;transition:all 0.15s;white-space:nowrap}' +
          '.dqn-viz .step-pill:hover{border-color:#3B82F6}' +
        '</style>' +

        // 标题
        '<div style="font-size:1.5rem;font-weight:700;margin-bottom:2px">DQN (Deep Q-Network)</div>' +
        '<div style="color:#8b949e;margin-bottom:14px;font-size:0.88rem">用神经网络近似 Q 函数，结合<strong>经验回放</strong>与<strong>目标网络</strong>稳定训练。</div>' +

        // 教育面板
        '<div class="edu-panel">' +
          '<div class="edu-title">📖 零基础讲解：DQN 的四个支柱</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🧠 ① 神经网络近似 Q</div>' +
            '<div class="edu-text">经典 Q-Learning 用表格存 Q 值，状态空间一大就爆炸。DQN 把 Q 参数化为 Q<sub>θ</sub>(s,a)（本示例用线性 s·W[:,a]，真实 DQN 用多层 CNN/MLP），让模型能<b>泛化</b>到没见过的状态。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">📦 ② Replay Buffer：为什么要回放</div>' +
            '<div class="edu-text">连续采集的经验高度相关（走了一个状态很可能下一步也在附近）。把经验存进 buffer 再<b>随机采样</b>，让训练样本近似 i.i.d.，符合 SGD 的前提；同时一条经验可多次被利用，提升<b>数据效率</b>。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🎯 ③ Target Network：为什么要两个网络</div>' +
            '<div class="edu-text">如果用同一个网络计算 y 和 Q(s,a)，梯度更新会让 y 立刻跟着动——就像"狗追自己的尾巴"，目标一直在变。引入一个更新缓慢的 <b>Q<sub>target</sub></b> 来算 y，让 online 网络朝一个"相对静止"的目标收敛。<br><br><b>Hard update</b>：每 C 步整体拷贝 W<sub>q</sub> → W<sub>target</sub>。<br><b>Soft update</b>：W<sub>target</sub> ← τ·W<sub>q</sub> + (1-τ)·W<sub>target</sub>，τ 很小（0.005 左右）。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">⚡ ④ Bellman 方程 & TD 学习</div>' +
            '<div class="edu-text">Q*(s,a) = E[r + γ·max<sub>a\'</sub> Q*(s\',a\')]。DQN 的损失把 Q 往 <b>y = r + γ·max Q<sub>target</sub></b> 拉近：L = ½(y - Q(s,a))²，这就是<b>半梯度 TD 学习</b>（只对 Q 求导，不对 y 求导）。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">💀 Deadly Triad（致命三元）</div>' +
            '<div class="edu-text">当<b>函数近似 + 自举（bootstrapping）+ off-policy</b>三者同时出现，TD 学习可能发散。DQN 用 replay + target network 正是为了驯服这个魔鬼。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🔄 DQN 变体速览</div>' +
            '<div class="edu-text">' +
              '• <b>Double DQN</b>：用 online 网络选动作、target 网络估值 → 缓解最大化偏差。<br>' +
              '• <b>Dueling DQN</b>：把 Q(s,a) 分解为 V(s) + A(s,a) → 更稳、更好识别状态价值。<br>' +
              '• <b>Prioritized Experience Replay (PER)</b>：按 TD 误差加权采样，重要经验多回放。<br>' +
              '• <b>Rainbow</b>：把以上 + noisy net + n-step + distributional 全部整合。' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 控制栏
        '<div class="section">' +
          '<div class="sec-title">🎮 控制</div>' +
          '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">' +
            '<button class="ctrl-btn" onclick="window._dqnInstance.reset()">↻ 重置</button>' +
            '<button class="ctrl-btn" onclick="window._dqnInstance.stepBack()">⏮ 上一步</button>' +
            playBtn +
            '<button class="ctrl-btn" onclick="window._dqnInstance.stepForward()">⏭ 下一步</button>' +
            '<select class="speed-select" onchange="window._dqnInstance.setSpeed(this.value)">' +
              '<option value="0.5">0.5×</option>' +
              '<option value="1" selected>1×</option>' +
              '<option value="1.5">1.5×</option>' +
              '<option value="2">2×</option>' +
              '<option value="3">3×</option>' +
            '</select>' +
            '<span style="color:#666;font-size:0.78rem;margin-left:8px">γ=' + this.gamma + '   α=' + this.alpha + '   τ=' + this.tau + '   iter=' + this.iteration + '</span>' +
          '</div>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepBar + '</div>' +
        '</div>' +

        // 总公式
        '<div class="formula-box" style="font-size:0.86rem">y = r + γ · max<sub>a\'</sub> Q<sub>target</sub>(s\', a\')   ;   L = ½(y − Q(s,a))²   ;   W<sub>target</sub> ← τ·W<sub>q</sub> + (1-τ)·W<sub>target</sub></div>' +

        // 当前步骤标题
        '<div class="section">' +
          '<div class="sec-title">🧭 当前步骤：' + this.STEPS[st] + '</div>' +
          '<div style="color:#888;font-size:0.8rem">（步 ' + (st + 1) + ' / ' + this.STEPS.length + ' · 迭代 #' + (this.iteration + 1) + '）</div>' +
        '</div>' +

        // 布局：两列 —— 左 buffer + 当前详情，右 Q-table + drift + loss
        '<div class="two-col">' +
          '<div>' +
            this._renderReplayBuffer() +
            this._renderStepDetail() +
          '</div>' +
          '<div>' +
            '<div class="section">' +
              '<div class="sec-title">🔵 Q_online（训练中的网络）</div>' +
              this._qTable(this.W_q, 'Q_online(s, a)', '#3B82F6',
                (st >= 2 ? { s: this._computed.e.s, a: this._computed.e.a } : null), false) +
            '</div>' +
            '<div class="section">' +
              '<div class="sec-title">🟣 Q_target（缓慢跟随的目标网络）</div>' +
              this._qTable(this.W_target, 'Q_target(s, a)', '#8B5CF6',
                (st === 3 && !this._computed.e.done ? { s: this._computed.e.s_next, a: this._computed.argmax } : null), true) +
              '<div style="color:#888;font-size:0.72rem;margin-top:6px">黄色 = 大于 Q_online / 紫色 = 小于 Q_online。差异越大说明漂移越大。</div>' +
            '</div>' +
            this._renderDriftIndicator() +
            this._renderLossChart() +
          '</div>' +
        '</div>' +
      '</div>';

    window._dqnInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._dqnInstance === this) {
      try { delete window._dqnInstance; } catch (e) { window._dqnInstance = null; }
    }
    this.container = null;
  }
}

window.DQN = DQN;
