// Q-Learning 算法可视化
// 无模型、off-policy 的 TD(0) 控制算法
// Q(s,a) ← Q(s,a) + α [r + γ · max_{a'} Q(s',a') - Q(s,a)]
class QLearning {
  constructor() {
    this.container = null;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // ========== 与 rl-mdp.js 一致的环境 ==========
    this.states = ['S0', 'S1', 'S2'];
    this.actions = ['a0', 'a1'];
    this.P = {
      '0_0': [{ s: 0, p: 0.7 }, { s: 1, p: 0.3 }],
      '0_1': [{ s: 1, p: 0.4 }, { s: 2, p: 0.6 }],
      '1_0': [{ s: 0, p: 0.2 }, { s: 2, p: 0.8 }],
      '1_1': [{ s: 0, p: 0.5 }, { s: 1, p: 0.5 }],
      '2_0': [{ s: 1, p: 0.3 }, { s: 2, p: 0.7 }],
      '2_1': [{ s: 0, p: 0.9 }, { s: 1, p: 0.1 }]
    };
    this.R = [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 2.0]
    ];

    // ========== 超参数 ==========
    this.alpha = 0.1;
    this.gamma = 0.9;
    this.epsilon = 0.2;
    this.epsDecay = 0.995;
    this.epsMin = 0.05;

    // ========== Q 表 ==========
    this.Q = [[0, 0], [0, 0], [0, 0]];

    // ========== 回合 / 步骤追踪 ==========
    this.currentEpisode = 0;
    this.stepsPerEpisode = 20;
    this.stepInEpisode = 0;
    this.phase = 0; // 0 idle, 1 pick action, 2 env step, 3 update, 4 next
    this.trace = { s: null, a: null, r: null, s_next: null, qOld: null, qNew: null, delta: null, greedy: null, maxQNext: null };

    // ========== 历史 ==========
    this.updateHistory = [];     // 最近 5 次更新细节
    this.episodeReturns = [];    // 每回合折扣回报
    this.maxQHistory = [];       // 每回合结束时 max(Q)
    this.currentReturn = 0;
  }

  init(container) {
    this.container = container;
    this.resetAll();
    this.render();
    return this;
  }

  resetAll() {
    this.Q = [[0, 0], [0, 0], [0, 0]];
    this.currentEpisode = 0;
    this.stepInEpisode = 0;
    this.phase = 0;
    this.trace = { s: null, a: null, r: null, s_next: null, qOld: null, qNew: null, delta: null, greedy: null, maxQNext: null };
    this.updateHistory = [];
    this.episodeReturns = [];
    this.maxQHistory = [];
    this.currentReturn = 0;
    this.isPlaying = false;
    clearTimeout(this.timer);
  }

  reset() { this.resetAll(); this.render(); }
  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); }
  setSpeed(s) { this.speed = parseFloat(s) || 1; }

  stepForward() {
    if (!this.isPlaying) this._singleStep();
    this.render();
  }

  stepBack() {
    // 简化：回退等价于撤销最后一次更新
    if (this.updateHistory.length > 0) {
      const last = this.updateHistory[this.updateHistory.length - 1];
      this.Q[last.s][last.a] = last.qOld;
      this.updateHistory.pop();
      this.stepInEpisode = Math.max(0, this.stepInEpisode - 1);
      this.phase = 0;
      this.render();
    }
  }

  goTo() { /* no-op for this viz */ }

  _auto() {
    if (!this.isPlaying) return;
    this._singleStep();
    this.render();
    this.timer = setTimeout(() => this._auto(), 1100 / this.speed);
  }

  // ========== 核心算法步骤 ==========
  _singleStep() {
    // 每个逻辑单元：选动作 → 与环境交互 → 更新 Q
    if (this.phase === 0 || this.phase === 4) {
      // 起始 / 继续
      if (this.stepInEpisode >= this.stepsPerEpisode) {
        // 结束本回合
        this.episodeReturns.push(this.currentReturn);
        this.maxQHistory.push(this._maxQ());
        this.currentEpisode++;
        this.stepInEpisode = 0;
        this.currentReturn = 0;
        this.epsilon = Math.max(this.epsMin, this.epsilon * this.epsDecay);
      }
      // 选起始状态
      const s = (this.trace.s_next !== null && this.phase === 4) ? this.trace.s_next : Math.floor(Math.random() * 3);
      this.trace = { s, a: null, r: null, s_next: null, qOld: null, qNew: null, delta: null, greedy: null, maxQNext: null };
      this.phase = 1;
      return;
    }
    if (this.phase === 1) {
      // ε-greedy 选动作
      const s = this.trace.s;
      const isGreedy = Math.random() >= this.epsilon;
      let a;
      if (isGreedy) a = (this.Q[s][0] >= this.Q[s][1]) ? 0 : 1;
      else a = Math.floor(Math.random() * 2);
      this.trace.a = a;
      this.trace.greedy = isGreedy;
      this.phase = 2;
      return;
    }
    if (this.phase === 2) {
      // 与环境交互
      const { s, a } = this.trace;
      const r = this.R[s][a];
      const trans = this.P[s + '_' + a];
      let cum = 0, rand = Math.random(), s_next = trans[0].s;
      for (const t of trans) { cum += t.p; if (rand < cum) { s_next = t.s; break; } }
      this.trace.r = r;
      this.trace.s_next = s_next;
      this.currentReturn += r * Math.pow(this.gamma, this.stepInEpisode);
      this.phase = 3;
      return;
    }
    if (this.phase === 3) {
      // Q 更新
      const { s, a, r, s_next } = this.trace;
      const qOld = this.Q[s][a];
      const maxQNext = Math.max(this.Q[s_next][0], this.Q[s_next][1]);
      const td = r + this.gamma * maxQNext - qOld;
      const delta = this.alpha * td;
      const qNew = qOld + delta;
      this.Q[s][a] = qNew;
      this.trace.qOld = qOld;
      this.trace.qNew = qNew;
      this.trace.maxQNext = maxQNext;
      this.trace.delta = delta;
      this.trace.tdError = td;
      // 记录历史
      this.updateHistory.push(Object.assign({}, this.trace));
      if (this.updateHistory.length > 5) this.updateHistory.shift();
      this.stepInEpisode++;
      this.phase = 4;
      return;
    }
  }

  _maxQ() {
    let m = -Infinity;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) if (this.Q[i][j] > m) m = this.Q[i][j];
    return m;
  }

  _runOneEpisodeFast() {
    // 快速执行一个完整回合（不逐阶段推进，直接跑完）
    let s = Math.floor(Math.random() * 3);
    let G = 0;
    for (let k = 0; k < this.stepsPerEpisode; k++) {
      const a = (Math.random() < this.epsilon) ? Math.floor(Math.random() * 2) : (this.Q[s][0] >= this.Q[s][1] ? 0 : 1);
      const r = this.R[s][a];
      const trans = this.P[s + '_' + a];
      let cum = 0, rand = Math.random(), sNext = trans[0].s;
      for (const t of trans) { cum += t.p; if (rand < cum) { sNext = t.s; break; } }
      const maxQNext = Math.max(this.Q[sNext][0], this.Q[sNext][1]);
      this.Q[s][a] = this.Q[s][a] + this.alpha * (r + this.gamma * maxQNext - this.Q[s][a]);
      G += r * Math.pow(this.gamma, k);
      s = sNext;
    }
    this.episodeReturns.push(G);
    this.maxQHistory.push(this._maxQ());
    this.currentEpisode++;
    this.epsilon = Math.max(this.epsMin, this.epsilon * this.epsDecay);
    // 复位 phase 状态
    this.stepInEpisode = 0;
    this.currentReturn = 0;
    this.phase = 0;
  }

  runFullEpisode() {
    this._runOneEpisodeFast();
    this.render();
  }

  run100Episodes() {
    for (let i = 0; i < 100; i++) this._runOneEpisodeFast();
    this.render();
  }

  // ========== 渲染组件 ==========
  _renderQTable() {
    const t = this.trace;
    let rows = '';
    for (let i = 0; i < 3; i++) {
      let cells = '';
      for (let j = 0; j < 2; j++) {
        const v = this.Q[i][j];
        const isCur = t.s === i && t.a === j && this.phase >= 1;
        const isUp = t.s === i && t.a === j && this.phase === 4;
        const intensity = Math.min(Math.abs(v) * 0.25, 0.5);
        const bg = v === 0 ? '#141414' : (v > 0 ? 'rgba(74,222,128,' + (0.08 + intensity) + ')' : 'rgba(248,113,113,' + (0.08 + intensity) + ')');
        const color = v > 0 ? '#4ade80' : (v < 0 ? '#f87171' : '#888');
        let style = 'padding:10px 16px;border:1px solid #333;background:' + bg + ';color:' + color + ';font-family:JetBrains Mono,monospace;font-size:0.85rem;text-align:center;min-width:72px;transition:all 0.3s;';
        if (isUp) style += 'outline:2px solid #4ade80;outline-offset:-2px;animation:pulse 0.6s;';
        else if (isCur) style += 'outline:2px solid #F97316;outline-offset:-2px;';
        cells += '<td style="' + style + '">' + v.toFixed(4) + '</td>';
      }
      rows += '<tr><td style="padding:10px 16px;border:1px solid #333;color:#a5f3fc;font-weight:600;background:#111;text-align:center">' + this.states[i] + '</td>' + cells + '</tr>';
    }
    return '<table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.85rem">' +
      '<tr><th style="padding:8px 16px;border:1px solid #333;color:#888"></th>' +
      '<th style="padding:8px 16px;border:1px solid #3B82F6;color:#3B82F6">a0</th>' +
      '<th style="padding:8px 16px;border:1px solid #F97316;color:#F97316">a1</th></tr>' + rows + '</table>';
  }

  _renderPolicyBars() {
    // 每个状态一组 Q 值柱状图 + 贪心箭头
    let out = '';
    const qAll = [].concat(this.Q[0], this.Q[1], this.Q[2]);
    const maxAbs = Math.max(0.5, Math.max.apply(null, qAll.map(Math.abs)));
    for (let s = 0; s < 3; s++) {
      const q = this.Q[s];
      const greedyA = q[0] >= q[1] ? 0 : 1;
      const mkBar = (val, a) => {
        const w = Math.max(4, Math.abs(val) / maxAbs * 100);
        const color = a === 0 ? '#3B82F6' : '#F97316';
        const isGreedy = a === greedyA;
        const arrow = isGreedy ? '<span style="color:#4ade80;margin-left:6px;font-weight:600">← π*</span>' : '';
        return '<div style="display:flex;align-items:center;gap:8px;margin:4px 0">' +
          '<span style="color:' + color + ';width:30px;font-family:JetBrains Mono,monospace;font-size:0.8rem">a' + a + '</span>' +
          '<div style="flex:1;background:#0a0a0a;border-radius:3px;height:18px;position:relative">' +
          '<div style="background:' + color + ';height:100%;width:' + w + '%;border-radius:3px;opacity:0.7"></div>' +
          '</div>' +
          '<span style="color:' + color + ';width:60px;font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:right">' + val.toFixed(3) + '</span>' +
          arrow +
          '</div>';
      };
      out += '<div style="padding:8px 10px;margin-bottom:8px;background:#111;border-radius:5px;border-left:3px solid #a5f3fc">' +
        '<div style="font-weight:600;color:#a5f3fc;font-size:0.84rem;margin-bottom:4px">' + this.states[s] + '  <span style="color:#888;font-size:0.74rem;font-weight:400">贪心动作 = ' + this.actions[greedyA] + '</span></div>' +
        mkBar(q[0], 0) + mkBar(q[1], 1) +
        '</div>';
    }
    return out;
  }

  _renderConvergencePlot() {
    const data = this.maxQHistory;
    if (data.length < 2) {
      return '<div style="color:#666;font-size:0.82rem;padding:16px;text-align:center">至少完成 2 个回合后才能看到曲线。</div>';
    }
    const W = 520, H = 180, PAD = 30;
    const maxX = Math.max(1, data.length - 1);
    const minY = Math.min.apply(null, data);
    const maxY = Math.max.apply(null, data);
    const spanY = Math.max(1e-6, maxY - minY);
    const xToPx = x => PAD + (x / maxX) * (W - 2 * PAD);
    const yToPx = y => H - PAD - ((y - minY) / spanY) * (H - 2 * PAD);
    let path = 'M ' + xToPx(0) + ' ' + yToPx(data[0]);
    for (let i = 1; i < data.length; i++) path += ' L ' + xToPx(i) + ' ' + yToPx(data[i]);
    // 网格
    let grid = '';
    for (let g = 0; g <= 4; g++) {
      const y = PAD + g * (H - 2 * PAD) / 4;
      const yVal = maxY - g * spanY / 4;
      grid += '<line x1="' + PAD + '" y1="' + y + '" x2="' + (W - PAD) + '" y2="' + y + '" stroke="#222" stroke-width="1"/>';
      grid += '<text x="' + (PAD - 4) + '" y="' + (y + 3) + '" fill="#666" font-size="9" font-family="JetBrains Mono,monospace" text-anchor="end">' + yVal.toFixed(2) + '</text>';
    }
    // x-axis ticks
    const tickStep = Math.max(1, Math.floor(maxX / 5));
    let ticks = '';
    for (let i = 0; i <= maxX; i += tickStep) {
      ticks += '<text x="' + xToPx(i) + '" y="' + (H - PAD + 14) + '" fill="#666" font-size="9" text-anchor="middle" font-family="JetBrains Mono,monospace">' + i + '</text>';
    }
    // 最后一个点
    const lastX = xToPx(data.length - 1), lastY = yToPx(data[data.length - 1]);
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:auto;max-height:220px;background:#0a0a0a;border-radius:4px">' +
      grid + ticks +
      '<path d="' + path + '" fill="none" stroke="#4ade80" stroke-width="2"/>' +
      '<circle cx="' + lastX + '" cy="' + lastY + '" r="3.5" fill="#4ade80"/>' +
      '<text x="' + (W / 2) + '" y="' + (H - 4) + '" fill="#888" font-size="10" text-anchor="middle">episode</text>' +
      '<text x="' + 8 + '" y="' + 14 + '" fill="#888" font-size="10">max Q</text>' +
      '</svg>';
  }

  _renderReturnsPlot() {
    const data = this.episodeReturns;
    if (data.length < 2) return '<div style="color:#666;font-size:0.82rem;padding:8px">需更多数据...</div>';
    const W = 520, H = 140, PAD = 30;
    const maxX = Math.max(1, data.length - 1);
    const minY = Math.min.apply(null, data);
    const maxY = Math.max.apply(null, data);
    const spanY = Math.max(1e-6, maxY - minY);
    const xToPx = x => PAD + (x / maxX) * (W - 2 * PAD);
    const yToPx = y => H - PAD - ((y - minY) / spanY) * (H - 2 * PAD);
    let path = 'M ' + xToPx(0) + ' ' + yToPx(data[0]);
    for (let i = 1; i < data.length; i++) path += ' L ' + xToPx(i) + ' ' + yToPx(data[i]);
    // 滑动平均 window=10
    const smooth = [];
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - 9);
      let s = 0;
      for (let j = start; j <= i; j++) s += data[j];
      smooth.push(s / (i - start + 1));
    }
    let smoothPath = 'M ' + xToPx(0) + ' ' + yToPx(smooth[0]);
    for (let i = 1; i < smooth.length; i++) smoothPath += ' L ' + xToPx(i) + ' ' + yToPx(smooth[i]);
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:auto;max-height:180px;background:#0a0a0a;border-radius:4px">' +
      '<text x="' + 8 + '" y="' + 14 + '" fill="#888" font-size="10">return G</text>' +
      '<path d="' + path + '" fill="none" stroke="#3B82F6" stroke-width="1.2" opacity="0.5"/>' +
      '<path d="' + smoothPath + '" fill="none" stroke="#F97316" stroke-width="2"/>' +
      '<text x="' + (W - PAD) + '" y="' + 14 + '" fill="#F97316" font-size="10" text-anchor="end">10-episode MA</text>' +
      '</svg>';
  }

  _renderCurrentStepPanel() {
    const t = this.trace;
    if (this.phase === 0 || t.s === null) {
      return '<div style="color:#666;font-size:0.85rem;padding:10px">点击"单步"或"▶ 播放"开始。算法会在 <code>状态 → 选动作 → 交互 → 更新 Q</code> 之间循环。</div>';
    }
    let lines = [];
    const sc = '#a5f3fc';
    const ac = t.a === 0 ? '#3B82F6' : '#F97316';

    // Phase 1
    lines.push('<div style="color:#888;font-size:0.78rem;margin-top:4px">① 当前状态</div>');
    lines.push('<div class="formula-box">s = <span style="color:' + sc + ';font-weight:600">' + this.states[t.s] + '</span></div>');

    // Phase 2 完成后有 action
    if (this.phase >= 2) {
      const how = t.greedy ? '利用 (argmax)' : '探索 (random)';
      const howColor = t.greedy ? '#4ade80' : '#F59E0B';
      lines.push('<div style="color:#888;font-size:0.78rem;margin-top:8px">② ε-greedy 选动作 (ε=' + this.epsilon.toFixed(3) + ')</div>');
      lines.push('<div class="formula-box">' +
        'r ~ U(0,1): <span style="color:' + howColor + '">' + (t.greedy ? '>= ε' : '< ε') + '</span> → <b>' + how + '</b><br>' +
        'a = <span style="color:' + ac + ';font-weight:600">' + this.actions[t.a] + '</span>' +
        (t.greedy ? '  = argmax_a Q(' + this.states[t.s] + ',a) = argmax[' + this.Q[t.s][0].toFixed(3) + ', ' + this.Q[t.s][1].toFixed(3) + ']' : '') +
        '</div>');
    }

    if (this.phase >= 3) {
      const rc = t.r > 0 ? '#4ade80' : (t.r < 0 ? '#f87171' : '#888');
      lines.push('<div style="color:#888;font-size:0.78rem;margin-top:8px">③ 环境返回</div>');
      const trans = this.P[t.s + '_' + t.a];
      const transStr = trans.map(x => 'P(' + this.states[x.s] + ')=' + x.p.toFixed(1)).join(', ');
      lines.push('<div class="formula-box">' +
        'r = R(' + this.states[t.s] + ',' + this.actions[t.a] + ') = <span style="color:' + rc + ';font-weight:600">' + t.r.toFixed(1) + '</span><br>' +
        's\' ~ P(·|' + this.states[t.s] + ',' + this.actions[t.a] + ') = {' + transStr + '} → <span style="color:' + sc + ';font-weight:600">' + this.states[t.s_next] + '</span>' +
        '</div>');
    }

    if (this.phase === 4) {
      const dc = t.delta >= 0 ? '#4ade80' : '#f87171';
      const tdc = t.tdError >= 0 ? '#4ade80' : '#f87171';
      lines.push('<div style="color:#888;font-size:0.78rem;margin-top:8px">④ TD(0) 更新</div>');
      lines.push('<div class="formula-box">' +
        '<b>Q(s,a) ← Q(s,a) + α[r + γ·max_{a\'} Q(s\',a\') − Q(s,a)]</b><br><br>' +
        'Q_old(' + this.states[t.s] + ',' + this.actions[t.a] + ') = <span style="color:#a5f3fc">' + t.qOld.toFixed(4) + '</span><br>' +
        'max_{a\'} Q(' + this.states[t.s_next] + ',·) = max[' + this.Q[t.s_next][0].toFixed(3) + ', ' + this.Q[t.s_next][1].toFixed(3) + '] = <span style="color:#a5f3fc">' + t.maxQNext.toFixed(4) + '</span><br>' +
        'TD target = r + γ·maxQ\' = ' + t.r.toFixed(1) + ' + ' + this.gamma + '·' + t.maxQNext.toFixed(3) + ' = <span style="color:#a5f3fc">' + (t.r + this.gamma * t.maxQNext).toFixed(4) + '</span><br>' +
        'TD error δ = target − Q_old = <span style="color:' + tdc + ';font-weight:600">' + t.tdError.toFixed(4) + '</span><br>' +
        'Δ = α·δ = ' + this.alpha + '·' + t.tdError.toFixed(3) + ' = <span style="color:' + dc + '">' + t.delta.toFixed(4) + '</span><br>' +
        'Q_new = ' + t.qOld.toFixed(4) + ' + ' + t.delta.toFixed(4) + ' = <span style="color:#4ade80;font-weight:600">' + t.qNew.toFixed(4) + '</span>' +
        '</div>');
    }
    return lines.join('');
  }

  _renderHistory() {
    if (this.updateHistory.length === 0) return '<div style="color:#666;font-size:0.82rem">暂无更新。</div>';
    let rows = '';
    for (let i = this.updateHistory.length - 1; i >= 0; i--) {
      const u = this.updateHistory[i];
      const ac = u.a === 0 ? '#3B82F6' : '#F97316';
      const dc = u.delta >= 0 ? '#4ade80' : '#f87171';
      rows += '<div style="display:flex;gap:8px;padding:5px 8px;margin:3px 0;background:#111;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.76rem;align-items:center;flex-wrap:wrap">' +
        '<span style="color:#a5f3fc">' + this.states[u.s] + '</span>' +
        '<span style="color:' + ac + '">' + this.actions[u.a] + '</span>' +
        '<span style="color:#888">r=' + u.r.toFixed(1) + '</span>' +
        '<span style="color:#666">→</span>' +
        '<span style="color:#a5f3fc">' + this.states[u.s_next] + '</span>' +
        '<span style="color:' + dc + '">Δ=' + u.delta.toFixed(4) + '</span>' +
        '<span style="color:#888;font-size:0.72rem">Q:' + u.qOld.toFixed(3) + '→' + u.qNew.toFixed(3) + '</span>' +
        '</div>';
    }
    return rows;
  }

  // ========== 主渲染 ==========
  render() {
    if (!this.container) return;
    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._qlInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._qlInstance.play()">▶ 播放</button>';

    const epsBadge = '<span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#F59E0B">ε=' + this.epsilon.toFixed(3) + '</span>';

    this.container.innerHTML =
      '<div class="ql-viz">' +
        '<style>' +
          '.ql-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.ql-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;font-size:0.85rem}' +
          '.ql-viz .ctrl-btn:hover{background:#252525;border-color:#3B82F6}' +
          '.ql-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.ql-viz .ctrl-btn.accent{border-color:#F97316;color:#F97316}' +
          '.ql-viz .ctrl-btn.accent:hover{background:#F97316;color:#fff}' +
          '.ql-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#a5f3fc;margin:6px 0;line-height:1.6}' +
          '.ql-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.ql-viz .edu-panel{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin-bottom:16px}' +
          '.ql-viz .edu-title{font-size:1.02rem;font-weight:700;color:#fff;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #21262d}' +
          '.ql-viz .edu-section{margin-bottom:12px}' +
          '.ql-viz .edu-section-title{font-size:0.82rem;font-weight:600;color:#58a6ff;margin-bottom:4px}' +
          '.ql-viz .edu-text{font-size:0.86rem;color:#c9d1d9;line-height:1.7}' +
          '.ql-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}' +
          '.ql-viz .three-col{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}' +
          '@media(max-width:900px){.ql-viz .two-col,.ql-viz .three-col{grid-template-columns:1fr}}' +
          '@keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.08)}100%{transform:scale(1)}}' +
        '</style>' +

        // 头部
        '<div style="font-size:1.5rem;font-weight:600;margin-bottom:4px">Q-Learning</div>' +
        '<div style="color:#888;margin-bottom:16px;font-size:0.9rem">Model-free · Off-policy · TD(0) 控制 — 通过 <code>r + γ·max Q(s\',·)</code> 自举学习最优动作价值函数</div>' +

        // 算法概念
        '<div class="edu-panel">' +
          '<div class="edu-title">📖 Q-Learning 是什么？</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🎯 核心目标：学会 Q*(s,a)</div>' +
            '<div class="edu-text">' +
              'Q*(s,a) = 在状态 s 执行动作 a 后，<b>按最优策略</b>玩到底能获得的期望折扣回报。<br>' +
              '一旦学会 Q*，最优策略就是：<code>π*(s) = argmax_a Q*(s,a)</code>——直接查表即可。<br>' +
              '关键是：<b>我们不知道 P 和 R</b>（model-free），只能通过与环境交互来学习 Q 表。' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🧮 更新公式</div>' +
            '<div class="formula-box" style="font-size:0.92rem;text-align:center;padding:14px">Q(s,a) ← Q(s,a) + α · [ r + γ·max<sub>a\'</sub> Q(s\',a\') − Q(s,a) ]</div>' +
            '<div class="edu-text">' +
              '• <b style="color:#a5f3fc">r</b>：环境刚返回的即时奖励<br>' +
              '• <b style="color:#a5f3fc">γ·max<sub>a\'</sub> Q(s\',a\')</b>：下一状态的最乐观估计（这就是 "off-policy" 的根源——不管当前怎么选，都假设未来按最优走）<br>' +
              '• <b style="color:#a5f3fc">r + γ·max Q(s\',·)</b>：<b>TD target</b>（时序差分目标，Bellman 最优方程右边的采样估计）<br>' +
              '• <b style="color:#F59E0B">δ = target − Q(s,a)</b>：<b>TD error</b>（新老估计的差）<br>' +
              '• <b style="color:#4ade80">α</b>：学习率，控制修正步长' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🧭 ε-greedy 行为策略</div>' +
            '<div class="edu-text">智能体与环境交互时<b>不能一直贪心</b>——否则初始估计就把它锁死。ε-greedy：<br>' +
              '• 以概率 ε 随机选动作（<b style="color:#F59E0B">探索</b>）<br>' +
              '• 以概率 1−ε 选 argmax Q（<b style="color:#4ade80">利用</b>）<br>' +
              '通常 ε 从 1.0 指数衰减到 0.05 左右。注意：ε-greedy 是<b>收集数据</b>用的行为策略，<b>更新目标</b>却始终是贪心策略 → 典型的 off-policy。' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 控制
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">控制</div>' +
          playBtn +
          '<button class="ctrl-btn" onclick="window._qlInstance.stepForward()">▸ 单步</button>' +
          '<button class="ctrl-btn" onclick="window._qlInstance.stepBack()">◂ 回退</button>' +
          '<button class="ctrl-btn" onclick="window._qlInstance.reset()">↻ 重置</button>' +
          '<select onchange="window._qlInstance.setSpeed(this.value)" style="background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 10px;border-radius:6px;font-size:0.85rem;margin-right:8px">' +
            '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option>' +
          '</select>' +
          '<button class="ctrl-btn accent" onclick="window._qlInstance.runFullEpisode()">🎬 跑完当前回合</button>' +
          '<button class="ctrl-btn accent" onclick="window._qlInstance.run100Episodes()">▶▶ 跑 100 回合</button>' +
          '<div style="margin-top:10px;font-size:0.82rem;color:#888;font-family:JetBrains Mono,monospace">' +
            'episode=<span style="color:#a5f3fc">' + this.currentEpisode + '</span>  ' +
            'step=<span style="color:#a5f3fc">' + this.stepInEpisode + '/' + this.stepsPerEpisode + '</span>  ' +
            'α=' + this.alpha + '  γ=' + this.gamma + '  ' + epsBadge +
          '</div>' +
        '</div>' +

        // Q-Table + 当前步详情
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">📋 Q-Table (3 × 2)</div>' +
            this._renderQTable() +
            '<div style="margin-top:10px;font-size:0.76rem;color:#888">橙色边框 = 当前 (s,a) &nbsp;|&nbsp; 绿色脉冲 = 刚更新 &nbsp;|&nbsp; 颜色深度 ∝ |Q|</div>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">🔍 当前步详情 (phase=' + this.phase + ')</div>' +
            this._renderCurrentStepPanel() +
          '</div>' +
        '</div>' +

        // 策略柱状图 + 更新历史
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">🏆 贪心策略可视化 (每个状态的 Q 值对比)</div>' +
            this._renderPolicyBars() +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">🕒 最近 5 次更新</div>' +
            this._renderHistory() +
          '</div>' +
        '</div>' +

        // 收敛曲线
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">📈 收敛曲线：max<sub>s,a</sub> Q(s,a) vs 回合数</div>' +
          this._renderConvergencePlot() +
          '<div style="margin-top:10px;font-size:0.8rem;color:#888">max Q 应该随训练单调上升并趋于一个稳定值（理论上等于最优 V*）。</div>' +
          '<div style="margin-top:12px;font-weight:600;font-size:0.88rem;margin-bottom:6px">📊 每回合折扣回报（越高越好）</div>' +
          this._renderReturnsPlot() +
        '</div>' +

        // 教育面板：off-policy vs on-policy
        '<div class="edu-panel">' +
          '<div class="edu-title">⚖️ Q-Learning vs SARSA：Off-policy vs On-policy</div>' +
          '<div class="two-col">' +
            '<div>' +
              '<div class="edu-section-title" style="color:#4ade80">Q-Learning (off-policy)</div>' +
              '<div class="formula-box">Q(s,a) ← Q(s,a) + α[r + γ·<b>max<sub>a\'</sub> Q(s\',a\')</b> − Q(s,a)]</div>' +
              '<div class="edu-text" style="font-size:0.84rem">目标用<b>贪心</b>策略（与行为策略无关）。即使你用 ε-greedy 探索，更新时假装未来完全最优。可从<b>旧数据、他人数据、离线数据</b>中学习（经验回放可行）。缺点：在危险环境可能过于乐观。</div>' +
            '</div>' +
            '<div>' +
              '<div class="edu-section-title" style="color:#F59E0B">SARSA (on-policy)</div>' +
              '<div class="formula-box">Q(s,a) ← Q(s,a) + α[r + γ·<b>Q(s\',a\')</b> − Q(s,a)]<br><span style="color:#888">其中 a\' ~ π (相同的行为策略)</span></div>' +
              '<div class="edu-text" style="font-size:0.84rem">目标用<b>实际采取</b>的下一动作 a\'。<b>更保守</b>：如果行为策略有探索风险，SARSA 会学到"带探索偏见"的 Q。经典例子：悬崖行走（cliff walking）中 SARSA 会绕远路避免掉下去，Q-Learning 会贴边走。</div>' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 收敛性与陷阱
        '<div class="edu-panel">' +
          '<div class="edu-title">🔬 为什么 Q-Learning 会收敛？</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">Banach 不动点定理 + Bellman 最优算子的收缩性</div>' +
            '<div class="edu-text">定义 Bellman 最优算子 T*：<br>' +
              '<span style="font-family:JetBrains Mono,monospace;color:#a5f3fc">(T*Q)(s,a) = Σ_{s\'} P(s\'|s,a)[R(s,a) + γ·max_{a\'} Q(s\',a\')]</span><br><br>' +
              '关键性质：T* 是 <b>γ-收缩的</b>——对任意 Q₁, Q₂：<br>' +
              '<span style="font-family:JetBrains Mono,monospace;color:#a5f3fc">‖T*Q₁ − T*Q₂‖_∞ ≤ γ · ‖Q₁ − Q₂‖_∞</span><br><br>' +
              '由 Banach 不动点定理，T* 有唯一不动点——就是 Q*。Q-Learning 是 T* 的<b>随机近似</b>（用采样替代期望），在条件：<br>' +
              '• 每个 (s, a) 被访问无穷多次<br>' +
              '• 学习率 α 满足 Σα=∞ 且 Σα²&lt;∞（如 α_t = 1/t）<br>' +
              '下，Q_t → Q*（几乎必然收敛，Watkins 1989, Tsitsiklis 1994）。' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">⚠️ 实战陷阱</div>' +
            '<div class="edu-text">' +
              '• <b>探索不足</b>：若 ε 太小或衰减太快，某些 (s,a) 可能永远访问不到，收敛保证失效。<br>' +
              '• <b>最大化偏差 (Maximization Bias)</b>：max 操作对噪声敏感，会高估 Q。改进：Double Q-Learning（两张 Q 表交替更新）。<br>' +
              '• <b>状态太大</b>：表格形式不现实（围棋有 10¹⁷⁰ 个状态）→ 用神经网络近似 Q(s,a;θ) → DQN。<br>' +
              '• <b>连续动作</b>：max_{a\'} 无法枚举 → 改用 Policy Gradient / DDPG / SAC。' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 伪代码
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">📜 标准 Q-Learning 伪代码</div>' +
          '<div class="formula-box" style="white-space:pre-wrap;line-height:1.7">' +
            '初始化 Q(s,a) = 0  对所有 (s,a)\n' +
            '对每个回合 episode = 1..N:\n' +
            '    初始化 s ~ 初始分布\n' +
            '    重复直到终止或达到步数上限:\n' +
            '        a ← 用 ε-greedy(Q, s) 选动作\n' +
            '        执行 a, 观察 r 和 s\'\n' +
            '        Q(s,a) ← Q(s,a) + α · [r + γ · max_{a\'} Q(s\',a\') − Q(s,a)]\n' +
            '        s ← s\'\n' +
            '    ε ← max(ε_min, ε · decay)' +
          '</div>' +
        '</div>' +

        // 超参数指南
        '<div class="edu-panel">' +
          '<div class="edu-title">🎛️ 超参数调优指南</div>' +
          '<div class="three-col">' +
            '<div>' +
              '<div class="edu-section-title">α（学习率）</div>' +
              '<div class="edu-text" style="font-size:0.82rem">典型 0.01 ~ 0.5。<br>太大：震荡/发散<br>太小：收敛慢<br>表格 Q-Learning 可用 1/N(s,a)（访问次数倒数）保证理论收敛。</div>' +
            '</div>' +
            '<div>' +
              '<div class="edu-section-title">γ（折扣因子）</div>' +
              '<div class="edu-text" style="font-size:0.82rem">典型 0.9 ~ 0.999。<br>γ 越大：越重视长期，有效时域 ≈ 1/(1−γ)。<br>γ=0.99 → 约 100 步；γ=0.999 → 约 1000 步。<br>稀疏奖励任务需要大 γ。</div>' +
            '</div>' +
            '<div>' +
              '<div class="edu-section-title">ε（探索率）</div>' +
              '<div class="edu-text" style="font-size:0.82rem">典型起始 1.0，指数衰减至 0.05。<br>或用更智能的探索：UCB、Thompson sampling、好奇心驱动（ICM）等。</div>' +
            '</div>' +
          '</div>' +
        '</div>' +

      '</div>';

    window._qlInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._qlInstance === this) {
      try { delete window._qlInstance; } catch (e) { window._qlInstance = null; }
    }
    this.container = null;
  }
}

window.QLearning = QLearning;
