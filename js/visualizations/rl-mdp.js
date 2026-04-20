// MDP 基础可视化
// 3-state / 2-action 的教学用 MDP：含状态图、Bellman 价值迭代、回合模拟
class RLMDPTutorial {
  constructor() {
    this.container = null;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // 状态与动作
    this.states = ['S0', 'S1', 'S2'];
    this.actions = ['a0', 'a1'];

    // 转移概率 P(s' | s, a)：键 = "s_a"
    this.P = {
      '0_0': [{ s: 0, p: 0.7 }, { s: 1, p: 0.3 }],
      '0_1': [{ s: 1, p: 0.4 }, { s: 2, p: 0.6 }],
      '1_0': [{ s: 0, p: 0.2 }, { s: 2, p: 0.8 }],
      '1_1': [{ s: 0, p: 0.5 }, { s: 1, p: 0.5 }],
      '2_0': [{ s: 1, p: 0.3 }, { s: 2, p: 0.7 }],
      '2_1': [{ s: 0, p: 0.9 }, { s: 1, p: 0.1 }]
    };

    // 奖励函数 R(s, a)
    this.R = [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 2.0]
    ];

    this.gamma = 0.9;

    // 节点布局（SVG 坐标）
    this.layout = {
      S0: { cx: 120, cy: 90 },
      S1: { cx: 360, cy: 90 },
      S2: { cx: 240, cy: 260 }
    };

    // Bellman 价值迭代状态
    this.V = [0.0, 0.0, 0.0];
    this.vIter = 0;
    this.vHistory = [[0, 0, 0]];
    this.vConverged = false;

    // 回合模拟状态
    this.episode = { trajectory: [], G: 0, done: false };
    this.episodeStep = 0;
    this.maxEpisodeSteps = 8;

    // 交互高亮
    this.hoverState = null;
  }

  init(container) {
    this.container = container;
    this.resetAll();
    this.render();
    return this;
  }

  resetAll() {
    this.V = [0.0, 0.0, 0.0];
    this.vIter = 0;
    this.vHistory = [[0, 0, 0]];
    this.vConverged = false;
    this.episode = { trajectory: [], G: 0, done: false };
    this.episodeStep = 0;
    this.hoverState = null;
    this.isPlaying = false;
    clearTimeout(this.timer);
  }

  reset() { this.resetAll(); this.render(); }
  play() { this.isPlaying = true; this._autoStep(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); }
  setSpeed(s) { this.speed = parseFloat(s) || 1; }

  stepForward() {
    if (!this.vConverged) this.runValueIteration(1);
    else if (!this.episode.done) this.simulateEpisodeStep();
    this.render();
  }

  stepBack() {
    if (this.vHistory.length > 1) {
      this.vHistory.pop();
      this.V = this.vHistory[this.vHistory.length - 1].slice();
      this.vIter = Math.max(0, this.vIter - 1);
      this.vConverged = false;
      this.render();
    }
  }

  goTo(i) {
    if (i >= 0 && i < this.vHistory.length) {
      this.V = this.vHistory[i].slice();
      this.vIter = i;
      this.render();
    }
  }

  _autoStep() {
    if (!this.isPlaying) return;
    if (!this.vConverged) this.runValueIteration(1);
    else if (!this.episode.done) this.simulateEpisodeStep();
    else { this.isPlaying = false; this.render(); return; }
    this.render();
    this.timer = setTimeout(() => this._autoStep(), 1200 / this.speed);
  }

  // ==================
  // Bellman 价值迭代
  // ==================
  runValueIteration(iters = 1) {
    for (let k = 0; k < iters; k++) {
      const Vnew = [0, 0, 0];
      for (let s = 0; s < 3; s++) {
        let bestQ = -Infinity;
        for (let a = 0; a < 2; a++) {
          const trans = this.P[s + '_' + a];
          let expV = 0;
          for (const t of trans) expV += t.p * this.V[t.s];
          const q = this.R[s][a] + this.gamma * expV;
          if (q > bestQ) bestQ = q;
        }
        Vnew[s] = bestQ;
      }
      const delta = Math.max.apply(null, Vnew.map((v, i) => Math.abs(v - this.V[i])));
      this.V = Vnew;
      this.vHistory.push(Vnew.slice());
      this.vIter++;
      if (delta < 1e-4 || this.vIter > 60) { this.vConverged = true; break; }
    }
  }

  runValueIterationAll() {
    while (!this.vConverged) this.runValueIteration(1);
    this.render();
  }

  // ==================
  // 回合模拟
  // ==================
  _greedyAction(s) {
    let bestA = 0, bestQ = -Infinity;
    for (let a = 0; a < 2; a++) {
      const trans = this.P[s + '_' + a];
      let expV = 0;
      for (const t of trans) expV += t.p * this.V[t.s];
      const q = this.R[s][a] + this.gamma * expV;
      if (q > bestQ) { bestQ = q; bestA = a; }
    }
    return bestA;
  }

  simulateEpisodeStep() {
    if (this.episode.done) return;
    let s;
    if (this.episode.trajectory.length === 0) {
      s = Math.floor(Math.random() * 3);
    } else {
      const last = this.episode.trajectory[this.episode.trajectory.length - 1];
      s = last.s_next;
    }
    // 若已经收敛则使用贪心策略，否则随机
    const useGreedy = this.vConverged;
    const a = useGreedy ? this._greedyAction(s) : Math.floor(Math.random() * 2);
    const r = this.R[s][a];
    const trans = this.P[s + '_' + a];
    let cum = 0, rand = Math.random(), s_next = trans[0].s;
    for (const t of trans) {
      cum += t.p;
      if (rand < cum) { s_next = t.s; break; }
    }
    const k = this.episode.trajectory.length;
    const discounted = r * Math.pow(this.gamma, k);
    this.episode.G += discounted;
    this.episode.trajectory.push({ s, a, r, s_next, discounted, k, policy: useGreedy ? 'greedy' : 'random' });
    this.episodeStep++;
    if (this.episodeStep >= this.maxEpisodeSteps) this.episode.done = true;
  }

  resetEpisode() {
    this.episode = { trajectory: [], G: 0, done: false };
    this.episodeStep = 0;
    this.render();
  }

  runFullEpisode() {
    this.resetEpisode();
    while (!this.episode.done) this.simulateEpisodeStep();
    this.render();
  }

  setHover(s) {
    this.hoverState = s;
    this.render();
  }

  // ==================
  // SVG 状态图
  // ==================
  _renderGraph() {
    const w = 520, h = 360;
    const L = this.layout;
    const hov = this.hoverState;
    const stateColors = ['#3B82F6', '#8B5CF6', '#F59E0B'];

    // 箭头 defs
    let defs = '<defs>';
    const aColors = { a0: '#3B82F6', a1: '#F97316' };
    for (const k in aColors) {
      defs += '<marker id="arr-' + k + '" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">' +
        '<path d="M0,0 L0,6 L9,3 z" fill="' + aColors[k] + '"/></marker>';
      defs += '<marker id="arr-' + k + '-dim" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">' +
        '<path d="M0,0 L0,6 L9,3 z" fill="' + aColors[k] + '" opacity="0.25"/></marker>';
    }
    defs += '</defs>';

    // 画边
    let edges = '';
    const drawEdge = (sIdx, aIdx, tgtIdx, p) => {
      const src = L[this.states[sIdx]];
      const tgt = L[this.states[tgtIdx]];
      const color = aIdx === 0 ? '#3B82F6' : '#F97316';
      const dim = (hov !== null && hov !== sIdx);
      const opacity = dim ? 0.22 : 0.95;
      const marker = dim ? 'arr-a' + aIdx + '-dim' : 'arr-a' + aIdx;
      const dx = tgt.cx - src.cx, dy = tgt.cy - src.cy;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / len, uy = dy / len;
      // self-loop case
      if (sIdx === tgtIdx) {
        const cx = src.cx, cy = src.cy;
        const offX = aIdx === 0 ? 0 : 40;
        const loopPath = 'M ' + (cx + 20) + ' ' + (cy - 20) + ' C ' + (cx + 80 + offX) + ' ' + (cy - 70) +
          ', ' + (cx + 80 + offX) + ' ' + (cy + 20) + ', ' + (cx + 20) + ' ' + cy;
        edges += '<path d="' + loopPath + '" fill="none" stroke="' + color + '" stroke-width="2" opacity="' + opacity + '" marker-end="url(#' + marker + ')"/>';
        edges += '<text x="' + (cx + 90 + offX) + '" y="' + (cy - 25) + '" fill="' + color + '" font-size="11" font-family="JetBrains Mono,monospace" opacity="' + opacity + '">' + aIdx + ':' + p.toFixed(1) + '</text>';
        return;
      }
      // offset so parallel edges don't overlap
      const perpX = -uy, perpY = ux;
      const off = (aIdx === 0 ? -12 : 12);
      const startX = src.cx + ux * 30 + perpX * off;
      const startY = src.cy + uy * 30 + perpY * off;
      const endX = tgt.cx - ux * 32 + perpX * off;
      const endY = tgt.cy - uy * 32 + perpY * off;
      const midX = (startX + endX) / 2 + perpX * 14;
      const midY = (startY + endY) / 2 + perpY * 14;
      const path = 'M ' + startX + ' ' + startY + ' Q ' + midX + ' ' + midY + ' ' + endX + ' ' + endY;
      edges += '<path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="2" opacity="' + opacity + '" marker-end="url(#' + marker + ')"/>';
      const labX = midX + perpX * 8;
      const labY = midY + perpY * 8;
      edges += '<text x="' + labX + '" y="' + labY + '" fill="' + color + '" font-size="10.5" font-family="JetBrains Mono,monospace" opacity="' + opacity + '" text-anchor="middle">a' + aIdx + ':' + p.toFixed(1) + '</text>';
    };

    for (let s = 0; s < 3; s++) {
      for (let a = 0; a < 2; a++) {
        const trans = this.P[s + '_' + a];
        for (const t of trans) drawEdge(s, a, t.s, t.p);
      }
    }

    // 画节点
    let nodes = '';
    for (let i = 0; i < 3; i++) {
      const n = L[this.states[i]];
      const isHov = hov === i;
      const fill = isHov ? '#1e293b' : '#0f0f0f';
      const stroke = stateColors[i];
      const sw = isHov ? 3.5 : 2;
      nodes += '<g style="cursor:pointer" onmouseover="window._mdpInstance.setHover(' + i + ')" onmouseout="window._mdpInstance.setHover(null)">';
      nodes += '<circle cx="' + n.cx + '" cy="' + n.cy + '" r="32" fill="' + fill + '" stroke="' + stroke + '" stroke-width="' + sw + '"/>';
      nodes += '<text x="' + n.cx + '" y="' + (n.cy - 2) + '" text-anchor="middle" fill="' + stroke + '" font-size="15" font-weight="700">' + this.states[i] + '</text>';
      nodes += '<text x="' + n.cx + '" y="' + (n.cy + 14) + '" text-anchor="middle" fill="#a5f3fc" font-size="10" font-family="JetBrains Mono,monospace">V=' + this.V[i].toFixed(2) + '</text>';
      nodes += '</g>';
    }

    return '<svg viewBox="0 0 ' + w + ' ' + h + '" style="width:100%;height:auto;max-height:420px">' + defs + edges + nodes + '</svg>';
  }

  // ==================
  // 子表格渲染
  // ==================
  _renderRewardTable() {
    let rows = '';
    for (let i = 0; i < 3; i++) {
      let cells = '';
      for (let j = 0; j < 2; j++) {
        const v = this.R[i][j];
        const bg = v > 0 ? 'rgba(74,222,128,' + Math.min(0.15 + v * 0.25, 0.55) + ')' : (v < 0 ? 'rgba(248,113,113,0.3)' : '#141414');
        const color = v > 0 ? '#4ade80' : (v < 0 ? '#f87171' : '#888');
        cells += '<td style="padding:6px 12px;border:1px solid #333;background:' + bg + ';color:' + color + ';text-align:center">' + v.toFixed(1) + '</td>';
      }
      rows += '<tr><td style="padding:6px 12px;border:1px solid #333;color:#a5f3fc;font-weight:600;background:#111">' + this.states[i] + '</td>' + cells + '</tr>';
    }
    return '<table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.82rem;margin-top:4px">' +
      '<tr><th style="padding:6px 12px;border:1px solid #333;color:#888"></th><th style="padding:6px 12px;border:1px solid #3B82F6;color:#3B82F6">a0</th><th style="padding:6px 12px;border:1px solid #F97316;color:#F97316">a1</th></tr>' +
      rows + '</table>';
  }

  _renderTransitionTable() {
    let rows = '';
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 2; j++) {
        const trans = this.P[i + '_' + j];
        let tStr = trans.map(t => 'P(' + this.states[t.s] + ')=<span style="color:#a5f3fc">' + t.p.toFixed(1) + '</span>').join(' &nbsp; ');
        const ac = j === 0 ? '#3B82F6' : '#F97316';
        rows += '<tr>' +
          '<td style="padding:5px 10px;border:1px solid #333;color:#a5f3fc;font-weight:600;background:#111">' + this.states[i] + '</td>' +
          '<td style="padding:5px 10px;border:1px solid #333;color:' + ac + ';font-weight:600">' + this.actions[j] + '</td>' +
          '<td style="padding:5px 10px;border:1px solid #333;color:#e5e5e5">' + tStr + '</td>' +
          '</tr>';
      }
    }
    return '<table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.78rem;width:100%">' +
      '<tr><th style="padding:5px 10px;border:1px solid #333;color:#888">s</th>' +
      '<th style="padding:5px 10px;border:1px solid #333;color:#888">a</th>' +
      '<th style="padding:5px 10px;border:1px solid #333;color:#888">P(s\'|s,a)</th></tr>' + rows + '</table>';
  }

  _renderValueIterHistory() {
    const maxShow = Math.min(this.vHistory.length, 12);
    const start = Math.max(0, this.vHistory.length - maxShow);
    let rows = '';
    for (let k = start; k < this.vHistory.length; k++) {
      const v = this.vHistory[k];
      const vCells = v.map(x => {
        const bg = x > 0 ? 'rgba(74,222,128,' + Math.min(0.1 + x * 0.06, 0.45) + ')' : '#141414';
        return '<td style="padding:4px 8px;border:1px solid #333;background:' + bg + ';color:#a5f3fc;text-align:center">' + x.toFixed(3) + '</td>';
      }).join('');
      const marker = k === this.vHistory.length - 1 ? ' style="outline:1.5px solid #F97316;outline-offset:-1px"' : '';
      rows += '<tr' + marker + '><td style="padding:4px 8px;border:1px solid #333;color:#888">k=' + k + '</td>' + vCells + '</tr>';
    }
    return '<table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.75rem;width:100%">' +
      '<tr><th style="padding:4px 8px;border:1px solid #333;color:#888">iter</th>' +
      '<th style="padding:4px 8px;border:1px solid #333;color:#3B82F6">V(S0)</th>' +
      '<th style="padding:4px 8px;border:1px solid #333;color:#8B5CF6">V(S1)</th>' +
      '<th style="padding:4px 8px;border:1px solid #333;color:#F59E0B">V(S2)</th></tr>' + rows + '</table>';
  }

  _renderEpisodeTrajectory() {
    if (this.episode.trajectory.length === 0) {
      return '<div style="color:#666;font-size:0.82rem;padding:8px">点击"模拟一步"或"完整回合"开始。</div>';
    }
    let rows = '';
    for (const step of this.episode.trajectory) {
      const ac = step.a === 0 ? '#3B82F6' : '#F97316';
      const rcolor = step.r > 0 ? '#4ade80' : (step.r < 0 ? '#f87171' : '#888');
      rows += '<div style="display:flex;gap:8px;padding:5px 8px;margin:3px 0;background:#111;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.78rem;align-items:center;flex-wrap:wrap">' +
        '<span style="color:#888">k=' + step.k + '</span>' +
        '<span style="color:#a5f3fc">' + this.states[step.s] + '</span>' +
        '<span style="color:' + ac + '">' + this.actions[step.a] + '</span>' +
        '<span style="color:' + rcolor + '">r=' + step.r.toFixed(1) + '</span>' +
        '<span style="color:#666">→</span>' +
        '<span style="color:#a5f3fc">' + this.states[step.s_next] + '</span>' +
        '<span style="color:#8b949e;font-size:0.72rem">γ^' + step.k + '·r=' + step.discounted.toFixed(3) + '</span>' +
        '<span style="color:#888;font-size:0.7rem">[' + step.policy + ']</span>' +
        '</div>';
    }
    rows += '<div style="margin-top:8px;padding:8px;background:#0d1117;border:1px solid #21262d;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.84rem">' +
      '累计回报 G = <span style="color:#a5f3fc;font-weight:600">' + this.episode.G.toFixed(4) + '</span>' +
      (this.episode.done ? '  <span style="color:#888">(回合结束)</span>' : '') +
      '</div>';
    return rows;
  }

  _renderPolicyCards() {
    if (!this.vConverged) return '<div style="color:#666;font-size:0.82rem">先运行价值迭代至收敛，再查看最优策略。</div>';
    let out = '';
    for (let s = 0; s < 3; s++) {
      const qVals = [];
      for (let a = 0; a < 2; a++) {
        const trans = this.P[s + '_' + a];
        let expV = 0;
        for (const t of trans) expV += t.p * this.V[t.s];
        qVals.push(this.R[s][a] + this.gamma * expV);
      }
      const bestA = qVals[0] >= qVals[1] ? 0 : 1;
      const ac = bestA === 0 ? '#3B82F6' : '#F97316';
      out += '<div style="display:flex;align-items:center;gap:10px;padding:6px 10px;margin:4px 0;background:#111;border-radius:4px;border-left:3px solid ' + ac + '">' +
        '<span style="color:#a5f3fc;font-weight:600;font-family:JetBrains Mono,monospace">π*(' + this.states[s] + ')</span>' +
        '<span style="color:#666">=</span>' +
        '<span style="color:' + ac + ';font-weight:600;font-family:JetBrains Mono,monospace">' + this.actions[bestA] + '</span>' +
        '<span style="color:#888;font-family:JetBrains Mono,monospace;font-size:0.78rem">Q=[' + qVals[0].toFixed(3) + ', ' + qVals[1].toFixed(3) + ']</span>' +
        '</div>';
    }
    return out;
  }

  _renderBellmanExpansion() {
    // 展开一个示例：V(S1) 的 Bellman 右边计算
    const s = 1;
    let best = -Infinity, bestA = 0, details = [];
    for (let a = 0; a < 2; a++) {
      const trans = this.P[s + '_' + a];
      const termStrs = trans.map(t => t.p.toFixed(1) + '·[' + this.R[s][a].toFixed(1) + '+' + this.gamma + '·V(' + this.states[t.s] + ')]').join(' + ');
      let expV = 0;
      for (const t of trans) expV += t.p * this.V[t.s];
      const qv = this.R[s][a] + this.gamma * expV;
      const numeric = trans.map(t => t.p.toFixed(1) + '·[' + this.R[s][a].toFixed(1) + '+' + this.gamma + '·' + this.V[t.s].toFixed(3) + ']').join(' + ');
      details.push({ a, termStrs, numeric, qv });
      if (qv > best) { best = qv; bestA = a; }
    }
    let body = '<div style="color:#8b949e;font-size:0.82rem;margin-bottom:8px">以 <b style="color:#8B5CF6">S1</b> 为例，当前迭代 k=' + this.vIter + '：</div>';
    for (const d of details) {
      const ac = d.a === 0 ? '#3B82F6' : '#F97316';
      const winner = d.a === bestA ? ' background:rgba(74,222,128,0.08);border-left:3px solid #4ade80;' : '';
      body += '<div style="margin:4px 0;padding:6px 10px;border-radius:4px;' + winner + '">' +
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:' + ac + '">Q(S1, a' + d.a + ') = Σ P·[R + γ·V(s\')]</div>' +
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#a5f3fc;margin-top:2px">= ' + d.termStrs + '</div>' +
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#a5f3fc">= ' + d.numeric + ' = <b>' + d.qv.toFixed(4) + '</b></div>' +
        '</div>';
    }
    body += '<div style="margin-top:6px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#4ade80">V(S1)_new = max = ' + best.toFixed(4) + '  (选 a' + bestA + ')</div>';
    return body;
  }

  // ==================
  // 主渲染
  // ==================
  render() {
    if (!this.container) return;
    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._mdpInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._mdpInstance.play()">▶ 播放</button>';

    const convergedBadge = this.vConverged
      ? '<span style="color:#4ade80;font-size:0.78rem;margin-left:8px">● 已收敛 (k=' + this.vIter + ')</span>'
      : '<span style="color:#F59E0B;font-size:0.78rem;margin-left:8px">● 迭代中 (k=' + this.vIter + ')</span>';

    this.container.innerHTML =
      '<div class="mdp-viz">' +
        '<style>' +
          '.mdp-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.mdp-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;font-size:0.85rem}' +
          '.mdp-viz .ctrl-btn:hover{background:#252525;border-color:#3B82F6}' +
          '.mdp-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.mdp-viz .ctrl-btn.accent{border-color:#F97316;color:#F97316}' +
          '.mdp-viz .ctrl-btn.accent:hover{background:#F97316;color:#fff}' +
          '.mdp-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0;line-height:1.6}' +
          '.mdp-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.mdp-viz .edu-panel{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin-bottom:16px}' +
          '.mdp-viz .edu-title{font-size:1.02rem;font-weight:700;color:#fff;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #21262d}' +
          '.mdp-viz .edu-section{margin-bottom:12px}' +
          '.mdp-viz .edu-section-title{font-size:0.82rem;font-weight:600;color:#58a6ff;margin-bottom:4px}' +
          '.mdp-viz .edu-text{font-size:0.86rem;color:#c9d1d9;line-height:1.7}' +
          '.mdp-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}' +
          '.mdp-viz .three-col{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}' +
          '@media(max-width:900px){.mdp-viz .two-col,.mdp-viz .three-col{grid-template-columns:1fr}}' +
          '.mdp-viz .tag{display:inline-block;padding:1px 7px;border-radius:4px;font-size:0.7rem;font-family:JetBrains Mono,monospace;margin-right:4px}' +
          '.mdp-viz .tag-s{background:rgba(59,130,246,0.15);color:#3B82F6;border:1px solid rgba(59,130,246,0.3)}' +
          '.mdp-viz .tag-a{background:rgba(249,115,22,0.15);color:#F97316;border:1px solid rgba(249,115,22,0.3)}' +
          '.mdp-viz .tag-r{background:rgba(74,222,128,0.15);color:#4ade80;border:1px solid rgba(74,222,128,0.3)}' +
          '.mdp-viz .tag-p{background:rgba(139,92,246,0.15);color:#8B5CF6;border:1px solid rgba(139,92,246,0.3)}' +
          '.mdp-viz .tag-g{background:rgba(245,158,11,0.15);color:#F59E0B;border:1px solid rgba(245,158,11,0.3)}' +
        '</style>' +

        // 头部
        '<div style="font-size:1.5rem;font-weight:600;margin-bottom:4px">马尔可夫决策过程 (MDP)</div>' +
        '<div style="color:#888;margin-bottom:16px;font-size:0.9rem">' +
          '<span class="tag tag-s">S</span>' +
          '<span class="tag tag-a">A</span>' +
          '<span class="tag tag-p">P</span>' +
          '<span class="tag tag-r">R</span>' +
          '<span class="tag tag-g">γ</span>' +
          ' — 强化学习的数学语言' +
        '</div>' +

        // 教育面板 1：定义
        '<div class="edu-panel">' +
          '<div class="edu-title">📖 MDP 是什么？五元组 (S, A, P, R, γ)</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🎯 任何序列决策问题的数学框架</div>' +
            '<div class="edu-text">MDP 把"智能体—环境"交互形式化：智能体观察<b>状态</b>，选择<b>动作</b>，环境根据<b>转移概率</b>跳到下一个状态并给出<b>奖励</b>。无论是下棋、打游戏、控制机器人还是调度资源，几乎都可以写成 MDP。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">📌 五元组逐一解释</div>' +
            '<div class="edu-text">' +
              '<b style="color:#3B82F6">S（状态空间）</b>：所有可能的环境状态集合。本例 S = {S0, S1, S2}，共 3 个状态。<br>' +
              '<b style="color:#F97316">A（动作空间）</b>：智能体在每个状态可选的动作。本例每个状态有 2 个动作：a0 和 a1。<br>' +
              '<b style="color:#8B5CF6">P（转移概率）</b>：<code>P(s\'|s,a)</code> = 在状态 s 执行动作 a 后，转移到 s\' 的概率。<b>关键：动态是随机的</b>——现实世界永远有不确定性。同一个 (s, a) 下所有 s\' 的概率之和为 1。<br>' +
              '<b style="color:#4ade80">R（奖励函数）</b>：<code>R(s,a)</code> 或 <code>R(s,a,s\')</code> 表示在状态 s 执行动作 a 后即时奖励。本例 R(S2, a1)=2 是最"甜"的糖果。<br>' +
              '<b style="color:#F59E0B">γ（折扣因子）</b>：γ∈[0,1)。衡量"未来有多重要"。γ=0 → 短视；γ→1 → 高瞻远瞩。本例 γ=0.9。' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">🏗️ 马尔可夫性（Markov Property）</div>' +
            '<div class="edu-text">"给定现在，未来与过去无关"：<br>' +
              '<span style="font-family:JetBrains Mono,monospace;color:#a5f3fc">P(S_{t+1}|S_t, A_t) = P(S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0)</span><br>' +
              '这意味着：只要你把"所有决策所需信息"塞进状态，历史就可以被完全遗忘。这是一个<b>建模约束</b>——如果现实不满足，就需要扩充状态（如拼接历史帧）。' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 控制条
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">控制</div>' +
          playBtn +
          '<button class="ctrl-btn" onclick="window._mdpInstance.stepForward()">▸ 单步</button>' +
          '<button class="ctrl-btn" onclick="window._mdpInstance.stepBack()">◂ 回退</button>' +
          '<button class="ctrl-btn" onclick="window._mdpInstance.reset()">↻ 重置</button>' +
          '<select onchange="window._mdpInstance.setSpeed(this.value)" style="background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 10px;border-radius:6px;font-size:0.85rem;margin-right:8px">' +
            '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="3">3×</option>' +
          '</select>' +
          '<button class="ctrl-btn accent" onclick="window._mdpInstance.runValueIterationAll()">▶▶ 运行价值迭代至收敛</button>' +
          '<button class="ctrl-btn accent" onclick="window._mdpInstance.runFullEpisode()">🎲 模拟完整回合</button>' +
          '<button class="ctrl-btn" onclick="window._mdpInstance.resetEpisode()">↻ 重置回合</button>' +
          convergedBadge +
        '</div>' +

        // 状态图 + 奖励 + 转移表
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">🌐 状态转移图 <span style="color:#888;font-size:0.78rem;font-weight:400">(悬停状态高亮其出边)</span></div>' +
            this._renderGraph() +
            '<div style="margin-top:8px;font-size:0.78rem;color:#888;text-align:center">' +
              '<span style="color:#3B82F6">━━ a0</span> &nbsp;|&nbsp; <span style="color:#F97316">━━ a1</span> &nbsp;|&nbsp; 节点下方 V = 当前估计价值' +
            '</div>' +
          '</div>' +
          '<div>' +
            '<div class="section">' +
              '<div style="font-weight:600;margin-bottom:8px">🍬 奖励矩阵 R(s, a)</div>' +
              this._renderRewardTable() +
              '<div style="margin-top:8px;font-size:0.78rem;color:#888">最高奖励是 <b style="color:#4ade80">R(S2, a1)=2.0</b>，所以智能体应该学会到达 S2 并选 a1。</div>' +
            '</div>' +
            '<div class="section">' +
              '<div style="font-weight:600;margin-bottom:8px">🎲 转移概率 P(s\'|s,a)</div>' +
              this._renderTransitionTable() +
            '</div>' +
          '</div>' +
        '</div>' +

        // 核心公式：策略 / 回报 / 价值函数
        '<div class="edu-panel">' +
          '<div class="edu-title">🧮 核心定义：策略、回报、价值函数</div>' +
          '<div class="three-col">' +
            '<div>' +
              '<div class="edu-section-title">策略 π(a|s)</div>' +
              '<div class="formula-box">π(a|s) = P(A_t=a | S_t=s)</div>' +
              '<div class="edu-text" style="font-size:0.82rem">智能体的决策规则。<br>' +
                '• <b>确定性</b>：π(s) = a（每个状态一个确定动作）<br>' +
                '• <b>随机性</b>：对动作给概率分布<br>' +
                '最优策略 π* 最大化期望回报。' +
              '</div>' +
            '</div>' +
            '<div>' +
              '<div class="edu-section-title">回报（Return）</div>' +
              '<div class="formula-box">G_t = R_t + γR_{t+1} + γ²R_{t+2} + ...<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= Σ_{k=0}^∞ γ^k · R_{t+k}</div>' +
              '<div class="edu-text" style="font-size:0.82rem">从时刻 t 开始，未来所有奖励的折扣加权和。<br>γ 越接近 1，越重视长期；γ 越接近 0，越"贪心即时"。</div>' +
            '</div>' +
            '<div>' +
              '<div class="edu-section-title">价值函数 V(s) / Q(s,a)</div>' +
              '<div class="formula-box">V_π(s) = E_π[G_t | S_t=s]<br>Q_π(s,a) = E_π[G_t | S_t=s, A_t=a]</div>' +
              '<div class="edu-text" style="font-size:0.82rem">V：状态价值（在 s 按 π 玩下去期望多少回报）<br>Q：动作价值（在 s 先做 a，再按 π 玩）<br>关系：V(s) = Σ_a π(a|s) Q(s,a)</div>' +
            '</div>' +
          '</div>' +
        '</div>' +

        // Bellman 方程
        '<div class="edu-panel">' +
          '<div class="edu-title">🔑 Bellman 方程：价值的"递归"定义</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">Bellman 期望方程（给定策略 π）</div>' +
            '<div class="formula-box">V_π(s) = Σ_a π(a|s) Σ_{s\'} P(s\'|s,a) [R(s,a) + γ · V_π(s\')]</div>' +
            '<div class="edu-text" style="font-size:0.85rem">意思：在 s 的价值 = 按 π 选动作，环境转移，得到即时奖励 + γ·下一步价值，取期望。<br><b>递归关系</b>——V(s) 用 V(s\') 定义，形成线性方程组。</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">Bellman 最优方程（最优策略 π*）</div>' +
            '<div class="formula-box">V*(s) = max_a Σ_{s\'} P(s\'|s,a) [R(s,a) + γ · V*(s\')]<br>Q*(s,a) = Σ_{s\'} P(s\'|s,a) [R(s,a) + γ · max_{a\'} Q*(s\',a\')]</div>' +
            '<div class="edu-text" style="font-size:0.85rem">把 "Σ_a π(a|s)" 换成 "max_a"——最优策略永远选价值最高的动作。<br>这是<b>非线性</b>方程（因为 max），没有闭式解，需要<b>迭代求解</b>——这就是价值迭代。</div>' +
          '</div>' +
        '</div>' +

        // Bellman 价值迭代演示
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">🔁 价值迭代当前步展开</div>' +
            this._renderBellmanExpansion() +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">📈 V(s) 迭代历史（最近 12 步）</div>' +
            this._renderValueIterHistory() +
            '<div style="margin-top:8px;font-size:0.78rem;color:#888">所有状态价值稳定（相邻两步差 &lt; 1e-4）即视为收敛。</div>' +
          '</div>' +
        '</div>' +

        // 最优策略
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">🏆 最优策略 π*(s)</div>' +
          this._renderPolicyCards() +
        '</div>' +

        // 回合模拟
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">🎬 回合轨迹模拟 (' + (this.vConverged ? '使用贪心策略' : '随机策略') + ')</div>' +
          '<div style="color:#888;font-size:0.82rem;margin-bottom:6px">每步显示：(s_k, a_k, r_k) → s_{k+1}，并累计折扣回报 G。</div>' +
          this._renderEpisodeTrajectory() +
        '</div>' +

        // 其他教育面板
        '<div class="edu-panel">' +
          '<div class="edu-title">💡 常见问题与陷阱</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">为什么需要折扣因子 γ？</div>' +
            '<div class="edu-text">1) <b>数学收敛</b>：无限时域下 Σ R 可能发散，但 Σ γ^k R 是有限的（只要奖励有界）。<br>' +
              '2) <b>偏好近期</b>：现在的 1 块钱比 10 年后的 1 块钱更值钱（金融类比）。<br>' +
              '3) <b>体现不确定性</b>：未来越远越不可预测，应打折。' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">确定性策略 vs 随机性策略</div>' +
            '<div class="edu-text">MDP 的最优策略<b>一定存在确定性版本</b>（定理保证）。但在以下场景随机策略更好：<br>' +
              '• <b>部分可观测（POMDP）</b>：状态信息不完整时，随机能避免被"卡住"<br>' +
              '• <b>对抗环境</b>：石头剪刀布里确定性策略必败<br>' +
              '• <b>Policy Gradient</b>：连续动作空间天然用随机策略（高斯分布）' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">V 和 Q 哪个更好用？</div>' +
            '<div class="edu-text">• <b>V(s)</b>：只需 |S| 个值，但行动时要知道 P 和 R 才能选动作（model-based）<br>' +
              '• <b>Q(s,a)</b>：|S|·|A| 个值，但直接 argmax_a Q(s,a) 就能行动（model-free 首选）<br>' +
              '→ 这就是为什么 Q-Learning / DQN 学 Q 而不是 V。' +
            '</div>' +
          '</div>' +
          '<div class="edu-section">' +
            '<div class="edu-section-title">MDP 求解方法小结</div>' +
            '<div class="edu-text">' +
              '1) <b>已知 P 和 R</b>（model-based）：<br>' +
              '&nbsp;&nbsp;• 价值迭代（Value Iteration）——本页演示<br>' +
              '&nbsp;&nbsp;• 策略迭代（Policy Iteration）<br>' +
              '&nbsp;&nbsp;• 线性规划<br>' +
              '2) <b>未知 P 和 R</b>（model-free，真正的 RL）：<br>' +
              '&nbsp;&nbsp;• 蒙特卡洛（MC）：完整回合回报平均<br>' +
              '&nbsp;&nbsp;• 时序差分（TD）：<b>Q-Learning、SARSA</b><br>' +
              '&nbsp;&nbsp;• 函数逼近 + 深度网络 → DQN、PPO 等' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 折扣回报手算示例
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">🧮 折扣回报手算示例 (γ=0.9)</div>' +
          '<div class="formula-box">若轨迹：S0 --a0(r=0)--> S1 --a0(r=1)--> S2 --a1(r=2)--> 终态</div>' +
          '<div class="formula-box">G = 0·γ⁰ + 1·γ¹ + 2·γ² = 0 + 0.9 + 1.62 = <span style="color:#4ade80">2.52</span></div>' +
          '<div style="color:#888;font-size:0.82rem;margin-top:6px">不同 γ 下同一轨迹的 G：γ=0 → 0；γ=0.5 → 1.0；γ=0.9 → 2.52；γ=0.99 → 2.86。可见 γ 越接近 1 越重视远期奖励。</div>' +
        '</div>' +
      '</div>';

    window._mdpInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._mdpInstance === this) {
      try { delete window._mdpInstance; } catch (e) { window._mdpInstance = null; }
    }
    this.container = null;
  }
}

window.RLMDPTutorial = RLMDPTutorial;
