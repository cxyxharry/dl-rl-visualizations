// Self-Attention 可视化
// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
// 固定参数: d_model = 4, seq_len = 3, d_k = d_v = 4
class AttentionVisualization {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // 交互态：用户可点击任一 Score/Attention/Output 单元格查看其计算细节
    this.selectedCell = { kind: 'score', i: 0, j: 1 };
    // Softmax 展开专注哪一行
    this.softmaxRow = 0;
    // 是否启用 causal mask 演示
    this.showMask = false;

    this.d_model = 4;
    this.seq_len = 3;
    this.d_k = 4;

    this.tokens = ['The', 'cat', 'sat'];
    this.tokenColors = ['#3B82F6', '#8B5CF6', '#10B981']; // 每个 token 一个颜色

    // 输入 X (3×4) —— 每行是一个 token 的嵌入
    this.X = [
      [1.0, 0.5, 0.3, 0.2],
      [0.8, 1.0, 0.5, 0.1],
      [0.3, 0.7, 1.0, 0.4]
    ];

    this.WQ = [
      [ 1.0,  0.0,  0.5, -0.2],
      [ 0.0,  1.0, -0.3,  0.4],
      [ 0.5, -0.4,  1.0,  0.1],
      [-0.2,  0.3,  0.2,  1.0]
    ];
    this.WK = [
      [ 1.0,  0.2, -0.1,  0.0],
      [ 0.1,  1.0,  0.3, -0.2],
      [-0.2,  0.4,  1.0,  0.5],
      [ 0.3, -0.1,  0.2,  1.0]
    ];
    this.WV = [
      [ 1.0, -0.1,  0.0,  0.3],
      [ 0.2,  1.0,  0.4, -0.1],
      [ 0.0,  0.3,  1.0,  0.2],
      [-0.3,  0.1, -0.2,  1.0]
    ];

    this.STEPS = [
      { short: '① 输入',        long: '输入 X (3×4)' },
      { short: '② Q/K/V',       long: '三路投影: Q = X·W_Q, K = X·W_K, V = X·W_V' },
      { short: '③ Score',        long: 'Score = Q · Kᵀ  (两两相关度)' },
      { short: '④ Scale',        long: 'ScaledScore = Score / √d_k' },
      { short: '⑤ Mask*',        long: '（可选）Causal Mask —— Decoder 中屏蔽未来' },
      { short: '⑥ Softmax',      long: 'Attention = softmax(·) —— 按行归一化为概率' },
      { short: '⑦ Output',       long: 'Output = Attention · V —— 加权聚合' },
      { short: '⑧ 全貌',         long: '整体回顾 + 注意力流向图' }
    ];
  }

  // ---------- 矩阵运算（共享：见 js/utils/math.js） ----------
  // matmul / transpose / scale / softmax 都来自 Utils.*

  _applyMask(M) {
    // Causal mask: 上三角位置（j > i）设为 -∞（约等于 -1e9 后 softmax 后为 0）
    return M.map((row, i) => row.map((v, j) => j > i ? -1e9 : v));
  }

  // ---------- 生命周期 ----------
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this._compute();
    this.render();
    return this;
  }

  _compute() {
    const U = window.Utils;
    this.Q = U.matmul(this.X, this.WQ);
    this.K = U.matmul(this.X, this.WK);
    this.V = U.matmul(this.X, this.WV);
    this.KT = U.transpose(this.K);
    this.Score = U.matmul(this.Q, this.KT);
    this.sqrt_dk = Math.sqrt(this.d_k);
    this.ScaledScore = U.scale(this.Score, this.sqrt_dk);
    this.MaskedScore = this._applyMask(this.ScaledScore);
    this.Attn = U.softmax(this.ScaledScore);
    this.AttnMasked = U.softmax(this.MaskedScore);
    this.Output = U.matmul(this.Attn, this.V);
  }

  reset() {
    this.currentStep = 0;
    this.isPlaying = false;
    clearTimeout(this.timer);
    this.selectedCell = { kind: 'score', i: 0, j: 1 };
    this.softmaxRow = 0;
    this.showMask = false;
    this.render();
  }
  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  setSpeed(s) { this.speed = Number(s) || 1; }
  stepForward() { if (this.currentStep < this.STEPS.length - 1) { this.currentStep++; this.render(); } }
  stepBack() { if (this.currentStep > 0) { this.currentStep--; this.render(); } }
  goTo(i) { this.currentStep = Math.max(0, Math.min(i, this.STEPS.length - 1)); this.render(); }
  selectCell(kind, i, j) {
    this.selectedCell = { kind, i, j };
    if (kind === 'attn' || kind === 'softmax') this.softmaxRow = i;
    this.render();
  }
  setSoftmaxRow(i) { this.softmaxRow = i; this.render(); }
  toggleMask() { this.showMask = !this.showMask; this.render(); }

  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this.render();
      this.timer = setTimeout(() => this._auto(), 1700 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  // ---------- 通用渲染辅助 ----------
  // _fmt 已迁移到 Utils.fmt（见 js/utils/format.js），保留 this._fmt 作为薄封装以兼容旧调用点
  _fmt(v, d) { return window.Utils.fmt(v, d); }

  _bgColor(v, scale) {
    scale = scale || 1.5;
    if (Math.abs(v) < 1e-9) return '#0d0d0d';
    if (!Number.isFinite(v) || v < -1e8) return '#1a0505';
    const a = Math.min(0.15 + Math.abs(v) / scale * 0.45, 0.6);
    return v > 0 ? 'rgba(16,185,129,' + a.toFixed(3) + ')' : 'rgba(239,68,68,' + a.toFixed(3) + ')';
  }

  _intensity(v) {
    const a = Math.min(0.08 + v * 0.85, 0.92);
    return 'rgba(249,115,22,' + a.toFixed(3) + ')';
  }

  // 矩阵渲染（支持交互式选择）
  _matrixHTML(M, opts) {
    opts = opts || {};
    const {
      name = '',
      color = '#3B82F6',
      rowLabels = null,
      colLabels = null,
      highlightRow = -1,
      highlightCol = -1,
      highlightCell = null,
      intensityMode = false,
      visible = true,
      scale = 1.5,
      cellSize = 48,
      decimals = 2,
      dim = null,
      clickable = null,       // {kind}  若设置，则每个 cell 可点击调用 selectCell(kind, i, j)
      selectedCell = null,    // 已选中的单元 [i,j]
      maskUpper = false       // 是否绘制上三角 mask 覆盖层
    } = opts;

    const rows = M.length, cols = M[0].length;
    const dimStr = dim || (rows + '×' + cols);

    let head = '';
    if (colLabels) {
      head += '<div style="display:flex;margin-left:' + (rowLabels ? 56 : 0) + 'px">';
      for (let j = 0; j < cols; j++) {
        head += '<div style="width:' + cellSize + 'px;text-align:center;font-size:0.68rem;color:#888;font-family:JetBrains Mono,monospace">' + colLabels[j] + '</div>';
      }
      head += '</div>';
    }

    let body = '';
    for (let i = 0; i < rows; i++) {
      body += '<div style="display:flex;align-items:center">';
      if (rowLabels) {
        const rlColor = typeof rowLabels[i] === 'string' && this.tokens.includes(rowLabels[i])
          ? this.tokenColors[this.tokens.indexOf(rowLabels[i])] : color;
        body += '<div style="width:56px;font-size:0.72rem;color:' + rlColor + ';font-family:JetBrains Mono,monospace;text-align:right;padding-right:6px;font-weight:600">' + rowLabels[i] + '</div>';
      }
      for (let j = 0; j < cols; j++) {
        const v = M[i][j];
        const hlRow = (highlightRow === i);
        const hlCol = (highlightCol === j);
        const hlCell = highlightCell && highlightCell[0] === i && highlightCell[1] === j;
        const isSel  = selectedCell && selectedCell[0] === i && selectedCell[1] === j;
        const masked = maskUpper && j > i;

        let bg;
        if (!visible) bg = '#0a0a0a';
        else if (masked) bg = '#1a0505';
        else if (intensityMode) bg = this._intensity(v);
        else bg = this._bgColor(v, scale);

        let border = '1px solid ' + color;
        let boxShadow = '';
        if (hlCell || isSel) { border = '2px solid #FBBF24'; boxShadow = '0 0 12px rgba(251,191,36,0.75)'; }
        else if (hlRow || hlCol) { border = '2px solid #FBBF24'; }

        const txt = visible ? (masked ? '−∞' : this._fmt(v, decimals)) : '?';
        const txtColor = visible ? (masked ? '#EF4444' : '#e5e5e5') : '#444';

        const cursor = (clickable && !masked) ? 'pointer' : 'default';
        const onclick = (clickable && !masked)
          ? ' onclick="window._attnInstance.selectCell(\'' + clickable.kind + '\',' + i + ',' + j + ')"'
          : '';

        body += '<div' + onclick + ' style="width:' + cellSize + 'px;height:' + cellSize + 'px;margin:1px;background:' + bg +
          ';border:' + border + ';border-radius:4px;color:' + txtColor +
          ';font-family:JetBrains Mono,monospace;font-size:0.72rem;display:flex;align-items:center;justify-content:center;cursor:' + cursor + ';' +
          (boxShadow ? 'box-shadow:' + boxShadow + ';' : '') +
          'transition:border-color .15s, box-shadow .15s">' + txt + '</div>';
      }
      body += '</div>';
    }

    const title = name
      ? '<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">' +
        '<span style="color:' + color + ';font-weight:600;font-size:0.88rem">' + name + '</span>' +
        '<span style="color:#666;font-size:0.72rem;font-family:JetBrains Mono,monospace">(' + dimStr + ')</span>' +
        (clickable ? '<span style="color:#888;font-size:0.68rem;margin-left:auto">🖱️ 点击单元格查看</span>' : '') +
        '</div>'
      : '';

    return '<div class="matrix-wrap">' + title + head + body + '</div>';
  }

  // ---------- 柱状图: 概率分布 ----------
  _barChart(vec, labels, colors, selectedIdx) {
    const mx = Math.max(...vec, 0.001);
    return '<div style="display:flex;gap:8px;align-items:flex-end;height:100px;padding:8px 4px">' +
      vec.map((v, j) => {
        const h = Math.max(3, (v / mx) * 82);
        const c = colors ? colors[j] : '#F97316';
        const sel = j === selectedIdx;
        return '<div style="display:flex;flex-direction:column;align-items:center;gap:3px">' +
          '<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#ccc">' + v.toFixed(3) + '</div>' +
          '<div style="width:42px;height:' + h + 'px;background:' + c + ';border:' + (sel ? '2px solid #FBBF24' : '1px solid #333') + ';border-radius:3px 3px 0 0;opacity:' + (0.45 + v * 0.5).toFixed(2) + '"></div>' +
          '<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:' + c + '">' + (labels ? labels[j] : j) + '</div>' +
          '</div>';
      }).join('') +
      '</div>';
  }

  // ---------- 注意力流向图（SVG, 双行 token + 曲线连接） ----------
  _flowDiagram(attnMatrix, selectedQueryRow) {
    const tok = this.tokens;
    const N = tok.length;
    const w = 560, h = 220;
    const pad = 40;
    const innerW = w - 2 * pad;
    const xq = i => pad + (i + 0.5) * innerW / N;           // query 层 x 坐标
    const xk = j => pad + (j + 0.5) * innerW / N;           // key 层 x 坐标
    const yq = 30;                                           // query 在上
    const yk = h - 30;                                       // key 在下
    // tokens
    const nodeFor = (t, x, y, label, color) =>
      '<g>' +
        '<rect x="' + (x - 38) + '" y="' + (y - 14) + '" width="76" height="28" rx="6" ry="6" fill="#0d0d0d" stroke="' + color + '" stroke-width="1.5"/>' +
        '<text x="' + x + '" y="' + (y + 5) + '" text-anchor="middle" fill="' + color + '" font-family="Inter,sans-serif" font-size="12" font-weight="600">' + t + '</text>' +
        '<text x="' + x + '" y="' + (y + (label === 'Q' ? -22 : 32)) + '" text-anchor="middle" fill="#888" font-family="JetBrains Mono,monospace" font-size="10">' + label + '</text>' +
      '</g>';

    let nodes = '';
    for (let i = 0; i < N; i++) {
      nodes += nodeFor(tok[i], xq(i), yq, 'Q[' + tok[i] + ']', this.tokenColors[i]);
      nodes += nodeFor(tok[i], xk(i), yk, 'K[' + tok[i] + ']·V[' + tok[i] + ']', this.tokenColors[i]);
    }

    // 连线：从 Q[i] 到 K[j]，粗细 / 透明度 = attn[i][j]
    let edges = '';
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const w_ij = attnMatrix[i][j];
        if (w_ij < 0.02) continue;
        const x1 = xq(i), y1 = yq + 14, x2 = xk(j), y2 = yk - 14;
        const cx1 = x1, cy1 = y1 + (y2 - y1) * 0.5;
        const cx2 = x2, cy2 = y2 - (y2 - y1) * 0.5;
        const strokeW = Math.max(1, w_ij * 8);
        const opacity = Math.min(0.12 + w_ij * 0.85, 0.95);
        const isSelected = (selectedQueryRow === i);
        const color = isSelected ? '#F97316' : (selectedQueryRow === -1 ? '#F97316' : '#888');
        const effOpacity = (selectedQueryRow === -1 || isSelected) ? opacity : 0.15;
        edges += '<path d="M' + x1 + ',' + y1 + ' C' + cx1 + ',' + cy1 + ' ' + cx2 + ',' + cy2 + ' ' + x2 + ',' + y2 + '" fill="none" stroke="' + color + '" stroke-width="' + strokeW.toFixed(2) + '" opacity="' + effOpacity.toFixed(2) + '"/>';
        // 若是选中的行，把权重标注在中点
        if (isSelected) {
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + (j - i) * 4;
          edges += '<text x="' + mx + '" y="' + my + '" text-anchor="middle" fill="#FBBF24" font-family="JetBrains Mono,monospace" font-size="10" font-weight="700" style="paint-order:stroke;stroke:#0f0f0f;stroke-width:3">' + w_ij.toFixed(2) + '</text>';
        }
      }
    }

    const legend =
      '<text x="' + (w - 10) + '" y="14" text-anchor="end" fill="#888" font-family="Inter,sans-serif" font-size="10">线粗 ∝ 注意力权重</text>';

    // 行选择器
    const rowButtons = tok.map((t, i) =>
      '<button onclick="window._attnInstance.setSoftmaxRow(' + i + ')" style="background:' + (selectedQueryRow === i ? this.tokenColors[i] : '#1a1a1a') +
        ';color:' + (selectedQueryRow === i ? '#fff' : '#aaa') +
        ';border:1px solid ' + this.tokenColors[i] + ';padding:4px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.75rem;cursor:pointer;margin-right:4px">Q[' + t + ']</button>'
    ).join('');
    const allBtn = '<button onclick="window._attnInstance.setSoftmaxRow(-1)" style="background:' + (selectedQueryRow === -1 ? '#F97316' : '#1a1a1a') +
      ';color:' + (selectedQueryRow === -1 ? '#fff' : '#aaa') +
      ';border:1px solid #F97316;padding:4px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.75rem;cursor:pointer;margin-right:8px">全部</button>';

    return '<div>' +
      '<div style="margin-bottom:8px;color:#aaa;font-size:0.82rem">选择查看某个 token 作为 Query 时的注意力分布：' + allBtn + rowButtons + '</div>' +
      '<svg viewBox="0 0 ' + w + ' ' + h + '" style="width:100%;max-width:' + w + 'px;height:auto;background:#0a0a0a;border:1px solid #222;border-radius:8px;display:block">' +
        '<defs></defs>' +
        edges +
        nodes +
        legend +
      '</svg>' +
      '</div>';
  }

  // ---------- Softmax 按行展开 ----------
  _softmaxBreakdown(row, rowLabel) {
    const tok = this.tokens;
    const mx = Math.max(...row);
    const ex = row.map(v => Math.exp(v - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    const pr = ex.map(v => v / sum);

    const header = '<thead><tr style="color:#888;font-size:0.75rem">' +
      '<th style="text-align:left;padding:4px 6px">target</th>' +
      tok.map(t => '<th style="padding:4px 6px">→ ' + t + '</th>').join('') +
      '<th style="padding:4px 6px;color:#666">Σ</th></tr></thead>';
    const rowCells = (label, values, color, fmt, extra) =>
      '<tr>' +
        '<td style="padding:4px 6px;color:' + color + ';font-weight:600;font-family:JetBrains Mono,monospace;font-size:0.78rem">' + label + '</td>' +
        values.map(v => '<td style="padding:4px 6px;text-align:center;font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#e5e5e5">' + fmt(v) + '</td>').join('') +
        '<td style="padding:4px 6px;color:#888;text-align:center;font-family:JetBrains Mono,monospace;font-size:0.78rem">' + (extra != null ? extra : '') + '</td>' +
      '</tr>';

    return '<table style="border-collapse:collapse;margin-top:6px">' + header + '<tbody>' +
      rowCells('zⱼ (原始)', row, '#F97316', v => v.toFixed(3), '') +
      rowCells('zⱼ − max', row.map(v => v - mx), '#888', v => v.toFixed(3), '（数值稳定）') +
      rowCells('exp(·)', ex, '#FBBF24', v => v.toFixed(3), sum.toFixed(3)) +
      rowCells('÷ Σ = P(j)', pr, '#10B981', v => v.toFixed(3), pr.reduce((a, b) => a + b, 0).toFixed(3)) +
      '</tbody></table>' +
      '<div style="color:#888;font-size:0.75rem;margin-top:6px">📌 行 Q[<b style="color:' + this.tokenColors[this.softmaxRow] + '">' + rowLabel + '</b>] 的 softmax 展开：先减去 max（数值稳定 trick）、再 exp、再除和。最后一行就是<b>注意力权重</b>。</div>';
  }

  // ---------- 点积展开 ----------
  _dotBreakdown(a, b, labelA, labelB) {
    const terms = a.map((x, k) => x.toFixed(2) + '·' + b[k].toFixed(2));
    const result = a.reduce((s, x, k) => s + x * b[k], 0);
    return '<div class="formula-box" style="font-size:0.78rem">' +
      labelA + ' · ' + labelB + ' = ' + terms.join(' + ') +
      ' = <span style="color:#FBBF24;font-weight:600">' + result.toFixed(3) + '</span>' +
      '</div>';
  }

  // ---------- 主渲染 ----------
  render() {
    if (!this.container) return;
    const st = this.currentStep;
    const tok = this.tokens;
    const dimLbl = ['d₀', 'd₁', 'd₂', 'd₃'];

    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._attnInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._attnInstance.play()">▶ 播放</button>';

    // 步骤进度条
    const stepDots = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff;border-color:#3B82F6'
                    : (i < st ? 'background:#1e3a5f;color:#93c5fd;border-color:#2563eb'
                              : 'background:#0f0f0f;color:#666;border-color:#333');
      return '<div onclick="window._attnInstance.goTo(' + i + ')" ' +
        'title="' + s.long + '" ' +
        'style="padding:6px 10px;border:1px solid;border-radius:20px;font-size:0.72rem;cursor:pointer;white-space:nowrap;transition:all .15s;' + active + '">' +
        s.short + '</div>';
    }).join('');

    // ---------- 每步内容 ----------
    let detail = '';
    let mainViz = '';

    if (st === 0) {
      detail =
        '<div class="formula-box">X ∈ ℝ<sup>' + this.seq_len + '×' + this.d_model + '</sup> —— 每一行是一个 token 的嵌入向量（embedding + positional encoding）</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:8px">' +
          '• 序列长度 <b>seq_len = 3</b>: ' + tok.map((t, i) => '<span style="color:' + this.tokenColors[i] + ';font-weight:600">"' + t + '"</span>').join(', ') + '<br>' +
          '• 嵌入维度 <b>d_model = 4</b>（实际模型通常 512~12288）<br>' +
          '• X 的每一行就是一个 token 的"语义坐标"，后续所有计算都以它为起点' +
        '</div>';
      mainViz =
        '<div style="display:flex;justify-content:center;padding:12px 6px">' +
          this._matrixHTML(this.X, { name: 'X (输入嵌入)', color: '#e5e5e5', rowLabels: tok, colLabels: dimLbl, cellSize: 58, decimals: 2 }) +
        '</div>';
    }
    else if (st === 1) {
      // Q/K/V 三路并行投影
      const r0 = 0, c0 = 0;
      const xRow = this.X[r0];
      const qCol = this.WQ.map(r => r[c0]);
      detail =
        '<div class="formula-box">Q = X · W_Q &nbsp; · &nbsp; K = X · W_K &nbsp; · &nbsp; V = X · W_V &nbsp;&nbsp; 形状: (3×4) · (4×4) = (3×4)</div>' +
        this._dotBreakdown(xRow, qCol, 'X[' + tok[r0] + ']', 'W_Q[:,0]') +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '<b style="color:#3B82F6">Q（查询）</b>: "我要找什么信息？"<br>' +
          '<b style="color:#8B5CF6">K（键）</b>: "我是什么类型的 token？"（给别人识别）<br>' +
          '<b style="color:#10B981">V（值）</b>: "我真正的语义内容是什么？"<br>' +
          '三个权重矩阵都从同一个 X 投影，但分别学习到<b>不同用途</b>的子空间。' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:flex-start;justify-content:center;gap:14px;flex-wrap:wrap;padding:8px">' +
          '<div style="border:1px solid #222;border-radius:8px;padding:10px;background:rgba(59,130,246,0.03)">' +
            '<div style="color:#3B82F6;font-weight:600;font-size:0.85rem;margin-bottom:6px">🔷 Q = X · W_Q</div>' +
            '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">' +
              this._matrixHTML(this.X, { name: 'X', color: '#e5e5e5', rowLabels: tok, cellSize: 34, decimals: 1, highlightRow: 0 }) +
              '<span style="color:#888">·</span>' +
              this._matrixHTML(this.WQ, { name: 'W_Q', color: '#3B82F6', cellSize: 34, decimals: 2, highlightCol: 0 }) +
              '<span style="color:#888">=</span>' +
              this._matrixHTML(this.Q, { name: 'Q', color: '#3B82F6', rowLabels: tok, cellSize: 38, decimals: 2, highlightCell: [0, 0] }) +
            '</div>' +
          '</div>' +
          '<div style="border:1px solid #222;border-radius:8px;padding:10px;background:rgba(139,92,246,0.03)">' +
            '<div style="color:#8B5CF6;font-weight:600;font-size:0.85rem;margin-bottom:6px">🟣 K = X · W_K</div>' +
            '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">' +
              this._matrixHTML(this.X, { name: 'X', color: '#e5e5e5', rowLabels: tok, cellSize: 34, decimals: 1 }) +
              '<span style="color:#888">·</span>' +
              this._matrixHTML(this.WK, { name: 'W_K', color: '#8B5CF6', cellSize: 34, decimals: 2 }) +
              '<span style="color:#888">=</span>' +
              this._matrixHTML(this.K, { name: 'K', color: '#8B5CF6', rowLabels: tok, cellSize: 38, decimals: 2 }) +
            '</div>' +
          '</div>' +
          '<div style="border:1px solid #222;border-radius:8px;padding:10px;background:rgba(16,185,129,0.03)">' +
            '<div style="color:#10B981;font-weight:600;font-size:0.85rem;margin-bottom:6px">🟢 V = X · W_V</div>' +
            '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">' +
              this._matrixHTML(this.X, { name: 'X', color: '#e5e5e5', rowLabels: tok, cellSize: 34, decimals: 1 }) +
              '<span style="color:#888">·</span>' +
              this._matrixHTML(this.WV, { name: 'W_V', color: '#10B981', cellSize: 34, decimals: 2 }) +
              '<span style="color:#888">=</span>' +
              this._matrixHTML(this.V, { name: 'V', color: '#10B981', rowLabels: tok, cellSize: 38, decimals: 2 }) +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div style="text-align:center;color:#888;font-size:0.78rem;margin-top:8px">上方黄色示例：计算 Q[' + tok[0] + '][0]，就是 X 的第 0 行与 W_Q 的第 0 列的点积。</div>';
    }
    else if (st === 2) {
      // Score = Q · K^T, 交互式
      const sel = this.selectedCell;
      const si = (sel && sel.kind === 'score') ? sel.i : 0;
      const sj = (sel && sel.kind === 'score') ? sel.j : 1;
      const qi = this.Q[si];
      const kj = this.K[sj];
      const terms = qi.map((v, k) => v.toFixed(2) + '·' + kj[k].toFixed(2));
      const prod = qi.reduce((s, v, k) => s + v * kj[k], 0);

      detail =
        '<div class="formula-box">Score = Q · Kᵀ &nbsp;&nbsp; 形状: (3×4) · (4×3) = (3×3)</div>' +
        '<div class="formula-box" style="font-size:0.78rem">Score[' +
          '<span style="color:' + this.tokenColors[si] + '">' + tok[si] + '</span>→' +
          '<span style="color:' + this.tokenColors[sj] + '">' + tok[sj] + '</span>' +
          '] = Q[' + tok[si] + '] · K[' + tok[sj] + '] = ' + terms.join(' + ') +
          ' = <span style="color:#FBBF24;font-weight:600">' + prod.toFixed(3) + '</span></div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">' +
          '<b>Score[i][j]</b> = token i 的 Query 与 token j 的 Key 的点积。数值大表示<b>两个 token 在这一子空间中很相关</b>。<br>' +
          '🖱️ <b>点击下方 Score 矩阵的任意单元格</b>查看其具体计算。' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(this.Q, { name: 'Q', color: '#3B82F6', rowLabels: tok, cellSize: 42, decimals: 2, highlightRow: si }) +
          '<div style="color:#888;font-size:1.4rem">·</div>' +
          this._matrixHTML(this.KT, { name: 'Kᵀ', color: '#8B5CF6', colLabels: tok, cellSize: 42, decimals: 2, dim: '4×3', highlightCol: sj }) +
          '<div style="color:#888;font-size:1.4rem">=</div>' +
          this._matrixHTML(this.Score, { name: 'Score', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 56, decimals: 2, scale: 4,
            clickable: { kind: 'score' }, selectedCell: [si, sj] }) +
        '</div>';
    }
    else if (st === 3) {
      // Scale
      // 计算"放大倍数"的直观比较
      const maxOrig = Math.max(...this.Score.flat().map(Math.abs));
      const maxScaled = maxOrig / this.sqrt_dk;
      detail =
        '<div class="formula-box">ScaledScore = Score / √d_k &nbsp;&nbsp; d_k = ' + this.d_k + ', √d_k = ' + this.sqrt_dk.toFixed(3) + '</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '<b style="color:#FBBF24">为什么要除以 √d_k？</b><br>' +
          '• Q·K 是 d_k 个独立项的和；若 Q, K 每维 ~N(0,1)，那么点积 ~ N(0, d_k)，<b>方差随维度线性增长</b>。<br>' +
          '• 未缩放的大点积会让 softmax <b>极度尖锐</b>（一个位置接近 1，其余几乎为 0）。<br>' +
          '• 极端分布处梯度 ≈ 0 → 训练无法进行。<br>' +
          '• 除以 √d_k 让方差回到 O(1)，softmax 分布平滑，梯度健康。<br>' +
          '📏 本例中 max|Score| = <b>' + maxOrig.toFixed(2) + '</b> → 缩放后 max|ScaledScore| = <b>' + maxScaled.toFixed(2) + '</b>' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:center;justify-content:center;gap:22px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(this.Score, { name: 'Score', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 58, decimals: 2, scale: 4 }) +
          '<div style="display:flex;flex-direction:column;align-items:center;gap:3px">' +
            '<div style="color:#aaa;font-size:1.5rem">÷</div>' +
            '<div style="color:#FBBF24;font-family:JetBrains Mono,monospace;font-size:0.9rem">√' + this.d_k + ' = ' + this.sqrt_dk.toFixed(3) + '</div>' +
          '</div>' +
          this._matrixHTML(this.ScaledScore, { name: 'ScaledScore', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 58, decimals: 2, scale: 2 }) +
        '</div>';
    }
    else if (st === 4) {
      // Masking
      detail =
        '<div class="formula-box">Mask: 将不应被关注的位置设为 −∞ （softmax 后变成 0）</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '<b style="color:#EF4444">Causal Mask（因果掩码）</b>：在 Decoder 中，每个 token 只能看到<b>自己和之前</b>的 token —— 防止模型"偷看"未来。<br>' +
          '• <b>Encoder</b> 的 Self-Attention <b>不加 mask</b>（token 可以看到整个序列）<br>' +
          '• <b>Decoder</b> 的 Self-Attention <b>加 causal mask</b>（上三角置 −∞）<br>' +
          '• <b>Padding Mask</b>: 无论 Encoder/Decoder，把 [PAD] 位置置 −∞（不计算注意力）<br><br>' +
          '📌 本步骤<b>仅用于教学</b> —— 本模块是 Encoder-style，实际计算时不加 mask。下方演示 mask 启用后的效果：' +
        '</div>' +
        '<div style="text-align:center;margin-top:10px">' +
          '<button class="ctrl-btn ' + (this.showMask ? 'active' : '') + '" onclick="window._attnInstance.toggleMask()">' +
            (this.showMask ? '✓ 已启用 Causal Mask' : '🔒 点击启用 Causal Mask') +
          '</button>' +
        '</div>';
      const M = this.showMask ? this.MaskedScore : this.ScaledScore;
      const A = this.showMask ? this.AttnMasked : this.Attn;
      mainViz =
        '<div style="display:flex;align-items:center;justify-content:center;gap:18px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(M, { name: this.showMask ? 'Masked Score' : 'ScaledScore', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 58, decimals: 2, scale: 2, maskUpper: this.showMask }) +
          '<div style="color:#FBBF24;font-family:JetBrains Mono,monospace">→ softmax →</div>' +
          this._matrixHTML(A, { name: 'Attention', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 58, decimals: 3, intensityMode: true }) +
        '</div>' +
        (this.showMask ? '<div style="text-align:center;color:#EF4444;font-size:0.85rem;margin-top:8px">🚫 红色 −∞ 区域 softmax 后变为 0：token "' + tok[0] + '" 无法看到 "' + tok[1] + '"/"' + tok[2] + '"</div>' : '<div style="text-align:center;color:#666;font-size:0.78rem;margin-top:8px">未启用 mask 时，所有 token 之间都可以自由交互（Encoder 模式）</div>');
    }
    else if (st === 5) {
      // Softmax
      const r = this.softmaxRow;
      const rowLbl = tok[r];
      detail =
        '<div class="formula-box">Attention[i][j] = exp(ScaledScore[i][j]) / Σₖ exp(ScaledScore[i][k])</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">' +
          'softmax 按<b>每一行</b>独立归一化。下面以 <b style="color:' + this.tokenColors[r] + '">Q[' + rowLbl + ']</b> 行为例逐步展开：' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:flex-start;justify-content:center;gap:14px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(this.ScaledScore, { name: 'ScaledScore', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 48, decimals: 2, scale: 2, highlightRow: r }) +
          '<div style="color:#FBBF24;align-self:center">→ softmax(按行) →</div>' +
          this._matrixHTML(this.Attn, { name: 'Attention', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 48, decimals: 3, intensityMode: true, highlightRow: r,
            clickable: { kind: 'attn' }, selectedCell: [r, (this.selectedCell.kind === 'attn' ? this.selectedCell.j : 0)] }) +
        '</div>' +
        '<div style="margin-top:12px;padding:12px;background:#0a0a0a;border:1px solid #222;border-radius:8px">' +
          '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px;align-items:center">' +
            '<span style="color:#aaa;font-size:0.82rem">选择行进行展开:</span>' +
            tok.map((t, i) => '<button onclick="window._attnInstance.setSoftmaxRow(' + i + ')" style="background:' + (r === i ? this.tokenColors[i] : '#1a1a1a') + ';color:' + (r === i ? '#fff' : '#aaa') + ';border:1px solid ' + this.tokenColors[i] + ';padding:4px 10px;border-radius:4px;cursor:pointer;font-family:JetBrains Mono,monospace;font-size:0.75rem">Q[' + t + ']</button>').join('') +
          '</div>' +
          this._softmaxBreakdown(this.ScaledScore[r], rowLbl) +
        '</div>' +
        '<div style="margin-top:10px;display:flex;justify-content:center">' +
          '<div style="padding:10px 14px;background:#0a0a0a;border:1px solid #222;border-radius:8px">' +
            '<div style="color:#aaa;font-size:0.8rem;margin-bottom:6px">Q[<b style="color:' + this.tokenColors[r] + '">' + rowLbl + '</b>] 的注意力分布 (和为 1):</div>' +
            this._barChart(this.Attn[r], tok, this.tokenColors, null) +
          '</div>' +
        '</div>';
    }
    else if (st === 6) {
      // Output = Attention · V, 交互式
      const sel = this.selectedCell;
      const oi = (sel && (sel.kind === 'output' || sel.kind === 'attn')) ? sel.i : 0;
      const a_i = this.Attn[oi];
      const terms = a_i.map((a, j) => a.toFixed(3) + '·V[' + tok[j] + ']');
      detail =
        '<div class="formula-box">Output = Attention · V &nbsp;&nbsp; 形状: (3×3) · (3×4) = (3×4)</div>' +
        '<div class="formula-box" style="font-size:0.78rem">Output[<span style="color:' + this.tokenColors[oi] + '">' + tok[oi] + '</span>] = ' + terms.join(' + ') + '</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '每一个 output 行是 V 的<b>加权平均</b>，权重就是当前 token 的注意力分布。<br>' +
          '• 权重高的 V 行贡献大，低的几乎不贡献。<br>' +
          '• 这一步是 attention 的<b>核心</b>: "按相关度聚合上下文"。<br>' +
          '🖱️ 点击下方 Attention 或 Output 矩阵的任意行，查看对应行的加权求和。' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(this.Attn, { name: 'Attention', color: '#F97316', rowLabels: tok, colLabels: tok, cellSize: 44, decimals: 3, intensityMode: true, highlightRow: oi,
            clickable: { kind: 'attn' }, selectedCell: [oi, 0] }) +
          '<div style="color:#888;font-size:1.4rem">·</div>' +
          this._matrixHTML(this.V, { name: 'V', color: '#10B981', rowLabels: tok, cellSize: 44, decimals: 2 }) +
          '<div style="color:#888;font-size:1.4rem">=</div>' +
          this._matrixHTML(this.Output, { name: 'Output', color: '#06B6D4', rowLabels: tok, cellSize: 46, decimals: 2, highlightRow: oi,
            clickable: { kind: 'output' }, selectedCell: [oi, 0] }) +
        '</div>' +
        // 加权分解图
        '<div style="margin-top:12px;padding:12px;background:#0a0a0a;border:1px solid #222;border-radius:8px">' +
          '<div style="color:#aaa;font-size:0.82rem;margin-bottom:8px">Output[<b style="color:' + this.tokenColors[oi] + '">' + tok[oi] + '</b>] 的加权求和分解:</div>' +
          '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">' +
            a_i.map((a, j) =>
              '<div style="display:flex;flex-direction:column;align-items:center;border:1px solid ' + this.tokenColors[j] + ';border-radius:8px;padding:6px 8px;opacity:' + (0.4 + a * 0.6).toFixed(2) + '">' +
                '<div style="color:#FBBF24;font-family:JetBrains Mono,monospace;font-size:0.78rem;font-weight:600">' + a.toFixed(3) + '</div>' +
                '<div style="font-size:1.3rem;color:#666">×</div>' +
                this._matrixHTML([this.V[j]], { name: 'V[' + tok[j] + ']', color: this.tokenColors[j], cellSize: 34, decimals: 2 }) +
              '</div>'
            ).join('<div style="color:#F97316;font-size:1.6rem">+</div>') +
            '<div style="color:#666;font-size:1.6rem">=</div>' +
            '<div style="border:2px solid #06B6D4;border-radius:8px;padding:6px 10px;background:rgba(6,182,212,0.06)">' +
              this._matrixHTML([this.Output[oi]], { name: 'Output[' + tok[oi] + ']', color: '#06B6D4', cellSize: 40, decimals: 2 }) +
            '</div>' +
          '</div>' +
        '</div>';
    }
    else if (st === 7) {
      // 全貌：流向图 + 完整数据流
      detail =
        '<div class="formula-box" style="text-align:center;font-size:0.92rem">' +
          '<b>Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V</b>' +
        '</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:8px;text-align:center">' +
          '🎨 下方是 <b>token 注意力流向图</b> —— Self-Attention 最经典的可视化。线越粗 = 注意力权重越大。' +
        '</div>';
      mainViz =
        '<div style="padding:10px">' +
          this._flowDiagram(this.Attn, this.softmaxRow) +
        '</div>' +
        '<div style="margin-top:12px;padding:10px;background:#0a0a0a;border:1px solid #222;border-radius:8px;overflow-x:auto">' +
          '<div style="font-weight:600;margin-bottom:8px;font-size:0.88rem">📊 完整数据流</div>' +
          '<div style="display:flex;gap:10px;align-items:center;min-width:max-content">' +
            this._mini(this.X, 'X', '#e5e5e5', '(3×4)') +
            '<span style="color:#666">→</span>' +
            this._mini(this.Q, 'Q', '#3B82F6', '(3×4)') +
            this._mini(this.K, 'K', '#8B5CF6', '(3×4)') +
            this._mini(this.V, 'V', '#10B981', '(3×4)') +
            '<span style="color:#666">→</span>' +
            this._mini(this.Score, 'Score', '#F97316', '(3×3)') +
            '<span style="color:#666">÷√d_k,softmax</span>' +
            this._mini(this.Attn, 'Attn', '#F97316', '(3×3)', true) +
            '<span style="color:#666">·V →</span>' +
            this._mini(this.Output, 'Output', '#06B6D4', '(3×4)') +
          '</div>' +
        '</div>';
    }

    // ---------- 侧边栏：状态面板（始终显示当前各矩阵） ----------
    const overview = this._overviewPanel(st);

    // ---------- 组装 HTML ----------
    this.container.innerHTML =
      '<div class="attn-viz">' +
        '<style>' +
          '.attn-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.attn-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;font-size:0.85rem;transition:all .15s}' +
          '.attn-viz .ctrl-btn:hover{background:#252525;border-color:#555}' +
          '.attn-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.attn-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0;border:1px solid #1f2937;overflow-x:auto}' +
          '.attn-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.attn-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '.attn-viz .main-grid{display:grid;grid-template-columns:1fr 260px;gap:16px}' +
          '@media(max-width:1150px){.attn-viz .main-grid{grid-template-columns:1fr}}' +
          '@media(max-width:900px){.attn-viz .two-col{grid-template-columns:1fr}}' +
          '.attn-viz .matrix-wrap{display:inline-block;vertical-align:top}' +
          '.attn-viz .kbd{background:#0a0a0a;border:1px solid #333;border-radius:3px;padding:1px 6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#aaa}' +
        '</style>' +

        // 标题
        '<div class="section" style="text-align:center">' +
          '<div style="font-size:1.18rem;font-weight:700;margin-bottom:8px">Self-Attention 可视化</div>' +
          '<div class="formula-box" style="display:inline-block;color:#FBBF24;font-size:0.95rem">' +
            'Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V' +
          '</div>' +
          '<div style="color:#888;font-size:0.78rem;margin-top:6px">' +
            'd_model = 4 · seq_len = 3 · d_k = d_v = 4 &nbsp;|&nbsp; tokens: ' +
            tok.map((t, i) => '<span style="color:' + this.tokenColors[i] + ';font-weight:600">"' + t + '"</span>').join(' · ') +
          '</div>' +
        '</div>' +

        // 控制栏
        '<div class="section">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">' +
            '<div>' +
              playBtn +
              '<button class="ctrl-btn" onclick="window._attnInstance.stepBack()">⏮ 上一步</button>' +
              '<button class="ctrl-btn" onclick="window._attnInstance.stepForward()">⏭ 下一步</button>' +
              '<button class="ctrl-btn" onclick="window._attnInstance.reset()">↻ 重置</button>' +
              '<label style="color:#aaa;font-size:0.82rem;margin-left:8px">速度 ' +
                '<select onchange="window._attnInstance.setSpeed(this.value)" style="background:#0f0f0f;color:#e5e5e5;border:1px solid #333;padding:4px 6px;border-radius:4px">' +
                  '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option>' +
                '</select>' +
              '</label>' +
            '</div>' +
            '<div style="color:#aaa;font-size:0.78rem">' +
              '<span class="kbd">Space</span> 播放 · <span class="kbd">←</span> <span class="kbd">→</span> 步进 · <span class="kbd">R</span> 重置' +
            '</div>' +
          '</div>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepDots + '</div>' +
        '</div>' +

        // 主内容 + 侧边
        '<div class="main-grid">' +
          '<div>' +
            '<div class="section">' +
              '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">' +
                '<div style="background:#3B82F6;color:#fff;padding:3px 10px;border-radius:4px;font-size:0.78rem;font-family:JetBrains Mono,monospace">Step ' + (st + 1) + '/' + this.STEPS.length + '</div>' +
                '<div style="font-weight:600;font-size:1rem">' + this.STEPS[st].long + '</div>' +
              '</div>' +
              detail +
            '</div>' +
            '<div class="section" style="overflow-x:auto">' +
              '<div style="font-weight:600;margin-bottom:10px">📐 当前计算</div>' +
              mainViz +
            '</div>' +
          '</div>' +

          // 侧边状态面板
          '<div class="section" style="align-self:start;max-height:860px;overflow-y:auto">' +
            overview +
          '</div>' +
        '</div>' +

        // Token 注意力流向图（步骤 ≥6 始终展示）
        (st >= 5 ?
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">🌊 Token 注意力流向图 (Attention Pattern)</div>' +
            this._flowDiagram(this.Attn, this.softmaxRow) +
          '</div>' : '') +

        // 教育面板
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">📖 Q/K/V 图书馆类比</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.75">' +
              '你到图书馆查资料:<br>' +
              '• <b style="color:#3B82F6">Query</b>: "我想找关于 Transformer 的内容" (你的问题)<br>' +
              '• <b style="color:#8B5CF6">Key</b>: 书脊上的标签、索引卡 (供你查找)<br>' +
              '• <b style="color:#10B981">Value</b>: 书的正文内容 (真正的信息)<br><br>' +
              '<b style="color:#FBBF24">检索过程:</b><br>' +
              '1️⃣ 用 Query 与每本 Key 做相似度（点积）→ Score<br>' +
              '2️⃣ softmax 归一化 → 注意力权重（概率）<br>' +
              '3️⃣ 按权重混合 Value → 得到"你想要的综合信息"' +
            '</div>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">🎯 关键细节 & 常见误区</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.75">' +
              '• <b style="color:#FBBF24">"Self"</b>: Q、K、V 都来自<b>同一个 X</b>（区别 Cross-Attention）<br>' +
              '• <b>/ √d_k</b>: 避免高维下点积爆炸 → softmax 尖锐 → 梯度消失<br>' +
              '• <b>Softmax 按行</b>: 每个 token 的注意力分布独立归一化（行和=1）<br>' +
              '• <b style="color:#EF4444">Causal Mask</b>: 仅 Decoder 中使用，上三角→−∞<br>' +
              '• <b>复杂度</b>: O(n²·d) —— 长序列成本高（FlashAttention / 线性注意力是优化方向）<br>' +
              '• <b>可学习参数</b>: 只有 W_Q, W_K, W_V（MHA 多一个 W_O）' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 形状速查 + 代码对照
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">📐 形状速查表</div>' +
            '<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.78rem">' +
              '<thead><tr style="border-bottom:1px solid #333;color:#888">' +
                '<th style="text-align:left;padding:4px 6px">变量</th><th style="text-align:left;padding:4px 6px">维度</th><th style="text-align:left;padding:4px 6px;font-family:Inter,sans-serif">含义</th>' +
              '</tr></thead>' +
              '<tbody style="color:#ccc">' +
                '<tr><td style="padding:3px 6px">X</td><td style="padding:3px 6px">(3, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">输入嵌入</td></tr>' +
                '<tr><td style="padding:3px 6px;color:#3B82F6">W_Q, W_K, W_V</td><td style="padding:3px 6px">(4, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">可学习投影矩阵</td></tr>' +
                '<tr><td style="padding:3px 6px;color:#3B82F6">Q, K, V</td><td style="padding:3px 6px">(3, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">token 的 query/key/value</td></tr>' +
                '<tr><td style="padding:3px 6px;color:#F97316">Score</td><td style="padding:3px 6px">(3, 3)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">两两相关度原始分</td></tr>' +
                '<tr><td style="padding:3px 6px;color:#F97316">Attention</td><td style="padding:3px 6px">(3, 3)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">归一化后的权重（行和=1）</td></tr>' +
                '<tr><td style="padding:3px 6px;color:#06B6D4">Output</td><td style="padding:3px 6px">(3, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">融合上下文的新表示</td></tr>' +
              '</tbody>' +
            '</table>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">💻 PyTorch 代码对照</div>' +
            '<pre style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:12px;font-family:JetBrains Mono,monospace;font-size:0.75rem;line-height:1.55;color:#e5e5e5;margin:0;overflow-x:auto"><span style="color:#60a5fa">import</span> torch, math\n' +
              '<span style="color:#60a5fa">def</span> <span style="color:#FBBF24">self_attention</span>(X, W_Q, W_K, W_V):\n' +
              '    Q = X @ W_Q             <span style="color:#666"># (n, d_k)</span>\n' +
              '    K = X @ W_K\n' +
              '    V = X @ W_V\n' +
              '    d_k = Q.size(-1)\n' +
              '    scores = Q @ K.transpose(-2, -1) / <span style="color:#a5f3fc">math.sqrt</span>(d_k)\n' +
              '    <span style="color:#666"># mask = torch.triu(torch.ones_like(scores), 1).bool()</span>\n' +
              '    <span style="color:#666"># scores = scores.masked_fill(mask, -1e9)</span>\n' +
              '    attn = <span style="color:#a5f3fc">torch.softmax</span>(scores, dim=-1)\n' +
              '    <span style="color:#60a5fa">return</span> attn @ V       <span style="color:#666"># (n, d_v)</span></pre>' +
          '</div>' +
        '</div>' +

      '</div>';

    window._attnInstance = this;
  }

  // ---------- 侧边小矩阵 ----------
  _mini(M, name, color, dim, intensity) {
    return '<div style="display:flex;flex-direction:column;align-items:center">' +
      '<div style="color:' + color + ';font-size:0.7rem;font-weight:600;margin-bottom:3px">' + name + ' <span style="color:#666">' + dim + '</span></div>' +
      '<div>' + M.map(row =>
        '<div style="display:flex">' + row.map(v =>
          '<div style="width:28px;height:18px;margin:1px;background:' +
            (intensity ? this._intensity(v) : this._bgColor(v, 2)) +
            ';border:1px solid ' + color + ';border-radius:2px;color:#e5e5e5;font-family:JetBrains Mono,monospace;font-size:0.58rem;display:flex;align-items:center;justify-content:center">' +
            this._fmt(v, 1) +
          '</div>'
        ).join('') + '</div>'
      ).join('') + '</div>' +
    '</div>';
  }

  // ---------- 侧边概览 ----------
  _overviewPanel(st) {
    const rows = [];
    rows.push(this._miniSide(this.X, 'X', '#e5e5e5', 1));
    if (st >= 1) rows.push(this._miniSide(this.Q, 'Q', '#3B82F6', 2));
    if (st >= 1) rows.push(this._miniSide(this.K, 'K', '#8B5CF6', 2));
    if (st >= 1) rows.push(this._miniSide(this.V, 'V', '#10B981', 2));
    if (st >= 2) rows.push(this._miniSide(this.Score, 'Score', '#F97316', 2));
    if (st >= 3) rows.push(this._miniSide(this.ScaledScore, 'Scaled', '#F97316', 2));
    if (st >= 5) rows.push(this._miniSide(this.Attn, 'Attention', '#F97316', 3, true));
    if (st >= 6) rows.push(this._miniSide(this.Output, 'Output', '#06B6D4', 2));

    return '<div style="font-weight:600;margin-bottom:10px;font-size:0.9rem">📊 当前状态</div>' + rows.join('');
  }

  _miniSide(M, name, color, decimals, intensity) {
    return '<div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #222">' +
      '<div style="color:' + color + ';font-size:0.76rem;font-weight:600;margin-bottom:3px">' +
        name + '  <span style="color:#666;font-weight:400">(' + M.length + '×' + M[0].length + ')</span>' +
      '</div>' +
      '<div>' + M.map(row =>
        '<div style="display:flex">' + row.map(v =>
          '<div style="width:32px;height:22px;margin:1px;background:' +
            (intensity ? this._intensity(v) : this._bgColor(v, 2)) +
            ';border:1px solid ' + color + ';border-radius:2px;color:#e5e5e5;font-family:JetBrains Mono,monospace;font-size:0.62rem;display:flex;align-items:center;justify-content:center">' +
            this._fmt(v, decimals) +
          '</div>'
        ).join('') + '</div>'
      ).join('') + '</div>' +
    '</div>';
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._attnInstance === this) {
      try { delete window._attnInstance; } catch (e) { window._attnInstance = null; }
    }
    this.container = null;
  }
}

window.AttentionVisualization = AttentionVisualization;
