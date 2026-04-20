// Multi-Head Attention 可视化
// MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
// 固定参数: d_model = 4, num_heads = 2, d_k = d_v = 2
class MultiHeadAttention {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // 交互态
    this.selectedQueryRow = -1;  // -1 = 所有 query 都显示，否则只显示该行
    this.selectedHead = 0;       // 0 = 两个都看, 1 = Head 1 only, 2 = Head 2 only

    this.d_model = 4;
    this.num_heads = 2;
    this.d_k = 2;
    this.d_v = 2;
    this.seq_len = 3;

    this.tokens = ['I', 'love', 'NLP'];
    this.tokenColors = ['#60A5FA', '#F59E0B', '#F472B6'];

    this.X = [
      [1.0, 0.5, 0.3, 0.2],
      [0.8, 1.0, 0.5, 0.1],
      [0.3, 0.7, 1.0, 0.4]
    ];

    // Head 1 权重 (4×2)
    this.WQ1 = [[ 1.0,  0.0], [ 0.0,  1.0], [ 0.5, -0.3], [-0.2,  0.4]];
    this.WK1 = [[ 1.0,  0.2], [ 0.1,  1.0], [-0.2,  0.3], [ 0.3, -0.1]];
    this.WV1 = [[ 1.0, -0.1], [ 0.2,  1.0], [ 0.0,  0.3], [-0.3,  0.2]];

    // Head 2 权重 (4×2) —— 故意设计成关注不同模式
    this.WQ2 = [[ 0.3,  1.0], [ 1.0,  0.0], [-0.2,  0.5], [ 0.4,  0.1]];
    this.WK2 = [[-0.1,  1.0], [ 0.5,  0.0], [ 1.0,  0.2], [ 0.0,  0.3]];
    this.WV2 = [[ 0.2,  0.5], [-0.3,  1.0], [ 1.0,  0.0], [ 0.1,  0.4]];

    // 输出投影 W_O (4×4)
    this.WO = [
      [ 1.0,  0.2, -0.1,  0.0],
      [ 0.1,  1.0,  0.3, -0.2],
      [-0.2,  0.0,  1.0,  0.3],
      [ 0.3, -0.1,  0.2,  1.0]
    ];

    this.STEPS = [
      { short: '① 输入',           long: '输入 X (3×4)' },
      { short: '② 双路投影',       long: '并行投影: 两个 head 同时生成各自的 Q/K/V' },
      { short: '③ 双路 Attention', long: '两个 head 并行计算 self-attention' },
      { short: '④ Concat',          long: 'Concat 沿特征维拼接: (3×2) ⊕ (3×2) = (3×4)' },
      { short: '⑤ W_O 混合',        long: 'Output = Concat · W_O —— 混合 head 间信息' },
      { short: '⑥ 流向对比',        long: '并排对比: 两个 head 学到的不同关系模式' }
    ];

    this.C = {
      H1: '#3B82F6',
      H2: '#EC4899',
      H1rgb: [59, 130, 246],
      H2rgb: [236, 72, 153],
      Q:  '#60A5FA',
      K:  '#A78BFA',
      V:  '#34D399',
      Score: '#F97316',
      Out: '#06B6D4',
      Concat: '#FBBF24',
      Final: '#10B981'
    };
  }

  // ---------- 矩阵运算（共享：见 js/utils/math.js） ----------
  // matmul / transpose / scale / softmax 都来自 Utils.*

  _computeHead(WQ, WK, WV) {
    const U = window.Utils;
    const Q = U.matmul(this.X, WQ);
    const K = U.matmul(this.X, WK);
    const V = U.matmul(this.X, WV);
    const KT = U.transpose(K);
    const Score = U.matmul(Q, KT);
    const Scaled = U.scale(Score, Math.sqrt(this.d_k));
    const Attn = U.softmax(Scaled);
    const Out = U.matmul(Attn, V);
    return { Q, K, V, KT, Score, Scaled, Attn, Out };
  }

  _compute() {
    this.sqrt_dk = Math.sqrt(this.d_k);
    this.head1 = this._computeHead(this.WQ1, this.WK1, this.WV1);
    this.head2 = this._computeHead(this.WQ2, this.WK2, this.WV2);
    this.Concat = this.head1.Out.map((row, i) => row.concat(this.head2.Out[i]));
    this.Final = window.Utils.matmul(this.Concat, this.WO);
  }

  // ---------- 生命周期 ----------
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this._compute();
    this.render();
    return this;
  }

  reset() {
    this.currentStep = 0;
    this.isPlaying = false;
    clearTimeout(this.timer);
    this.selectedQueryRow = -1;
    this.selectedHead = 0;
    this.render();
  }
  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  setSpeed(s) { this.speed = Number(s) || 1; }
  stepForward() { if (this.currentStep < this.STEPS.length - 1) { this.currentStep++; this.render(); } }
  stepBack() { if (this.currentStep > 0) { this.currentStep--; this.render(); } }
  goTo(i) { this.currentStep = Math.max(0, Math.min(i, this.STEPS.length - 1)); this.render(); }
  setQueryRow(i) { this.selectedQueryRow = i; this.render(); }
  setHeadFilter(h) { this.selectedHead = h; this.render(); }

  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this.render();
      this.timer = setTimeout(() => this._auto(), 1900 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  // ---------- 渲染辅助 ----------
  // _fmt 已迁移到 Utils.fmt（见 js/utils/format.js）
  _fmt(v, d) { return window.Utils.fmt(v, d); }

  _bgColor(v, scale) {
    scale = scale || 1.5;
    if (Math.abs(v) < 1e-9) return '#0d0d0d';
    const a = Math.min(0.15 + Math.abs(v) / scale * 0.45, 0.6);
    return v > 0 ? 'rgba(16,185,129,' + a.toFixed(3) + ')' : 'rgba(239,68,68,' + a.toFixed(3) + ')';
  }

  _intensity(v, hue) {
    hue = hue || [249, 115, 22];
    const a = Math.min(0.08 + v * 0.85, 0.92);
    return 'rgba(' + hue[0] + ',' + hue[1] + ',' + hue[2] + ',' + a.toFixed(3) + ')';
  }

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
      intensityHue = [249, 115, 22],
      visible = true,
      scale = 1.5,
      cellSize = 42,
      decimals = 2,
      dim = null,
      splitCol = -1
    } = opts;

    const rows = M.length, cols = M[0].length;
    const dimStr = dim || (rows + '×' + cols);

    let head = '';
    if (colLabels) {
      head += '<div style="display:flex;margin-left:' + (rowLabels ? 50 : 0) + 'px">';
      for (let j = 0; j < cols; j++) {
        head += '<div style="width:' + cellSize + 'px;text-align:center;font-size:0.66rem;color:#888;font-family:JetBrains Mono,monospace">' + colLabels[j] + '</div>';
      }
      head += '</div>';
    }

    let body = '';
    for (let i = 0; i < rows; i++) {
      body += '<div style="display:flex;align-items:center">';
      if (rowLabels) {
        const rlColor = typeof rowLabels[i] === 'string' && this.tokens.includes(rowLabels[i])
          ? this.tokenColors[this.tokens.indexOf(rowLabels[i])] : color;
        body += '<div style="width:50px;font-size:0.72rem;color:' + rlColor + ';font-family:JetBrains Mono,monospace;text-align:right;padding-right:6px;font-weight:600">' + rowLabels[i] + '</div>';
      }
      for (let j = 0; j < cols; j++) {
        const v = M[i][j];
        const hlRow = (highlightRow === i);
        const hlCol = (highlightCol === j);
        const hlCell = highlightCell && highlightCell[0] === i && highlightCell[1] === j;

        let bg;
        if (!visible) bg = '#0a0a0a';
        else if (intensityMode) bg = this._intensity(v, intensityHue);
        else bg = this._bgColor(v, scale);

        let border = '1px solid ' + color;
        let boxShadow = '';
        if (hlCell) { border = '2px solid #FBBF24'; boxShadow = '0 0 12px rgba(251,191,36,0.75)'; }
        else if (hlRow || hlCol) { border = '2px solid #FBBF24'; }

        const marginRight = (splitCol >= 0 && j === splitCol) ? '12px' : '1px';
        const txt = visible ? this._fmt(v, decimals) : '?';

        body += '<div style="width:' + cellSize + 'px;height:' + cellSize + 'px;margin:1px ' + marginRight + ' 1px 1px;background:' + bg +
          ';border:' + border + ';border-radius:3px;color:#e5e5e5' +
          ';font-family:JetBrains Mono,monospace;font-size:0.72rem;display:flex;align-items:center;justify-content:center;' +
          (boxShadow ? 'box-shadow:' + boxShadow + ';' : '') + '">' + txt + '</div>';
      }
      body += '</div>';
    }

    const title = name
      ? '<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:3px"><span style="color:' + color + ';font-weight:600;font-size:0.84rem">' + name + '</span><span style="color:#666;font-size:0.7rem;font-family:JetBrains Mono,monospace">(' + dimStr + ')</span></div>'
      : '';

    return '<div class="matrix-wrap">' + title + head + body + '</div>';
  }

  // ---------- 注意力流向图（SVG） ----------
  _flowDiagram(attnMatrix, color, hue, label, selectedQuery) {
    const tok = this.tokens;
    const N = tok.length;
    const w = 480, h = 190;
    const pad = 36;
    const innerW = w - 2 * pad;
    const xq = i => pad + (i + 0.5) * innerW / N;
    const xk = j => pad + (j + 0.5) * innerW / N;
    const yq = 28;
    const yk = h - 28;

    const nodeFor = (t, x, y, lbl, c) =>
      '<g>' +
        '<rect x="' + (x - 34) + '" y="' + (y - 13) + '" width="68" height="26" rx="5" ry="5" fill="#0d0d0d" stroke="' + c + '" stroke-width="1.3"/>' +
        '<text x="' + x + '" y="' + (y + 5) + '" text-anchor="middle" fill="' + c + '" font-family="Inter,sans-serif" font-size="11" font-weight="600">' + t + '</text>' +
        '<text x="' + x + '" y="' + (y + (lbl === 'Q' ? -20 : 28)) + '" text-anchor="middle" fill="#777" font-family="JetBrains Mono,monospace" font-size="9">' + lbl + '</text>' +
      '</g>';

    let nodes = '';
    for (let i = 0; i < N; i++) {
      nodes += nodeFor(tok[i], xq(i), yq, 'Q', this.tokenColors[i]);
      nodes += nodeFor(tok[i], xk(i), yk, 'K/V', this.tokenColors[i]);
    }

    let edges = '';
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const w_ij = attnMatrix[i][j];
        if (w_ij < 0.02) continue;
        const x1 = xq(i), y1 = yq + 13, x2 = xk(j), y2 = yk - 13;
        const cx1 = x1, cy1 = y1 + (y2 - y1) * 0.5;
        const cx2 = x2, cy2 = y2 - (y2 - y1) * 0.5;
        const strokeW = Math.max(0.8, w_ij * 7);
        const isSelected = (selectedQuery === i || selectedQuery === -1);
        const opacity = isSelected ? Math.min(0.15 + w_ij * 0.85, 0.95) : 0.08;
        const strokeColor = isSelected ? color : '#555';
        edges += '<path d="M' + x1 + ',' + y1 + ' C' + cx1 + ',' + cy1 + ' ' + cx2 + ',' + cy2 + ' ' + x2 + ',' + y2 + '" fill="none" stroke="' + strokeColor + '" stroke-width="' + strokeW.toFixed(2) + '" opacity="' + opacity.toFixed(2) + '"/>';
        if (selectedQuery === i) {
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + (j - i) * 3;
          edges += '<text x="' + mx + '" y="' + my + '" text-anchor="middle" fill="#FBBF24" font-family="JetBrains Mono,monospace" font-size="10" font-weight="700" style="paint-order:stroke;stroke:#0f0f0f;stroke-width:3">' + w_ij.toFixed(2) + '</text>';
        }
      }
    }

    return '<div>' +
      '<div style="text-align:center;color:' + color + ';font-weight:600;margin-bottom:4px;font-size:0.88rem">' + label + '</div>' +
      '<svg viewBox="0 0 ' + w + ' ' + h + '" style="width:100%;max-width:' + w + 'px;height:auto;background:rgba(' + hue.join(',') + ',0.03);border:1px solid rgba(' + hue.join(',') + ',0.3);border-radius:8px;display:block">' +
        edges + nodes +
      '</svg>' +
    '</div>';
  }

  // ---------- 并排显示两个 head 的某个阶段 ----------
  _sideBySideHeadProjections() {
    const tok = this.tokens;
    const C = this.C;

    const headBox = (label, color, hue, WQ, WK, WV, head) => {
      return '<div style="border:2px solid ' + color + ';border-radius:10px;padding:12px;background:rgba(' + hue.join(',') + ',0.05);flex:1;min-width:420px">' +
        '<div style="color:' + color + ';font-weight:700;margin-bottom:10px;font-size:0.95rem">' + label + '</div>' +

        '<div style="font-size:0.74rem;color:#888;margin-bottom:3px">① 投影 Q/K/V（每个 W 都是 4×2）</div>' +
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">' +
          '<div>' +
            this._matrixHTML(WQ, { name: 'W_Q', color: C.Q, cellSize: 28, decimals: 2 }) +
            '<div style="margin-top:4px">' + this._matrixHTML(head.Q, { name: 'Q', color: C.Q, rowLabels: tok, cellSize: 30, decimals: 2 }) + '</div>' +
          '</div>' +
          '<div>' +
            this._matrixHTML(WK, { name: 'W_K', color: C.K, cellSize: 28, decimals: 2 }) +
            '<div style="margin-top:4px">' + this._matrixHTML(head.K, { name: 'K', color: C.K, rowLabels: tok, cellSize: 30, decimals: 2 }) + '</div>' +
          '</div>' +
          '<div>' +
            this._matrixHTML(WV, { name: 'W_V', color: C.V, cellSize: 28, decimals: 2 }) +
            '<div style="margin-top:4px">' + this._matrixHTML(head.V, { name: 'V', color: C.V, rowLabels: tok, cellSize: 30, decimals: 2 }) + '</div>' +
          '</div>' +
        '</div>' +

        '<div style="color:#888;font-size:0.72rem">子空间维度: <b style="color:' + color + '">d_k = d_v = ' + this.d_k + '</b>（d_model / num_heads）</div>' +
      '</div>';
    };

    return '<div style="display:flex;gap:14px;flex-wrap:wrap">' +
      headBox('🔷 Head 1', C.H1, C.H1rgb, this.WQ1, this.WK1, this.WV1, this.head1) +
      headBox('🟣 Head 2', C.H2, C.H2rgb, this.WQ2, this.WK2, this.WV2, this.head2) +
    '</div>';
  }

  _sideBySideHeadAttention() {
    const tok = this.tokens;
    const C = this.C;

    const box = (label, color, hue, head) => {
      return '<div style="border:2px solid ' + color + ';border-radius:10px;padding:12px;background:rgba(' + hue.join(',') + ',0.05);flex:1;min-width:380px">' +
        '<div style="color:' + color + ';font-weight:700;margin-bottom:10px;font-size:0.95rem">' + label + '</div>' +

        // Q·K^T → softmax → Attn
        '<div style="font-size:0.74rem;color:#888;margin-bottom:3px">① 相关度 → softmax</div>' +
        '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:10px">' +
          this._matrixHTML(head.Q, { name: 'Q', color: C.Q, rowLabels: tok, cellSize: 30, decimals: 2 }) +
          '<span style="color:#888">·</span>' +
          this._matrixHTML(head.KT, { name: 'Kᵀ', color: C.K, colLabels: tok, cellSize: 30, decimals: 2, dim: '2×3' }) +
          '<span style="color:#FBBF24;font-family:JetBrains Mono,monospace;font-size:0.72rem">÷√' + this.d_k + '→softmax</span>' +
          this._matrixHTML(head.Attn, { name: 'Attn', color: C.Score, rowLabels: tok, colLabels: tok, cellSize: 36, decimals: 3, intensityMode: true, intensityHue: hue }) +
        '</div>' +

        // Attn·V → head_out
        '<div style="font-size:0.74rem;color:#888;margin-bottom:3px">② 加权求和 V</div>' +
        '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">' +
          this._matrixHTML(head.Attn, { name: 'Attn', color: C.Score, rowLabels: tok, cellSize: 30, decimals: 3, intensityMode: true, intensityHue: hue }) +
          '<span style="color:#888">·</span>' +
          this._matrixHTML(head.V, { name: 'V', color: C.V, rowLabels: tok, cellSize: 30, decimals: 2 }) +
          '<span style="color:#888">=</span>' +
          this._matrixHTML(head.Out, { name: 'head', color: color, rowLabels: tok, cellSize: 36, decimals: 2 }) +
        '</div>' +
      '</div>';
    };

    return '<div style="display:flex;gap:14px;flex-wrap:wrap">' +
      box('🔷 Head 1 的 Attention', C.H1, C.H1rgb, this.head1) +
      box('🟣 Head 2 的 Attention', C.H2, C.H2rgb, this.head2) +
    '</div>';
  }

  // ---------- "head 特化" 分析：找出每个 head 最强的关注模式 ----------
  _headSpecialization() {
    const tok = this.tokens;
    const findTop = (head) => {
      let best = { i: 0, j: 0, v: 0 };
      for (let i = 0; i < head.Attn.length; i++) {
        for (let j = 0; j < head.Attn[0].length; j++) {
          if (head.Attn[i][j] > best.v) best = { i, j, v: head.Attn[i][j] };
        }
      }
      return best;
    };
    const top1 = findTop(this.head1);
    const top2 = findTop(this.head2);
    const describe = (t, color) =>
      '<div style="padding:10px 12px;border:1px solid ' + color + ';border-radius:8px;background:rgba(0,0,0,0.4)">' +
        '<div style="color:' + color + ';font-weight:600;font-size:0.85rem;margin-bottom:4px">Head 最关注的关系</div>' +
        '<div style="color:#ccc;font-size:0.9rem">' +
          '<b style="color:' + this.tokenColors[t.i] + '">' + tok[t.i] + '</b> → <b style="color:' + this.tokenColors[t.j] + '">' + tok[t.j] + '</b>' +
          ' <span style="color:#FBBF24;font-family:JetBrains Mono,monospace">(权重 ' + t.v.toFixed(3) + ')</span>' +
        '</div>' +
      '</div>';
    return '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
      describe(top1, this.C.H1) +
      describe(top2, this.C.H2) +
    '</div>';
  }

  // ---------- 参数对比面板 ----------
  _paramComparison() {
    const d = this.d_model, h = this.num_heads, d_k = this.d_k;
    const singleHead = 3 * d * d;                    // W_Q, W_K, W_V 各 d×d
    const multiHeadQKV = h * 3 * d * d_k;             // h 个 head，各 W_Q/W_K/W_V 为 d×d_k
    const multiHeadWO = d * d;                       // W_O
    const multiHeadTotal = multiHeadQKV + multiHeadWO;
    return '<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.8rem">' +
      '<thead><tr style="border-bottom:1px solid #333;color:#888">' +
        '<th style="text-align:left;padding:4px 6px">模型</th><th style="text-align:right;padding:4px 6px">参数量</th><th style="text-align:left;padding:4px 6px;font-family:Inter,sans-serif">细节</th>' +
      '</tr></thead>' +
      '<tbody style="color:#ccc">' +
        '<tr>' +
          '<td style="padding:4px 6px">单头 Attention</td>' +
          '<td style="padding:4px 6px;text-align:right;color:#aaa">' + singleHead + '</td>' +
          '<td style="padding:4px 6px;font-family:Inter,sans-serif;color:#aaa">3 × (d × d) = 3 × ' + d + ' × ' + d + '</td>' +
        '</tr>' +
        '<tr style="border-top:1px dashed #222">' +
          '<td style="padding:4px 6px;color:' + this.C.Final + '">MHA QKV</td>' +
          '<td style="padding:4px 6px;text-align:right;color:#10B981">' + multiHeadQKV + '</td>' +
          '<td style="padding:4px 6px;font-family:Inter,sans-serif;color:#aaa">' + h + ' head × 3 × (d × d_k) = ' + h + ' × 3 × ' + d + ' × ' + d_k + '</td>' +
        '</tr>' +
        '<tr>' +
          '<td style="padding:4px 6px;color:' + this.C.Final + '">MHA W_O</td>' +
          '<td style="padding:4px 6px;text-align:right;color:#10B981">' + multiHeadWO + '</td>' +
          '<td style="padding:4px 6px;font-family:Inter,sans-serif;color:#aaa">d × d = ' + d + ' × ' + d + '</td>' +
        '</tr>' +
        '<tr style="border-top:1px solid #333;background:rgba(16,185,129,0.05)">' +
          '<td style="padding:4px 6px;color:#10B981;font-weight:600">MHA 合计</td>' +
          '<td style="padding:4px 6px;text-align:right;color:#10B981;font-weight:600">' + multiHeadTotal + '</td>' +
          '<td style="padding:4px 6px;font-family:Inter,sans-serif;color:#aaa">与单头参数量同数量级（多了 W_O 的 ' + multiHeadWO + '）</td>' +
        '</tr>' +
      '</tbody></table>' +
      '<div style="color:#888;font-size:0.78rem;margin-top:6px">💡 将 d_model 切成 h 个 d_k 维子空间并行计算，<b>参数量与单头接近，表达力却更丰富</b>。</div>';
  }

  // ---------- 主渲染 ----------
  render() {
    if (!this.container) return;
    const st = this.currentStep;
    const tok = this.tokens;
    const C = this.C;

    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._mhaInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._mhaInstance.play()">▶ 播放</button>';

    const stepDots = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff;border-color:#3B82F6'
                    : (i < st ? 'background:#1e3a5f;color:#93c5fd;border-color:#2563eb'
                              : 'background:#0f0f0f;color:#666;border-color:#333');
      return '<div onclick="window._mhaInstance.goTo(' + i + ')" ' +
        'title="' + s.long + '" ' +
        'style="padding:6px 10px;border:1px solid;border-radius:20px;font-size:0.72rem;cursor:pointer;white-space:nowrap;transition:all .15s;' + active + '">' +
        s.short + '</div>';
    }).join('');

    let detail = '';
    let mainViz = '';

    if (st === 0) {
      detail =
        '<div class="formula-box">X ∈ ℝ<sup>3×4</sup>  —— d_model = 4, seq_len = 3</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:8px">' +
          '本模块用 <b>2-head attention</b>: 每个 head 在 <b>d_k = d_v = 2</b> 的子空间独立做注意力，再拼接 + 投影。<br>' +
          '通常 <b>num_heads × d_k = d_model</b>（这里 2×2 = 4），保证整体参数量与单头相近。<br>' +
          '后续每一步都会<b>并排</b>展示两个 head 的计算，让你直观看到它们学到的<b>不同模式</b>。' +
        '</div>';
      mainViz =
        '<div style="display:flex;justify-content:center;padding:10px">' +
          this._matrixHTML(this.X, { name: 'X', color: '#e5e5e5', rowLabels: tok, colLabels: ['d₀','d₁','d₂','d₃'], cellSize: 54, decimals: 2 }) +
        '</div>' +
        '<div style="text-align:center;color:#888;font-size:0.8rem;margin-top:8px">下一步会把 X 同时投影到 <b style="color:' + C.H1 + '">Head 1</b> 和 <b style="color:' + C.H2 + '">Head 2</b> 各自的子空间。</div>';
    }
    else if (st === 1) {
      // 两个 head 的 Q/K/V 并排
      detail =
        '<div class="formula-box">两个 head <b>并行</b>投影: Q_i = X · W_Q^i, &nbsp; K_i = X · W_K^i, &nbsp; V_i = X · W_V^i &nbsp; (i = 1, 2)</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '每个 head 有<b>独立</b>的 W_Q / W_K / W_V（形状 <b>d_model × d_k = 4×2</b>）。<br>' +
          '两个 head 从 X 出发投影到<b>不同的 2 维子空间</b>，学到 X 的不同侧面。<br>' +
          '• <b style="color:' + C.H1 + '">Head 1</b> 的 W 初始化偏向"按词嵌入前两维对齐"（可能捕捉某种语法关系）<br>' +
          '• <b style="color:' + C.H2 + '">Head 2</b> 的 W 初始化偏向"按后两维对齐"（可能捕捉另一种语义关系）<br>' +
          '训练时这些权重会<b>自动分化</b>，学到互补的关系模式。' +
        '</div>';
      mainViz = this._sideBySideHeadProjections();
    }
    else if (st === 2) {
      // 两个 head 的 Attention 计算并排
      const h1 = this.head1, h2 = this.head2;
      detail =
        '<div class="formula-box">head_i = softmax( Q_i · K_iᵀ / √d_k ) · V_i &nbsp; (两 head 完全独立运行 self-attention)</div>' +
        '<div class="formula-box" style="font-size:0.78rem">Head 1: 最大 Score' + '₁ = ' + Math.max(...h1.Score.flat()).toFixed(2) +
        ' &nbsp;|&nbsp; Head 2: 最大 Score₂ = ' + Math.max(...h2.Score.flat()).toFixed(2) + '</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '两个 head 各自走一遍完整的 self-attention：<br>' +
          '1) Q·Kᵀ 得到 3×3 的相关度矩阵 →<br>' +
          '2) 除以 √d_k，softmax 归一化 →<br>' +
          '3) 与 V 相乘得到 (3×2) 的 head_i 输出。<br>' +
          '<b style="color:#FBBF24">关键观察</b>: 同样的输入 X，两个 head 输出的 <b>Attn 矩阵完全不同</b>（下一步会看到流向图对比）。' +
        '</div>';
      mainViz = this._sideBySideHeadAttention();
    }
    else if (st === 3) {
      // Concat
      detail =
        '<div class="formula-box">Concat(head₁, head₂) = [ head₁ | head₂ ] &nbsp; 沿<b>特征维</b>拼接</div>' +
        '<div class="formula-box" style="font-size:0.78rem">(3×2) ⊕ (3×2) = (3×4) &nbsp; —— 行数不变，列数相加</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '拼接把两个 head 的输出"并排放好"，<b>不做任何混合</b>。<br>' +
          '• 前 2 列 = Head 1 的输出（<b style="color:' + C.H1 + '">蓝</b>）<br>' +
          '• 后 2 列 = Head 2 的输出（<b style="color:' + C.H2 + '">粉</b>）<br>' +
          '此时 head 之间<b>没有任何信息交流</b>，就是简单拼接。下一步的 W_O 才会真正混合。' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(this.head1.Out, { name: 'head₁', color: C.H1, rowLabels: tok, cellSize: 52, decimals: 2 }) +
          '<div style="color:#FBBF24;font-size:1.5rem;font-family:JetBrains Mono,monospace">⊕</div>' +
          this._matrixHTML(this.head2.Out, { name: 'head₂', color: C.H2, rowLabels: tok, cellSize: 52, decimals: 2 }) +
          '<div style="color:#FBBF24;font-size:1.5rem">=</div>' +
          this._matrixHTML(this.Concat, { name: 'Concat', color: C.Concat, rowLabels: tok, colLabels: ['h₁,0','h₁,1','h₂,0','h₂,1'], cellSize: 52, decimals: 2, splitCol: 1 }) +
        '</div>' +
        '<div style="text-align:center;color:#888;font-size:0.78rem;margin-top:10px">Concat 矩阵中间的<b style="color:#FBBF24">黄色间隔</b>标出了两个 head 的分界线。</div>';
    }
    else if (st === 4) {
      // W_O 混合
      detail =
        '<div class="formula-box">Output = Concat · W_O &nbsp; (3×4) · (4×4) = (3×4)</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '<b style="color:#FBBF24">W_O 的核心作用</b>: 让两个 head 的子空间输出<b>互相混合</b>，产生协同。<br>' +
          '• Concat 本身没有跨 head 的信息流<br>' +
          '• W_O 的每一行都是<b>全部 4 维特征</b>（2 个 Head 的所有通道）的线性组合<br>' +
          '• 训练时 W_O 会学会哪个 head 的信息更重要、怎样融合<br>' +
          '• 输出形状回到 (3, 4) = (seq_len, d_model)，便于接入下一层（Add & Norm, FFN）' +
        '</div>';
      mainViz =
        '<div style="display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;padding:8px">' +
          this._matrixHTML(this.Concat, { name: 'Concat', color: C.Concat, rowLabels: tok, colLabels: ['h₁,0','h₁,1','h₂,0','h₂,1'], cellSize: 46, decimals: 2, splitCol: 1 }) +
          '<div style="color:#888;font-size:1.5rem">·</div>' +
          this._matrixHTML(this.WO, { name: 'W_O', color: '#e5e5e5', cellSize: 42, decimals: 2 }) +
          '<div style="color:#888;font-size:1.5rem">=</div>' +
          this._matrixHTML(this.Final, { name: 'Output', color: C.Final, rowLabels: tok, colLabels: ['d₀','d₁','d₂','d₃'], cellSize: 48, decimals: 2 }) +
        '</div>' +
        '<div style="text-align:center;margin-top:12px">' +
          '<div style="display:inline-block;padding:10px 16px;background:rgba(16,185,129,0.08);border:1px solid #10B981;border-radius:8px;color:#10B981;font-weight:600">' +
            '✅ Multi-Head Attention 完成：(3, 4) → (3, 4)，形状与输入一致' +
          '</div>' +
        '</div>';
    }
    else if (st === 5) {
      // 流向对比
      const sq = this.selectedQueryRow;
      detail =
        '<div class="formula-box" style="text-align:center">下方并排展示两个 head 学到的<b>不同注意力模式</b> —— 多头的核心价值</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.75;margin-top:6px">' +
          '观察：同一组输入 X，两个 head 的流向图<b>完全不同</b>。<br>' +
          '• 某些 token 在 Head 1 中相互强关联，在 Head 2 中却几乎无联系<br>' +
          '• 这说明它们关注的是<b>不同类型</b>的关系<br>' +
          '• 训练过程中，网络自动让不同 head 分化以捕捉更全面的结构<br>' +
          '🖱️ 下方按钮切换查看不同 Query 视角的注意力流。' +
        '</div>';

      const rowBtns =
        '<button onclick="window._mhaInstance.setQueryRow(-1)" style="background:' + (sq === -1 ? '#F97316' : '#1a1a1a') + ';color:' + (sq === -1 ? '#fff' : '#aaa') + ';border:1px solid #F97316;padding:4px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.75rem;cursor:pointer;margin-right:6px">全部 Query</button>' +
        tok.map((t, i) =>
          '<button onclick="window._mhaInstance.setQueryRow(' + i + ')" style="background:' + (sq === i ? this.tokenColors[i] : '#1a1a1a') + ';color:' + (sq === i ? '#fff' : '#aaa') + ';border:1px solid ' + this.tokenColors[i] + ';padding:4px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.75rem;cursor:pointer;margin-right:4px">Q[' + t + ']</button>'
        ).join('');

      mainViz =
        '<div style="text-align:center;margin-bottom:10px">' + rowBtns + '</div>' +
        '<div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center">' +
          this._flowDiagram(this.head1.Attn, C.H1, C.H1rgb, '🔷 Head 1 的注意力模式', sq) +
          this._flowDiagram(this.head2.Attn, C.H2, C.H2rgb, '🟣 Head 2 的注意力模式', sq) +
        '</div>' +
        this._headSpecialization();
    }

    // ---------- 右侧状态面板 ----------
    const state = this._statePanel(st);

    // ---------- 组装 HTML ----------
    this.container.innerHTML =
      '<div class="mha-viz">' +
        '<style>' +
          '.mha-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.mha-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;font-size:0.85rem;transition:all .15s}' +
          '.mha-viz .ctrl-btn:hover{background:#252525;border-color:#555}' +
          '.mha-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.mha-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0;border:1px solid #1f2937;overflow-x:auto}' +
          '.mha-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.mha-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '.mha-viz .main-grid{display:grid;grid-template-columns:1fr 240px;gap:14px}' +
          '@media(max-width:1150px){.mha-viz .main-grid{grid-template-columns:1fr}}' +
          '@media(max-width:900px){.mha-viz .two-col{grid-template-columns:1fr}}' +
          '.mha-viz .matrix-wrap{display:inline-block;vertical-align:top}' +
          '.mha-viz .kbd{background:#0a0a0a;border:1px solid #333;border-radius:3px;padding:1px 6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#aaa}' +
          '.mha-viz .head-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:600;font-family:JetBrains Mono,monospace}' +
        '</style>' +

        // 标题
        '<div class="section" style="text-align:center">' +
          '<div style="font-size:1.18rem;font-weight:700;margin-bottom:8px">Multi-Head Attention 可视化</div>' +
          '<div class="formula-box" style="display:inline-block;color:#FBBF24;font-size:0.95rem">' +
            'MultiHead(Q, K, V) = Concat(head₁, head₂) · W_O &nbsp;|&nbsp; head_i = Attention(X·W_Q<sup>i</sup>, X·W_K<sup>i</sup>, X·W_V<sup>i</sup>)' +
          '</div>' +
          '<div style="margin-top:8px">' +
            '<span class="head-badge" style="background:rgba(59,130,246,0.2);color:' + C.H1 + ';border:1px solid ' + C.H1 + '">🔷 Head 1</span>&nbsp;' +
            '<span class="head-badge" style="background:rgba(236,72,153,0.2);color:' + C.H2 + ';border:1px solid ' + C.H2 + '">🟣 Head 2</span>' +
            '<span style="color:#888;font-size:0.78rem;margin-left:10px">d_model = 4 · num_heads = 2 · d_k = d_v = 2 · tokens: ' +
              tok.map((t, i) => '<span style="color:' + this.tokenColors[i] + ';font-weight:600">"' + t + '"</span>').join(' · ') +
            '</span>' +
          '</div>' +
        '</div>' +

        // 控制栏
        '<div class="section">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">' +
            '<div>' +
              playBtn +
              '<button class="ctrl-btn" onclick="window._mhaInstance.stepBack()">⏮ 上一步</button>' +
              '<button class="ctrl-btn" onclick="window._mhaInstance.stepForward()">⏭ 下一步</button>' +
              '<button class="ctrl-btn" onclick="window._mhaInstance.reset()">↻ 重置</button>' +
              '<label style="color:#aaa;font-size:0.82rem;margin-left:8px">速度 ' +
                '<select onchange="window._mhaInstance.setSpeed(this.value)" style="background:#0f0f0f;color:#e5e5e5;border:1px solid #333;padding:4px 6px;border-radius:4px">' +
                  '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option>' +
                '</select>' +
              '</label>' +
            '</div>' +
            '<div style="color:#aaa;font-size:0.78rem"><span class="kbd">Space</span> 播放 · <span class="kbd">←</span> <span class="kbd">→</span> 步进 · <span class="kbd">R</span> 重置</div>' +
          '</div>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepDots + '</div>' +
        '</div>' +

        // 主内容
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
              '<div style="font-weight:600;margin-bottom:10px">🔀 计算过程</div>' +
              mainViz +
            '</div>' +
          '</div>' +
          '<div class="section" style="align-self:start;max-height:860px;overflow-y:auto">' +
            state +
          '</div>' +
        '</div>' +

        // Head 1 vs Head 2 注意力矩阵对比（始终显示）
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">🔀 Head 1 vs Head 2 注意力矩阵对比</div>' +
          this._renderAttnComparison() +
        '</div>' +

        // 教育面板
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">🎓 为什么要多头？</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.75">' +
              '单头 attention 只能学一种相关性模式。多头能<b>同时</b>学多种:<br>' +
              '• <b style="color:' + C.H1 + '">有的 head</b> 关注<b>语法依赖</b>（主谓宾、修饰关系）<br>' +
              '• <b style="color:' + C.H2 + '">有的 head</b> 关注<b>共指 / 语义相近</b>（"他"↔"张三"）<br>' +
              '• 还有的关注<b>位置关系</b>（上一个/下一个词）<br>' +
              '• 训练后每个 head 会自动"找到自己擅长"的关系类型<br><br>' +
              '<b style="color:#FBBF24">关键直觉</b>: 同样的参数预算，分给多个子空间<b>独立学习</b>，比集中在一个大空间里学到的关系<b>更多样、更结构化</b>。' +
            '</div>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">📐 参数量对比</div>' +
            this._paramComparison() +
          '</div>' +
        '</div>' +

        // 统一实现 + 形状表
        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">⚡ 实现技巧: 一次大 matmul 代替 h 次</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-bottom:8px">' +
              '本模块为教学展示把 h 个 head 分开画。实际 PyTorch 代码中通常合并成一次矩阵乘：' +
            '</div>' +
            '<pre style="background:#0a0a0a;border:1px solid #222;border-radius:6px;padding:10px;font-family:JetBrains Mono,monospace;font-size:0.74rem;line-height:1.55;color:#e5e5e5;margin:0;overflow-x:auto"><span style="color:#60a5fa">def</span> <span style="color:#FBBF24">mha</span>(X, W_Q, W_K, W_V, W_O, h):\n' +
              '    n, d = X.shape\n' +
              '    d_k = d // h\n' +
              '    Q = X @ W_Q     <span style="color:#666"># (n, d) —— 包含所有 head</span>\n' +
              '    K = X @ W_K\n' +
              '    V = X @ W_V\n' +
              '    <span style="color:#666"># reshape 成 (h, n, d_k)</span>\n' +
              '    Q = Q.view(n, h, d_k).transpose(0, 1)\n' +
              '    K = K.view(n, h, d_k).transpose(0, 1)\n' +
              '    V = V.view(n, h, d_k).transpose(0, 1)\n' +
              '    attn = <span style="color:#a5f3fc">softmax</span>(Q @ K.transpose(-2,-1) / <span style="color:#a5f3fc">sqrt</span>(d_k), -1)\n' +
              '    out = (attn @ V).transpose(0, 1).reshape(n, d)\n' +
              '    <span style="color:#60a5fa">return</span> out @ W_O</pre>' +
            '<div style="color:#888;font-size:0.75rem;margin-top:6px">💡 单个大 W_Q (d×d) 在 GPU 上比 h 个小 W_Q (d×d_k) <b>快得多</b>。</div>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:10px">📐 形状速查</div>' +
            '<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.78rem">' +
              '<thead><tr style="border-bottom:1px solid #333;color:#888">' +
                '<th style="text-align:left;padding:3px 6px">变量</th><th style="text-align:left;padding:3px 6px">维度</th><th style="text-align:left;padding:3px 6px;font-family:Inter,sans-serif">含义</th>' +
              '</tr></thead>' +
              '<tbody style="color:#ccc">' +
                '<tr><td style="padding:3px 6px">X</td><td style="padding:3px 6px">(3, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">输入嵌入</td></tr>' +
                '<tr><td style="padding:3px 6px;color:' + C.H1 + '">W_Q¹, W_K¹, W_V¹</td><td style="padding:3px 6px">(4, 2)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">Head 1 投影权重</td></tr>' +
                '<tr><td style="padding:3px 6px;color:' + C.H2 + '">W_Q², W_K², W_V²</td><td style="padding:3px 6px">(4, 2)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">Head 2 投影权重</td></tr>' +
                '<tr><td style="padding:3px 6px">Q_i, K_i, V_i</td><td style="padding:3px 6px">(3, 2)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">每个 head 的子空间 Q/K/V</td></tr>' +
                '<tr><td style="padding:3px 6px;color:#F97316">Attn_i</td><td style="padding:3px 6px">(3, 3)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">每 head 的注意力权重</td></tr>' +
                '<tr><td style="padding:3px 6px">head_i</td><td style="padding:3px 6px">(3, 2)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">每 head 输出</td></tr>' +
                '<tr><td style="padding:3px 6px;color:' + C.Concat + '">Concat</td><td style="padding:3px 6px">(3, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">拼接后张量</td></tr>' +
                '<tr><td style="padding:3px 6px">W_O</td><td style="padding:3px 6px">(4, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">输出投影</td></tr>' +
                '<tr><td style="padding:3px 6px;color:' + C.Final + '">Output</td><td style="padding:3px 6px">(3, 4)</td><td style="padding:3px 6px;font-family:Inter,sans-serif;color:#aaa">与输入同形</td></tr>' +
              '</tbody>' +
            '</table>' +
          '</div>' +
        '</div>' +
      '</div>';

    window._mhaInstance = this;
  }

  // ---------- 始终显示的 Head 1 / Head 2 注意力对比 ----------
  _renderAttnComparison() {
    const tok = this.tokens;
    const C = this.C;

    const cellGrid = (M, color, hue) =>
      '<div>' +
        '<div style="display:flex;gap:4px;align-items:center;margin-bottom:3px">' +
          '<div style="width:50px"></div>' +
          tok.map(t => '<div style="width:50px;text-align:center;font-size:0.7rem;color:#888;font-family:JetBrains Mono,monospace">→ ' + t + '</div>').join('') +
        '</div>' +
        tok.map((t, i) =>
          '<div style="display:flex;gap:4px;align-items:center;margin-bottom:1px">' +
            '<div style="width:50px;text-align:right;font-size:0.72rem;color:' + this.tokenColors[i] + ';font-family:JetBrains Mono,monospace;font-weight:600">' + t + '</div>' +
            M[i].map(v =>
              '<div style="width:50px;height:28px;background:' + this._intensity(v, hue) + ';border:1px solid ' + color + ';border-radius:3px;color:#e5e5e5;font-family:JetBrains Mono,monospace;font-size:0.7rem;display:flex;align-items:center;justify-content:center">' + v.toFixed(3) + '</div>'
            ).join('') +
          '</div>'
        ).join('') +
      '</div>';

    const panel = (label, color, hue, head) =>
      '<div style="flex:1;min-width:300px;border:2px solid ' + color + ';border-radius:8px;padding:12px;background:rgba(' + hue.join(',') + ',0.05)">' +
        '<div style="color:' + color + ';font-weight:700;margin-bottom:8px">' + label + '</div>' +
        cellGrid(head.Attn, color, hue) +
      '</div>';

    return '<div style="display:flex;gap:14px;flex-wrap:wrap">' +
      panel('🔷 Head 1 Attention (行=Query, 列=Key)', C.H1, C.H1rgb, this.head1) +
      panel('🟣 Head 2 Attention (行=Query, 列=Key)', C.H2, C.H2rgb, this.head2) +
    '</div>' +
    '<div style="color:#888;font-size:0.78rem;margin-top:8px">👀 两个 head 的注意力分布<b>明显不同</b> —— 这就是多头的价值：<b>同一组输入学出多种关系</b>。</div>';
  }

  // ---------- 侧边状态面板 ----------
  _statePanel(st) {
    const rows = [];
    rows.push(this._miniSide(this.X, 'X', '#e5e5e5', 1));
    if (st >= 1) {
      rows.push(this._miniSide(this.head1.Q, 'Q₁', this.C.H1, 2));
      rows.push(this._miniSide(this.head1.K, 'K₁', this.C.H1, 2));
      rows.push(this._miniSide(this.head1.V, 'V₁', this.C.H1, 2));
      rows.push(this._miniSide(this.head2.Q, 'Q₂', this.C.H2, 2));
      rows.push(this._miniSide(this.head2.K, 'K₂', this.C.H2, 2));
      rows.push(this._miniSide(this.head2.V, 'V₂', this.C.H2, 2));
    }
    if (st >= 2) {
      rows.push(this._miniSide(this.head1.Attn, 'Attn₁', '#F97316', 3, [59, 130, 246]));
      rows.push(this._miniSide(this.head2.Attn, 'Attn₂', '#F97316', 3, [236, 72, 153]));
      rows.push(this._miniSide(this.head1.Out, 'head₁', this.C.H1, 2));
      rows.push(this._miniSide(this.head2.Out, 'head₂', this.C.H2, 2));
    }
    if (st >= 3) rows.push(this._miniSide(this.Concat, 'Concat', this.C.Concat, 2));
    if (st >= 4) rows.push(this._miniSide(this.Final, 'Output', this.C.Final, 2));
    return '<div style="font-weight:600;margin-bottom:10px;font-size:0.9rem">📊 当前状态</div>' + rows.join('');
  }

  _miniSide(M, name, color, decimals, intensityHue) {
    return '<div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #222">' +
      '<div style="color:' + color + ';font-size:0.72rem;font-weight:600;margin-bottom:3px">' +
        name + '  <span style="color:#666;font-weight:400">(' + M.length + '×' + M[0].length + ')</span>' +
      '</div>' +
      '<div>' + M.map(row =>
        '<div style="display:flex">' + row.map(v =>
          '<div style="width:28px;height:18px;margin:1px;background:' +
            (intensityHue ? this._intensity(v, intensityHue) : this._bgColor(v, 2)) +
            ';border:1px solid ' + color + ';border-radius:2px;color:#e5e5e5;font-family:JetBrains Mono,monospace;font-size:0.58rem;display:flex;align-items:center;justify-content:center">' +
            this._fmt(v, decimals) +
          '</div>'
        ).join('') + '</div>'
      ).join('') + '</div>' +
    '</div>';
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._mhaInstance === this) {
      try { delete window._mhaInstance; } catch (e) { window._mhaInstance = null; }
    }
    this.container = null;
  }
}

window.MultiHeadAttention = MultiHeadAttention;
