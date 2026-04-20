// Transformer Decoder Block 可视化
// 完整数据流：
//   Input(+PE) → Masked Self-Attention → Add&LN1
//              → Cross-Attention(Q=dec, K,V=enc) → Add&LN2
//              → FFN → Add&LN3 → Linear+Softmax(vocab)
class TransformerDecoder {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // 目标序列: "我 爱 猫" (shifted right 后 decoder 第一个输入是 <BOS>)
    // 这里直接展示 3 个 target token 的前向（训练时并行）
    this.tokens = ['我', '爱', '猫'];
    this.seq_len = 3;
    this.d_model = 4;

    // Decoder 输入 (token embedding + PE)，硬编码使可复现
    this.X = [
      [1.0, 0.5, 0.3, 0.2],   // 我
      [0.8, 1.0, 0.5, 0.1],   // 爱
      [0.3, 0.7, 1.0, 0.4]    // 猫
    ];
    // 位置编码（3 × 4）
    this.PE = [
      [0.000, 1.000, 0.000, 1.000],   // pos=0
      [0.841, 0.540, 0.010, 1.000],   // pos=1
      [0.909, -0.416, 0.020, 1.000]   // pos=2
    ];
    // Decoder 输入 = X + PE
    this.DecInput = this.X.map((r, i) => r.map((v, j) => +(v + this.PE[i][j]).toFixed(3)));

    // Encoder 输出 (由 encoder 计算产生，这里硬编码作为已知量) 3 × 4
    this.EncOut = [
      [0.90, 0.40, 0.20, 0.10],
      [0.60, 0.95, 0.45, 0.30],
      [0.20, 0.55, 0.85, 0.50]
    ];

    // 为简化，Q/K/V 投影矩阵设为单位矩阵（让 Q=K=V=Input 以突出机制）
    this.WQ = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    this.WK = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    this.WV = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

    // FFN 权重 (4 → 8 → 4)
    this.W1 = [
      [0.30,-0.20, 0.50, 0.10,-0.30, 0.40, 0.00, 0.20],
      [0.10, 0.40,-0.20, 0.30, 0.50,-0.10, 0.60,-0.20],
      [-0.40, 0.20, 0.30,-0.10, 0.20, 0.50,-0.30, 0.10],
      [0.20, 0.10,-0.20, 0.40,-0.10, 0.30, 0.20,-0.30]
    ];
    this.b1 = new Array(8).fill(0);
    this.W2 = [
      [0.20, 0.10,-0.20, 0.30],
      [-0.30, 0.40, 0.10,-0.20],
      [0.10,-0.20, 0.30, 0.00],
      [0.20, 0.10,-0.30, 0.20],
      [-0.10, 0.20, 0.10, 0.30],
      [0.30,-0.10, 0.20,-0.20],
      [0.00, 0.20,-0.30, 0.10],
      [0.20, 0.00, 0.30,-0.10]
    ];
    this.b2 = new Array(4).fill(0);

    // 词表（示例 5 个词）
    this.vocab = ['我', '爱', '猫', '狗', '你'];
    // Linear(d_model → vocab_size=5)
    this.W_out = [
      [0.40,-0.20, 0.10, 0.30, 0.00],
      [0.10, 0.50,-0.10, 0.20,-0.30],
      [-0.20, 0.10, 0.60, 0.00, 0.30],
      [0.30, 0.20, 0.10,-0.40, 0.50]
    ];

    this.STEPS = [
      'Decoder 输入：Embedding + PE',
      'Masked Self-Attn：计算 Q/K/V',
      'Masked Self-Attn：Score 矩阵 = QKᵀ/√d_k',
      'Masked Self-Attn：应用 Causal Mask',
      'Masked Self-Attn：Softmax → Weights',
      'Masked Self-Attn：Weights · V → Attn₁',
      'Add & LayerNorm 1',
      'Cross-Attn：Q(decoder) · K(encoder)ᵀ',
      'Cross-Attn：Softmax → Attn₂',
      'Add & LayerNorm 2',
      'FFN (4 → 8 → 4)',
      'Add & LayerNorm 3',
      'Linear + Softmax → 下一个 token 概率'
    ];

    this._computeAll();
  }

  // ============================================
  // 基础数学
  // ============================================
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this.render();
    return this;
  }

  // ---------- 计算工具（共享：见 js/utils/math.js） ----------
  // matmul / transpose / scale / add / layerNormMatrix / reluMatrix 都来自 Utils.*
  _matmul(A, B) { return window.Utils.matmul(A, B); }
  _transpose(A) { return window.Utils.transpose(A); }
  _add(A, B) { return window.Utils.matAdd(A, B); }
  _scale(A, s) { return window.Utils.scale(A, s); }
  _layerNorm(A) { return window.Utils.layerNormMatrix(A); }
  _relu(A) { return window.Utils.reluMatrix(A); }

  // 特殊：softmax 保留本模块定制版本，因为它会处理「整行都被 mask 成 -Infinity」的边角情形，
  // Utils.softmax 在这种极端输入下会返回 NaN。（正常 causal mask 不会触发，但稳妥起见保留。）
  _softmaxRow(row) {
    const m = Math.max(...row.filter(v => isFinite(v)));
    const ex = row.map(v => v === -Infinity ? 0 : Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => s === 0 ? 0 : v / s);
  }
  _softmax(M) { return M.map(r => this._softmaxRow(r)); }

  _computeAll() {
    const d_k = this.d_model;
    // --- Masked Self-Attention ---
    this.Q1 = this._matmul(this.DecInput, this.WQ);
    this.K1 = this._matmul(this.DecInput, this.WK);
    this.V1 = this._matmul(this.DecInput, this.WV);

    this.Score1_raw = this._scale(this._matmul(this.Q1, this._transpose(this.K1)), Math.sqrt(d_k));
    // Causal mask: 上三角 = -Inf
    this.Mask = Array.from({ length: this.seq_len }, (_, i) =>
      Array.from({ length: this.seq_len }, (_, j) => j > i ? -Infinity : 0)
    );
    this.Score1_masked = this.Score1_raw.map((r, i) => r.map((v, j) => v + this.Mask[i][j]));
    this.Attn1_weights = this._softmax(this.Score1_masked);
    this.Attn1_out = this._matmul(this.Attn1_weights, this.V1);

    // --- Add & LN 1 ---
    this.AddLN1 = this._layerNorm(this._add(this.DecInput, this.Attn1_out));

    // --- Cross-Attention (Q from decoder post-LN1, K/V from EncOut) ---
    this.Q2 = this._matmul(this.AddLN1, this.WQ);
    this.K2 = this._matmul(this.EncOut, this.WK);
    this.V2 = this._matmul(this.EncOut, this.WV);
    this.Score2 = this._scale(this._matmul(this.Q2, this._transpose(this.K2)), Math.sqrt(d_k));
    this.Attn2_weights = this._softmax(this.Score2);
    this.Attn2_out = this._matmul(this.Attn2_weights, this.V2);

    // --- Add & LN 2 ---
    this.AddLN2 = this._layerNorm(this._add(this.AddLN1, this.Attn2_out));

    // --- FFN ---
    this.FFN_h = this._relu(this.AddLN2.map(row =>
      this.W1[0].map((_, j) => row.reduce((s, v, k) => s + v * this.W1[k][j], 0) + this.b1[j])
    ));
    this.FFN_out = this.FFN_h.map(row =>
      this.W2[0].map((_, j) => row.reduce((s, v, k) => s + v * this.W2[k][j], 0) + this.b2[j])
    );

    // --- Add & LN 3 ---
    this.AddLN3 = this._layerNorm(this._add(this.AddLN2, this.FFN_out));

    // --- Linear + Softmax → vocab ---
    this.Logits = this._matmul(this.AddLN3, this.W_out);
    this.Probs = this._softmax(this.Logits);
  }

  // ============================================
  // 控制
  // ============================================
  reset() { this.currentStep = 0; this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); }
  setSpeed(s) { this.speed = Number(s) || 1; }
  stepForward() { if (this.currentStep < this.STEPS.length - 1) { this.currentStep++; this.render(); } }
  stepBack() { if (this.currentStep > 0) { this.currentStep--; this.render(); } }
  goTo(i) { this.currentStep = Math.max(0, Math.min(i, this.STEPS.length - 1)); this.render(); }

  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this.render();
      this.timer = setTimeout(() => this._auto(), 1500 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  // ============================================
  // 渲染辅助：矩阵 / 颜色
  // ============================================
  _cellColor(v, maxAbs = 1.0) {
    if (!isFinite(v)) return '#5c1a1a'; // -Inf
    const n = Math.max(-1, Math.min(1, v / maxAbs));
    if (n >= 0) {
      const a = 0.1 + n * 0.55;
      return `rgba(16,185,129,${a.toFixed(3)})`;
    } else {
      const a = 0.1 + Math.abs(n) * 0.55;
      return `rgba(239,68,68,${a.toFixed(3)})`;
    }
  }

  _matrixTable(M, color = '#3B82F6', opts = {}) {
    const {
      rowLabels = null,
      colLabels = null,
      title = '',
      cellW = 52,
      cellH = 28,
      highlightRows = [],
      highlightCols = [],
      isMask = false,
      probMode = false,
      maxAbs = 1.0
    } = opts;

    let html = '';
    if (title) html += '<div style="color:' + color + ';font-size:0.82rem;margin-bottom:4px;font-family:JetBrains Mono,monospace">' + title + '</div>';
    html += '<div style="overflow:auto"><table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.68rem">';
    if (colLabels) {
      html += '<tr><td></td>';
      colLabels.forEach(l => html += '<td style="padding:2px 6px;color:#888;text-align:center;font-size:0.68rem">' + l + '</td>');
      html += '</tr>';
    }
    for (let i = 0; i < M.length; i++) {
      html += '<tr>';
      if (rowLabels) {
        const lbl = rowLabels[i];
        html += '<td style="padding:2px 6px;color:#888;text-align:right;font-size:0.68rem">' + lbl + '</td>';
      }
      for (let j = 0; j < M[i].length; j++) {
        const v = M[i][j];
        let bg, txt;
        if (isMask) {
          if (v === -Infinity) { bg = '#5c1a1a'; txt = '#ff8a8a'; }
          else { bg = '#0e2a17'; txt = '#4ade80'; }
        } else if (probMode) {
          // 概率颜色：0 → 黑，1 → 亮橙
          const a = 0.08 + v * 0.7;
          bg = `rgba(249,115,22,${a.toFixed(3)})`;
          txt = v > 0.4 ? '#fff' : '#e5e5e5';
        } else {
          bg = this._cellColor(v, maxAbs);
          txt = Math.abs(v) > 0.55 * maxAbs ? '#fff' : '#e5e5e5';
        }
        const rowHL = highlightRows.includes(i);
        const colHL = highlightCols.includes(j);
        const bd = (rowHL || colHL) ? color : '#2a2a2a';
        const bw = (rowHL || colHL) ? 2 : 1;
        let content;
        if (v === -Infinity) content = '-∞';
        else if (v === Infinity) content = '+∞';
        else content = v.toFixed(probMode ? 3 : 2);
        html += '<td style="width:' + cellW + 'px;height:' + cellH + 'px;padding:0;border:' + bw + 'px solid ' + bd + ';background:' + bg + ';color:' + txt + ';text-align:center;vertical-align:middle">' + content + '</td>';
      }
      html += '</tr>';
    }
    html += '</table></div>';
    return html;
  }

  // 编码器-解码器整体示意图（SVG）
  _renderArchDiagram(activeIdx) {
    // activeIdx 映射到哪个子模块高亮
    const W = 520, H = 360;
    const col = (x, w, y, h, label, color, active) => {
      const fill = active ? color + '33' : '#1a1a1a';
      const stroke = active ? color : '#333';
      return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${stroke}" stroke-width="${active ? 2 : 1}" rx="6"/>` +
        `<text x="${x + w / 2}" y="${y + h / 2 + 4}" text-anchor="middle" fill="${active ? color : '#aaa'}" font-size="11" font-family="Inter,sans-serif">${label}</text>`;
    };

    // 解码器子模块高亮对应关系
    const hl = {
      encOut: activeIdx === 7 || activeIdx === 8,
      mhaM: activeIdx >= 1 && activeIdx <= 5,
      ln1: activeIdx === 6,
      mhaC: activeIdx === 7 || activeIdx === 8,
      ln2: activeIdx === 9,
      ffn: activeIdx === 10,
      ln3: activeIdx === 11,
      head: activeIdx === 12,
      input: activeIdx === 0
    };

    let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="background:#0a0a0a;border:1px solid #222;border-radius:6px">`;
    // Encoder (left)
    svg += `<text x="110" y="20" text-anchor="middle" fill="#8b949e" font-size="11" font-family="Inter">Encoder (上游)</text>`;
    svg += col(30, 160, 30, 40, 'Input Embedding + PE', '#3B82F6', false);
    svg += col(30, 160, 80, 40, 'N × Encoder Block', '#8B5CF6', false);
    svg += col(30, 160, 130, 40, 'Encoder Output  (3×4)', '#8B5CF6', hl.encOut);
    // 箭头
    svg += `<line x1="110" y1="70" x2="110" y2="80" stroke="#444"/>`;
    svg += `<line x1="110" y1="120" x2="110" y2="130" stroke="#444"/>`;

    // Decoder (right)
    svg += `<text x="370" y="20" text-anchor="middle" fill="#8b949e" font-size="11" font-family="Inter">Decoder (当前)</text>`;
    svg += col(290, 160, 30, 36, '① Input (X + PE)', '#3B82F6', hl.input);
    svg += col(290, 160, 76, 36, '② Masked Self-Attn', '#8B5CF6', hl.mhaM);
    svg += col(290, 160, 122, 26, 'Add & LayerNorm 1', '#10B981', hl.ln1);
    svg += col(290, 160, 158, 36, '③ Cross-Attention', '#06B6D4', hl.mhaC);
    svg += col(290, 160, 204, 26, 'Add & LayerNorm 2', '#10B981', hl.ln2);
    svg += col(290, 160, 240, 36, '④ FFN (4→8→4)', '#F59E0B', hl.ffn);
    svg += col(290, 160, 286, 26, 'Add & LayerNorm 3', '#10B981', hl.ln3);
    svg += col(290, 160, 322, 30, '⑤ Linear + Softmax', '#F97316', hl.head);

    // decoder 箭头
    for (let [y1, y2] of [[66, 76], [112, 122], [148, 158], [194, 204], [230, 240], [276, 286], [312, 322]]) {
      svg += `<line x1="370" y1="${y1}" x2="370" y2="${y2}" stroke="#444"/>`;
    }
    // 从 encoder 输出横跨到 cross-attention（虚线）
    const crossArrowColor = hl.mhaC ? '#06B6D4' : '#555';
    svg += `<path d="M 190 150 C 230 150, 250 176, 290 176" fill="none" stroke="${crossArrowColor}" stroke-width="${hl.mhaC ? 2 : 1}" stroke-dasharray="${hl.mhaC ? '0' : '4 3'}"/>`;
    svg += `<text x="240" y="142" text-anchor="middle" fill="${crossArrowColor}" font-size="10" font-family="JetBrains Mono,monospace">K, V</text>`;

    svg += '</svg>';
    return svg;
  }

  // ============================================
  // 每步主视化
  // ============================================
  _stepView(st) {
    const rowLabels = this.tokens;
    if (st === 0) {
      // Input = X + PE
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">解码器输入：Token Embedding + Positional Encoding</div>' +
        '<div class="formula-box">Input = Embedding("我","爱","猫") + PE[0:3]</div>' +
        '<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px">' +
          '<div>' + this._matrixTable(this.X, '#3B82F6', { title: 'X (3×4)', rowLabels, maxAbs: 1.2 }) + '</div>' +
          '<div style="font-size:1.5rem;color:#666">+</div>' +
          '<div>' + this._matrixTable(this.PE, '#8B5CF6', { title: 'PE (3×4)', rowLabels, maxAbs: 1.0 }) + '</div>' +
          '<div style="font-size:1.5rem;color:#666">=</div>' +
          '<div>' + this._matrixTable(this.DecInput, '#06B6D4', { title: 'Dec Input (3×4)', rowLabels, maxAbs: 2.0 }) + '</div>' +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">提示：训练时，目标序列要 shift right 一个位置（前面插 &lt;BOS&gt;），这里简化省略移位。</div>' +
      '</div>';
    }
    if (st === 1) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Masked Self-Attention：计算 Q / K / V</div>' +
        '<div class="formula-box">Q = Input · W_Q &nbsp;|&nbsp; K = Input · W_K &nbsp;|&nbsp; V = Input · W_V</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-bottom:8px">本可视化令 W_Q=W_K=W_V=I（单位矩阵），所以 Q=K=V=Input，突出注意力机制本身。</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap">' +
          this._matrixTable(this.Q1, '#3B82F6', { title: 'Q (3×4)', rowLabels, maxAbs: 2 }) +
          this._matrixTable(this.K1, '#8B5CF6', { title: 'K (3×4)', rowLabels, maxAbs: 2 }) +
          this._matrixTable(this.V1, '#10B981', { title: 'V (3×4)', rowLabels, maxAbs: 2 }) +
        '</div>' +
      '</div>';
    }
    if (st === 2) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Score 矩阵 = Q · Kᵀ / √d_k (未掩码)</div>' +
        '<div class="formula-box">Score[i,j] 表示 Query i 对 Key j 的相似度，√d_k=' + Math.sqrt(this.d_model).toFixed(3) + ' 是缩放因子</div>' +
        '<div style="margin-top:10px">' +
          this._matrixTable(this.Score1_raw, '#F97316', { title: 'Score (3×3)  行=Q, 列=K', rowLabels, colLabels: rowLabels, maxAbs: 2 }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">注意右上三角：token "我" 不应该看到 "爱" 或 "猫"，但现在 Score 里有正常数值 — 下一步用 Causal Mask 遮蔽它们。</div>' +
      '</div>';
    }
    if (st === 3) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">应用 Causal Mask（下三角）</div>' +
        '<div class="formula-box">Mask[i,j] = -∞  若 j &gt; i， 0 否则</div>' +
        '<div class="formula-box">MaskedScore = Score + Mask</div>' +
        '<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:10px">' +
          '<div>' + this._matrixTable(this.Mask, '#ef4444', { title: 'Causal Mask', rowLabels, colLabels: rowLabels, isMask: true }) + '</div>' +
          '<div style="font-size:1.5rem;color:#666">→</div>' +
          '<div>' + this._matrixTable(this.Score1_masked, '#F97316', { title: 'Masked Score (3×3)', rowLabels, colLabels: rowLabels, maxAbs: 2 }) + '</div>' +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px"><b>上三角被置 -∞</b>，这样下一步 softmax 时这些位置 e^(-∞)=0，权重刚好为 0，实现"看不到未来"。</div>' +
      '</div>';
    }
    if (st === 4) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Softmax（按行）→ 注意力权重</div>' +
        '<div class="formula-box">AttnWeights = softmax(MaskedScore)，每行和 = 1</div>' +
        '<div style="margin-top:10px">' +
          this._matrixTable(this.Attn1_weights, '#F97316', { title: 'Masked AttnWeights (3×3)', rowLabels, colLabels: rowLabels, probMode: true }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">观察：<br>' +
          '• 第 1 行 (Q=我)：只在列 0 有值（只能看自己），概率 = 1.0<br>' +
          '• 第 2 行 (Q=爱)：在列 0、1 分布（能看到 "我" 和自己）<br>' +
          '• 第 3 行 (Q=猫)：在列 0、1、2 分布（全都能看）<br>' +
          '这就是 causal mask 的效果：<b>行 i 只在前 i+1 个位置有非零权重</b>。</div>' +
      '</div>';
    }
    if (st === 5) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">加权求和：AttnWeights · V</div>' +
        '<div class="formula-box">Attn₁ = AttnWeights · V ∈ ℝ^(3×4)</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.Attn1_weights, '#F97316', { title: 'AttnWeights (3×3)', rowLabels, colLabels: rowLabels, probMode: true }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">·</div>' +
          this._matrixTable(this.V1, '#10B981', { title: 'V (3×4)', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">=</div>' +
          this._matrixTable(this.Attn1_out, '#06B6D4', { title: 'Attn₁ 输出 (3×4)', rowLabels, maxAbs: 2 }) +
        '</div>' +
      '</div>';
    }
    if (st === 6) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Add & LayerNorm 1</div>' +
        '<div class="formula-box">AddLN₁ = LayerNorm(Input + Attn₁)</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.DecInput, '#3B82F6', { title: 'Input (residual)', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">+</div>' +
          this._matrixTable(this.Attn1_out, '#06B6D4', { title: 'Attn₁', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">→ LN →</div>' +
          this._matrixTable(this.AddLN1, '#10B981', { title: 'AddLN₁', rowLabels, maxAbs: 2 }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">残差保证梯度直通；LN 在<b>每个 token 的特征维度</b>上归一化（不跨 token）。</div>' +
      '</div>';
    }
    if (st === 7) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Cross-Attention：Q(decoder) × K(encoder)ᵀ</div>' +
        '<div class="formula-box">Q_dec = AddLN₁ · W_Q&nbsp;&nbsp; K_enc = EncoderOut · W_K&nbsp;&nbsp; V_enc = EncoderOut · W_V</div>' +
        '<div class="formula-box">Score₂ = Q_dec · K_encᵀ / √d_k &nbsp;&nbsp;<b>注意：没有 causal mask！</b></div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.Q2, '#3B82F6', { title: 'Q_dec (3×4)', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">·</div>' +
          this._matrixTable(this._transpose(this.K2), '#8B5CF6', { title: 'K_encᵀ (4×3)', maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">/√d_k =</div>' +
          this._matrixTable(this.Score2, '#F97316', { title: 'Score₂ (3×3)  行=Q_dec, 列=Enc 位置', rowLabels, colLabels: ['e0','e1','e2'], maxAbs: 2 }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">Decoder 每个位置向 Encoder 的所有位置发出查询。这是"翻译对齐"的所在：target 词对应 source 词。</div>' +
      '</div>';
    }
    if (st === 8) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Cross-Attn Softmax → Attn₂</div>' +
        '<div class="formula-box">AttnWeights₂ = softmax(Score₂)</div>' +
        '<div class="formula-box">Attn₂ = AttnWeights₂ · V_enc ∈ ℝ^(3×4)</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.Attn2_weights, '#F97316', { title: 'AttnWeights₂ (3×3)', rowLabels, colLabels: ['e0','e1','e2'], probMode: true }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">·</div>' +
          this._matrixTable(this.V2, '#10B981', { title: 'V_enc (3×4)', maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">=</div>' +
          this._matrixTable(this.Attn2_out, '#06B6D4', { title: 'Attn₂ (3×4)', rowLabels, maxAbs: 2 }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">每行是"该 target 位置应该关注哪些 source 位置"。这解释了 Seq2Seq 翻译模型"看到对应源词"的行为。</div>' +
      '</div>';
    }
    if (st === 9) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Add & LayerNorm 2</div>' +
        '<div class="formula-box">AddLN₂ = LayerNorm(AddLN₁ + Attn₂)</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.AddLN1, '#10B981', { title: 'AddLN₁ (residual)', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">+</div>' +
          this._matrixTable(this.Attn2_out, '#06B6D4', { title: 'Attn₂', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">→ LN →</div>' +
          this._matrixTable(this.AddLN2, '#10B981', { title: 'AddLN₂', rowLabels, maxAbs: 2 }) +
        '</div>' +
      '</div>';
    }
    if (st === 10) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Position-wise FFN (4 → 8 → 4)</div>' +
        '<div class="formula-box">FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.AddLN2, '#10B981', { title: '输入 (3×4)', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.1rem;color:#666;align-self:center">·W₁→ReLU→·W₂</div>' +
          this._matrixTable(this.FFN_h, '#F59E0B', { title: 'Hidden (3×8)', rowLabels, cellW: 40, maxAbs: 2 }) +
          '<div style="font-size:1.1rem;color:#666;align-self:center">→</div>' +
          this._matrixTable(this.FFN_out, '#06B6D4', { title: 'FFN Out (3×4)', rowLabels, maxAbs: 2 }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">FFN 对每个位置<b>独立</b>应用（参数共享）。升维→ReLU→降维 引入非线性和丰富的变换空间。</div>' +
      '</div>';
    }
    if (st === 11) {
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Add & LayerNorm 3 → Decoder Block 最终输出</div>' +
        '<div class="formula-box">AddLN₃ = LayerNorm(AddLN₂ + FFN_out)</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">' +
          this._matrixTable(this.AddLN2, '#10B981', { title: 'AddLN₂', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">+</div>' +
          this._matrixTable(this.FFN_out, '#06B6D4', { title: 'FFN Out', rowLabels, maxAbs: 2 }) +
          '<div style="font-size:1.5rem;color:#666;align-self:center">→ LN →</div>' +
          this._matrixTable(this.AddLN3, '#10B981', { title: 'Dec Block 输出', rowLabels, maxAbs: 2 }) +
        '</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">这就是一个 Decoder Block 的完整输出。可以叠加 N 层（GPT-3 用 96 层）。最后一层输出送入投影层。</div>' +
      '</div>';
    }
    if (st === 12) {
      // 取最后一个位置做 next-token 预测
      const lastLogits = this.Logits[this.seq_len - 1];
      const lastProbs = this.Probs[this.seq_len - 1];
      const topIdx = lastProbs.indexOf(Math.max(...lastProbs));
      let bars = '<div style="margin-top:10px">';
      this.vocab.forEach((w, i) => {
        const p = lastProbs[i];
        const isTop = i === topIdx;
        const bw = (p * 100).toFixed(1);
        bars += '<div style="display:flex;align-items:center;gap:8px;margin:4px 0">' +
          '<div style="width:48px;color:' + (isTop ? '#F97316' : '#aaa') + ';font-family:JetBrains Mono,monospace;font-size:0.88rem;text-align:right">' + w + '</div>' +
          '<div style="flex:1;height:20px;background:#0a0a0a;border:1px solid #222;border-radius:3px;overflow:hidden">' +
            '<div style="height:100%;width:' + bw + '%;background:' + (isTop ? '#F97316' : '#3B82F6') + ';transition:width 0.4s"></div>' +
          '</div>' +
          '<div style="width:70px;color:' + (isTop ? '#F97316' : '#aaa') + ';font-family:JetBrains Mono,monospace;font-size:0.78rem">' + (p * 100).toFixed(1) + '%</div>' +
          '<div style="width:70px;color:#666;font-family:JetBrains Mono,monospace;font-size:0.72rem">logit=' + lastLogits[i].toFixed(2) + '</div>' +
        '</div>';
      });
      bars += '</div>';
      return '<div class="flow-block active">' +
        '<div style="font-weight:600;margin-bottom:8px">Linear + Softmax → 预测下一个 token</div>' +
        '<div class="formula-box">Logits = DecOut · W_out &nbsp; (3×4) · (4×|V|) = (3×|V|)</div>' +
        '<div class="formula-box">Probs = softmax(Logits)，取最后一行做 next-token 预测</div>' +
        '<div style="margin-top:10px">' +
          this._matrixTable(this.Logits, '#F97316', { title: 'Logits (3×5)', rowLabels, colLabels: this.vocab, maxAbs: 2 }) +
        '</div>' +
        '<div style="margin-top:10px">' +
          this._matrixTable(this.Probs, '#F97316', { title: 'Probs (3×5)', rowLabels, colLabels: this.vocab, probMode: true }) +
        '</div>' +
        '<div style="margin-top:14px;font-weight:600">最后一位 (Q="猫") 的下一个 token 预测概率：</div>' +
        bars +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">贪心解码：argmax = "<b style="color:#F97316">' + this.vocab[topIdx] + '</b>"。实际会用束搜索/采样等策略。</div>' +
      '</div>';
    }
    return '';
  }

  render() {
    if (!this.container) return;
    const st = this.currentStep;

    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._tdInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._tdInstance.play()">▶ 播放</button>';

    const stepBar = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff' : (i < st ? 'background:#1e3a5f;color:#60a5fa' : 'background:#1a1a1a;color:#666');
      return '<div onclick="window._tdInstance.goTo(' + i + ')" style="padding:5px 9px;border:1px solid #333;border-radius:14px;font-size:0.68rem;cursor:pointer;white-space:nowrap;' + active + '">' + (i + 1) + '. ' + s + '</div>';
    }).join('');

    const stepView = this._stepView(st);
    const arch = this._renderArchDiagram(st);

    // 教育面板
    const eduPanels =
      '<div class="two-col">' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">为什么需要 Causal Mask？</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '• Decoder 是<b>自回归</b>的：生成第 i 个 token 时只能用前 i 个<br>' +
            '• 训练时所有位置<b>并行</b>计算（teacher forcing），若不 mask 会"偷看"标签 → 信息泄露<br>' +
            '• Mask 把 Score 上三角置为 -∞，softmax 后自然为 0，实现并行但仍保持因果结构<br>' +
            '• 推理时逐 token 生成，自然不需要看到未来 — Mask 保证训练/推理一致' +
          '</div>' +
        '</div>' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">Cross-Attention vs Self-Attention</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '• <b>Self-Attn</b>：Q, K, V 来自<b>同一</b>序列 — 序列内部建立联系<br>' +
            '• <b>Cross-Attn</b>：Q 来自 decoder，K/V 来自 <b>encoder</b> 输出 — 跨序列对齐<br>' +
            '• Cross-Attn <b>不加 causal mask</b>：target 位置可以看到 source 的所有位置<br>' +
            '• 是 seq2seq（翻译、摘要）的信息桥梁，也是"注意力对齐"直觉的来源<br>' +
            '• Encoder-only（BERT）没有 Cross-Attn；Decoder-only（GPT）也没有' +
          '</div>' +
        '</div>' +
      '</div>';

    const eduPanels2 =
      '<div class="two-col">' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">Teacher Forcing vs 自回归生成</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '<b>训练 (Teacher Forcing)</b><br>' +
            '• 输入整段目标序列（shift right 后加 &lt;BOS&gt;）<br>' +
            '• 一次前向 + causal mask 得到所有位置的预测<br>' +
            '• 每个位置和真实 label 算 cross-entropy 损失<br>' +
            '• 训练快（并行）但 "exposure bias"：模型只见过正确前缀<br><br>' +
            '<b>推理 (Autoregressive)</b><br>' +
            '• 从 &lt;BOS&gt; 开始逐步生成<br>' +
            '• 每步：把已生成的 tokens 喂给 decoder，预测下一个<br>' +
            '• 贪心/束搜索/采样选出 token，append 回输入<br>' +
            '• 慢（串行）但与部署一致' +
          '</div>' +
        '</div>' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">Decoder-only (GPT) 有何不同？</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '• 没有 Encoder，故<b>没有 Cross-Attention</b>，只保留 Masked Self-Attn<br>' +
            '• 每个 block：MaskedAttn → Add&LN → FFN → Add&LN<br>' +
            '• 本可视化中跳过步骤 ⑦⑧⑨ 就是 decoder-only 结构<br>' +
            '• 输入是"上下文 + 已生成 token"，输出是下一个 token 分布<br>' +
            '• 在大规模预训练后单模型可完成翻译/问答/代码等任务（GPT-3/4）' +
          '</div>' +
        '</div>' +
      '</div>';

    this.container.innerHTML =
      '<div class="dec-viz">' +
        '<style>' +
          '.dec-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.dec-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem}' +
          '.dec-viz .ctrl-btn:hover{background:#252525}' +
          '.dec-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.dec-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.dec-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.dec-viz .flow-block{background:#1a1a1a;border-radius:8px;padding:12px 14px;border:1px solid #333;margin-bottom:10px}' +
          '.dec-viz .flow-block.active{border-color:#3B82F6;box-shadow:0 0 14px rgba(59,130,246,0.25)}' +
          '.dec-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '.dec-viz .main-grid{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}' +
          '@media(max-width:1000px){.dec-viz .two-col, .dec-viz .main-grid{grid-template-columns:1fr}}' +
        '</style>' +

        '<div style="font-size:1.4rem;font-weight:600;margin-bottom:4px">Transformer Decoder Block（完整数据流）</div>' +
        '<div style="color:#888;margin-bottom:16px;font-size:0.88rem">目标序列 "我 爱 猫"；Encoder 输出给定；演示 Masked Self-Attn + Cross-Attn + FFN + 三次 Add&LN + 输出头</div>' +

        // 控制
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">控制</div>' +
          '<button class="ctrl-btn" onclick="window._tdInstance.stepBack()">⏮ 上一步</button>' +
          playBtn +
          '<button class="ctrl-btn" onclick="window._tdInstance.stepForward()">⏭ 下一步</button>' +
          '<button class="ctrl-btn" onclick="window._tdInstance.reset()">↻ 重置</button>' +
          '<select onchange="window._tdInstance.setSpeed(this.value)" style="background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 10px;border-radius:6px;font-size:0.85rem;margin-left:4px">' +
            '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="3">3×</option>' +
          '</select>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepBar + '</div>' +
        '</div>' +

        // 主网格：左侧步骤详情、右侧整体架构
        '<div class="main-grid">' +
          '<div>' +
            '<div class="section">' +
              '<div style="font-weight:600;margin-bottom:10px">当前步骤：' + (st + 1) + ' / ' + this.STEPS.length + ' — ' + this.STEPS[st] + '</div>' +
              stepView +
            '</div>' +
          '</div>' +
          '<div>' +
            '<div class="section">' +
              '<div style="font-weight:600;margin-bottom:8px">Encoder-Decoder 架构</div>' +
              '<div style="color:#888;font-size:0.76rem;margin-bottom:6px">亮色块 = 当前步骤激活模块；Cross-Attn 从 Encoder 输出获得 K/V</div>' +
              arch +
            '</div>' +
            '<div class="section">' +
              '<div style="font-weight:600;margin-bottom:8px">核心公式</div>' +
              '<div class="formula-box" style="font-size:0.78rem">MaskedAttn = softmax((QKᵀ/√d_k) + Mask) · V</div>' +
              '<div class="formula-box" style="font-size:0.78rem">CrossAttn = softmax(Q_dec · K_encᵀ / √d_k) · V_enc</div>' +
              '<div class="formula-box" style="font-size:0.78rem">FFN(x) = ReLU(x·W₁+b₁)·W₂+b₂</div>' +
              '<div class="formula-box" style="font-size:0.78rem">Output = LN(residual + sublayer)  ×3</div>' +
              '<div class="formula-box" style="font-size:0.78rem">Logits = Output · W_vocab;  Probs = softmax(Logits)</div>' +
            '</div>' +
          '</div>' +
        '</div>' +

        // 教育面板
        eduPanels +
        eduPanels2 +

        // 维度小结
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">维度速查 (seq_len=3, d_model=4, d_ff=8, |V|=5)</div>' +
          '<div class="formula-box">Input: (3×4)  →  Q/K/V: (3×4)  →  Score: (3×3)  →  Attn₁: (3×4)</div>' +
          '<div class="formula-box">Cross: Q=(3×4), K=(3×4), V=(3×4) → Score₂: (3×3) → Attn₂: (3×4)</div>' +
          '<div class="formula-box">FFN: (3×4) → hidden (3×8) → (3×4)</div>' +
          '<div class="formula-box">Head: (3×4) · W_out(4×5) = Logits (3×5) → Probs (3×5)</div>' +
        '</div>' +
      '</div>';

    window._tdInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._tdInstance === this) {
      try { delete window._tdInstance; } catch (e) { window._tdInstance = null; }
    }
    this.container = null;
  }
}

// 导出
window.TransformerDecoder = TransformerDecoder;
