// Transformer Encoder Block 可视化 - 完整流水线
// 流程: X → (+PE) → MHA → Add & Norm → FFN → Add & Norm → Output
class TransformerEncoder {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;
    this.hoverCell = null; // {matrix, i, j} - 悬停高亮一个 score 单元

    this.STEPS = [
      { key: 'input',     name: '输入 X',           short: 'X' },
      { key: 'pe',        name: 'X + PE',           short: '+PE' },
      { key: 'qkv',       name: '生成 Q/K/V',       short: 'QKV' },
      { key: 'score',     name: '打分 QK^T',        short: 'QKᵀ' },
      { key: 'scale',     name: '缩放 /√d_k',       short: '÷√d' },
      { key: 'softmax',   name: 'Softmax',          short: 'softmax' },
      { key: 'attn',      name: '加权 ·V',           short: '·V' },
      { key: 'add1',      name: '残差 + 输入',      short: 'Add₁' },
      { key: 'ln1',       name: 'LayerNorm ₁',      short: 'LN₁' },
      { key: 'ffn1',      name: 'FFN₁ (4→8)',       short: 'W₁' },
      { key: 'relu',      name: 'ReLU',             short: 'ReLU' },
      { key: 'ffn2',      name: 'FFN₂ (8→4)',       short: 'W₂' },
      { key: 'add2',      name: '残差 + LN₁',        short: 'Add₂' },
      { key: 'ln2',       name: 'LayerNorm ₂',      short: 'LN₂' }
    ];

    // 固定参数
    this.tokens = ['The', 'cat', 'sat'];
    this.d_model = 4;
    this.d_ff = 8;

    // 输入词嵌入 X (3×4)
    this.X_emb = [
      [1.0, 0.5, 0.3, 0.2],
      [0.8, 1.0, 0.5, 0.1],
      [0.3, 0.7, 1.0, 0.4]
    ];

    // 权重 (Self-Attention)
    this.WQ = [[1,0,0,0],[0,1,0,0],[1,1,0,0],[0,0,1,1]];
    this.WK = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    this.WV = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

    // FFN 权重 (4→8→4)
    this.W1 = [
      [0.5,-0.3, 0.8, 0.1,-0.2, 0.6, 0.0, 0.4],
      [0.2, 0.7,-0.4, 0.5, 0.3,-0.1, 0.9,-0.3],
      [-0.6,0.1, 0.3,-0.2, 0.8, 0.4,-0.5, 0.2],
      [0.4, 0.2,-0.1, 0.7,-0.3, 0.5, 0.6,-0.4]
    ];
    this.b1 = [0.1, 0.0,-0.2, 0.0, 0.1,-0.1, 0.0, 0.2];
    this.W2 = [
      [0.3, 0.1,-0.2, 0.4],
      [-0.5,0.6, 0.2,-0.1],
      [0.2,-0.3, 0.5, 0.0],
      [0.4, 0.1,-0.4, 0.3],
      [-0.1,0.3, 0.2, 0.5],
      [0.5,-0.2, 0.1,-0.3],
      [0.0, 0.4,-0.5, 0.2],
      [0.3, 0.0, 0.4,-0.1]
    ];
    this.b2 = [0.0, 0.1, 0.0,-0.1];

    this._compute();
  }

  init(container) {
    this.container = container;
    this.currentStep = 0;
    this.render();
    return this;
  }

  reset() { this.currentStep = 0; this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); }
  setSpeed(s) { this.speed = parseFloat(s); }
  stepForward() { if (this.currentStep < this.STEPS.length - 1) { this.currentStep++; this.render(); } }
  stepBack() { if (this.currentStep > 0) { this.currentStep--; this.render(); } }
  goTo(i) { this.currentStep = Math.max(0, Math.min(i, this.STEPS.length - 1)); this.render(); }

  _auto() {
    if (!this.isPlaying) return;
    if (this.currentStep < this.STEPS.length - 1) {
      this.currentStep++;
      this.render();
      this.timer = setTimeout(() => this._auto(), 1400 / this.speed);
    } else { this.isPlaying = false; this.render(); }
  }

  // ---------- 计算工具（共享：见 js/utils/math.js） ----------
  // matmul / transpose / scale / matAdd / softmax / layerNormMatrix / reluMatrix
  // 都来自 Utils.*，本模块不再持有私有副本

  _compute() {
    const U = window.Utils;
    const seq_len = this.X_emb.length;
    // 位置编码（与 Utils.generatePositionalEncoding 等价，这里保留内联以便教学注释）
    this.PE = U.generatePositionalEncoding(seq_len, this.d_model);
    this.X = U.matAdd(this.X_emb, this.PE);   // X + PE

    this.Q = U.matmul(this.X, this.WQ);
    this.K = U.matmul(this.X, this.WK);
    this.V = U.matmul(this.X, this.WV);

    this.KT = U.transpose(this.K);
    this.Score = U.matmul(this.Q, this.KT);                       // (3×3)
    this.ScoreScaled = U.scale(this.Score, Math.sqrt(this.d_model));
    this.Attn = U.softmax(this.ScoreScaled);
    this.AttnOut = U.matmul(this.Attn, this.V);                   // (3×4)

    this.Add1 = U.matAdd(this.AttnOut, this.X);
    this.LN1 = U.layerNormMatrix(this.Add1);

    // FFN: Z·W1 + b1 → ReLU → ·W2 + b2
    const addRow = (row, b) => row.map((v, i) => v + b[i]);
    this.FFN1 = this.LN1.map(r => addRow(U.matmul([r], this.W1)[0], this.b1));
    this.ReLU = U.reluMatrix(this.FFN1);
    this.FFN2 = this.ReLU.map(r => addRow(U.matmul([r], this.W2)[0], this.b2));

    this.Add2 = U.matAdd(this.FFN2, this.LN1);
    this.LN2 = U.layerNormMatrix(this.Add2);
    this.Output = this.LN2;
  }

  // ---------- 绘制矩阵 ----------
  _matCells(M, color, opts = {}) {
    const { highlight = null, rowLabels = null, colLabels = null, min = -2, max = 2 } = opts;
    const cellWidth = opts.cellWidth || 54;

    let html = '<div style="display:inline-block;vertical-align:top">';
    // Col labels
    if (colLabels) {
      html += '<div style="display:flex;margin-left:' + (rowLabels ? '44px' : '0') + '">';
      colLabels.forEach(l => {
        html += '<div style="min-width:' + cellWidth + 'px;text-align:center;color:#666;font-size:0.7rem;margin:0 2px">' + l + '</div>';
      });
      html += '</div>';
    }
    html += '<div>';
    M.forEach((row, i) => {
      html += '<div style="display:flex;align-items:center">';
      if (rowLabels) {
        html += '<div style="min-width:40px;text-align:right;color:' + color + ';font-size:0.72rem;padding-right:4px;font-family:JetBrains Mono,monospace">' + rowLabels[i] + '</div>';
      }
      row.forEach((v, j) => {
        const norm = Math.max(-1, Math.min(1, v / Math.max(Math.abs(min), Math.abs(max))));
        const intensity = Math.min(Math.abs(norm) * 0.5 + 0.12, 0.55);
        let bg = v === 0 ? '#0c0c0c' : (v > 0 ? 'rgba(16,185,129,' + intensity + ')' : 'rgba(239,68,68,' + intensity + ')');
        let border = color;
        let borderW = '1px';
        if (highlight && highlight.some(h => h.i === i && h.j === j)) {
          border = '#F97316';
          borderW = '2px';
          bg = 'rgba(249,115,22,0.35)';
        }
        html += '<div style="min-width:' + cellWidth + 'px;padding:5px 4px;margin:2px;background:' + bg + ';border:' + borderW + ' solid ' + border + ';border-radius:3px;color:' + color + ';font-family:JetBrains Mono,monospace;font-size:0.72rem;text-align:center">' + v.toFixed(2) + '</div>';
      });
      html += '</div>';
    });
    html += '</div>';
    html += '</div>';
    return html;
  }

  _matLabel(name, shape, color) {
    return '<div style="color:' + color + ';font-family:JetBrains Mono,monospace;font-size:0.78rem;font-weight:600;margin-bottom:4px">' + name + ' <span style="color:#666;font-weight:400">' + shape + '</span></div>';
  }

  // ---------- 架构图 ----------
  _archDiagram(currentKey) {
    // 流水线阶段对应的 CSS class
    const states = {
      attn: ['qkv', 'score', 'scale', 'softmax', 'attn'].includes(currentKey),
      add1: currentKey === 'add1',
      ln1: currentKey === 'ln1',
      ffn: ['ffn1', 'relu', 'ffn2'].includes(currentKey),
      add2: currentKey === 'add2',
      ln2: currentKey === 'ln2',
      pe: currentKey === 'pe',
      input: currentKey === 'input'
    };

    const idx = this.STEPS.findIndex(s => s.key === currentKey);
    const done = (key) => this.STEPS.findIndex(s => s.key === key) < idx;
    const active = (key) => this.STEPS.findIndex(s => s.key === key) === idx;

    const block = (label, color, isActive, isDone, sub) => {
      const bg = isActive ? 'rgba(59,130,246,0.2)' : (isDone ? '#0f1f15' : '#0a0a0a');
      const bd = isActive ? color : (isDone ? '#233' : '#222');
      const glow = isActive ? ';box-shadow:0 0 12px rgba(59,130,246,0.45)' : '';
      return '<div style="padding:10px 14px;border:2px solid ' + bd + ';background:' + bg + ';border-radius:8px;min-width:160px;text-align:center' + glow + '">' +
        '<div style="color:' + color + ';font-weight:600;font-size:0.85rem">' + label + '</div>' +
        (sub ? '<div style="color:#666;font-size:0.7rem;margin-top:2px">' + sub + '</div>' : '') +
        '</div>';
    };
    const arrow = (color = '#3B82F6') => '<div style="height:22px;width:2px;background:' + color + ';margin:4px auto;position:relative">' +
      '<div style="position:absolute;bottom:-4px;left:-4px;width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:6px solid ' + color + '"></div>' +
      '</div>';
    const residualSide = () => '<div style="position:absolute;right:-40px;top:0;height:100%;display:flex;align-items:center;color:#F59E0B;font-family:JetBrains Mono,monospace;font-size:0.72rem">⤵ residual</div>';

    return '<div style="display:flex;flex-direction:column;align-items:center;position:relative">' +
      block('Input Embedding', '#3B82F6', active('input') || active('pe'), done('input'), 'X ∈ ℝ^(3×4)') +
      arrow() +
      block('+ Positional Encoding', '#8B5CF6', active('pe'), done('pe'), 'sin / cos') +
      arrow() +
      '<div style="position:relative">' +
        block('Multi-Head Self-Attention', '#06B6D4', states.attn, idx > this.STEPS.findIndex(s => s.key === 'attn'), 'Q·Kᵀ → softmax → ·V') +
      '</div>' +
      arrow('#F59E0B') +
      block('Add & LayerNorm ①', '#10B981', states.add1 || states.ln1, done('ln1'), 'residual from Input') +
      arrow() +
      block('Feed-Forward (4→8→4)', '#F59E0B', states.ffn, idx > this.STEPS.findIndex(s => s.key === 'ffn2'), 'Linear → ReLU → Linear') +
      arrow('#F59E0B') +
      block('Add & LayerNorm ②', '#10B981', states.add2 || states.ln2, false, 'residual from LN₁') +
      arrow() +
      block('Encoder Output', '#3B82F6', active('ln2'), false, 'Y ∈ ℝ^(3×4)') +
      '</div>';
  }

  // ---------- 当前步骤的详情 ----------
  _stepDetail(st) {
    const key = st.key;
    const tokenLabels = this.tokens.map((t, i) => t + ' (t' + i + ')');
    const dimLabels4 = ['d₀', 'd₁', 'd₂', 'd₃'];
    const dimLabels8 = ['h₀', 'h₁', 'h₂', 'h₃', 'h₄', 'h₅', 'h₆', 'h₇'];

    let main = '', formula = '', narration = '';

    if (key === 'input') {
      main = this._matLabel('X (词嵌入)', '(3×4)', '#3B82F6') +
        this._matCells(this.X_emb, '#3B82F6', { rowLabels: tokenLabels, colLabels: dimLabels4 });
      formula = 'X ∈ ℝ^(seq_len × d_model) = ℝ^(3 × 4)';
      narration = '输入是 3 个 token 的嵌入向量。每个 token 是一个 4 维的词嵌入（词向量），从预训练的嵌入表中查询得到。';
    } else if (key === 'pe') {
      main = this._matLabel('PE (位置编码)', '(3×4)', '#8B5CF6') +
        this._matCells(this.PE, '#8B5CF6', { rowLabels: ['pos=0', 'pos=1', 'pos=2'], colLabels: dimLabels4 }) +
        '<div style="margin:12px 0;text-align:center;color:#666">+</div>' +
        this._matLabel('X + PE', '(3×4)', '#3B82F6') +
        this._matCells(this.X, '#3B82F6', { rowLabels: tokenLabels, colLabels: dimLabels4 });
      formula = 'PE(pos, 2i) = sin(pos / 10000^(2i/d))   |   PE(pos, 2i+1) = cos(...)';
      narration = 'Transformer 对位置不敏感（permutation-invariant），需要显式加上位置编码，把位置信息注入到词向量中。';
    } else if (key === 'qkv') {
      main = this._matLabel('Q = X · W_Q', '(3×4)', '#3B82F6') +
        this._matCells(this.Q, '#3B82F6', { rowLabels: tokenLabels }) +
        '<div style="height:6px"></div>' +
        this._matLabel('K = X · W_K', '(3×4)', '#8B5CF6') +
        this._matCells(this.K, '#8B5CF6', { rowLabels: tokenLabels }) +
        '<div style="height:6px"></div>' +
        this._matLabel('V = X · W_V', '(3×4)', '#10B981') +
        this._matCells(this.V, '#10B981', { rowLabels: tokenLabels });
      formula = 'Q = X W_Q ∈ ℝ^(3×4)   |   K = X W_K   |   V = X W_V';
      narration = 'Q（查询）、K（键）、V（值）由 3 个可学习权重矩阵从 X 投影得到。把"这是什么"与"查找什么"分离到不同子空间。';
    } else if (key === 'score') {
      // 高亮第 (1,2) 个分数: token "cat" 对 token "sat" 的分数
      const hi = [{ i: 1, j: 2 }];
      main = this._matLabel('Q', '(3×4)', '#3B82F6') +
        this._matCells(this.Q, '#3B82F6', { rowLabels: tokenLabels, highlight: [{i:1,j:0},{i:1,j:1},{i:1,j:2},{i:1,j:3}] }) +
        '<div style="margin:10px 0;text-align:center;color:#666">× Kᵀ</div>' +
        this._matLabel('Kᵀ', '(4×3)', '#8B5CF6') +
        this._matCells(this.KT, '#8B5CF6', { colLabels: tokenLabels, highlight: [{i:0,j:2},{i:1,j:2},{i:2,j:2},{i:3,j:2}] }) +
        '<div style="margin:10px 0;text-align:center;color:#666">=</div>' +
        this._matLabel('Score = Q · Kᵀ', '(3×3)', '#F97316') +
        this._matCells(this.Score, '#F97316', { rowLabels: tokenLabels, colLabels: tokenLabels, highlight: hi, min: -4, max: 4 });
      // 算一下高亮单元的具体数值
      const q1 = this.Q[1], k2 = this.K[2];
      const terms = q1.map((v, i) => v.toFixed(2) + '·' + k2[i].toFixed(2));
      formula = 'Score[1,2] = dot(Q[1], K[2]) = ' + terms.join(' + ') + ' = <b style="color:#F97316">' + this.Score[1][2].toFixed(3) + '</b>';
      narration = '分数矩阵 Score[i,j] = 第 i 个 token 的 query 和第 j 个 token 的 key 的点积，度量两个 token 的相关性。橙色框高亮了 "cat → sat" 的分数。';
    } else if (key === 'scale') {
      main = this._matLabel('Score', '(3×3)', '#F97316') +
        this._matCells(this.Score, '#F97316', { rowLabels: tokenLabels, colLabels: tokenLabels, min: -4, max: 4 }) +
        '<div style="margin:10px 0;text-align:center;color:#666">÷ √d_k = √4 = 2</div>' +
        this._matLabel('Score / √d_k', '(3×3)', '#F97316') +
        this._matCells(this.ScoreScaled, '#F97316', { rowLabels: tokenLabels, colLabels: tokenLabels, min: -4, max: 4 });
      formula = 'ScaledScore = Score / √d_k   |   d_k = d_model / num_heads = 4 / 1 = 4';
      narration = '缩放防止点积随维度变大而过大，否则 softmax 会进入饱和区、梯度消失。这里 d_k=4，√d_k=2。';
    } else if (key === 'softmax') {
      // 对高亮行做 softmax 分解
      const hiRow = 1;
      const row = this.ScoreScaled[hiRow];
      const mx = Math.max(...row);
      const ex = row.map(v => Math.exp(v - mx));
      const sum = ex.reduce((a, b) => a + b, 0);
      main = this._matLabel('Score / √d_k (每行)', '(3×3)', '#F97316') +
        this._matCells(this.ScoreScaled, '#F97316', { rowLabels: tokenLabels, colLabels: tokenLabels, highlight: row.map((_,j)=>({i:hiRow,j})), min: -4, max: 4 }) +
        '<div style="margin:10px 0;text-align:center;color:#666">↓ softmax 按行归一化</div>' +
        this._matLabel('Attention Weights', '(3×3)', '#F97316') +
        this._matCells(this.Attn, '#F97316', { rowLabels: tokenLabels, colLabels: tokenLabels, min: 0, max: 1 });
      formula = 'softmax(x_j) = exp(x_j) / Σ_k exp(x_k)   |   行 ' + hiRow + ' 的 [e^x_j] ≈ [' + ex.map(v => v.toFixed(2)).join(', ') + '], Σ = ' + sum.toFixed(2);
      narration = 'softmax 把每行分数转为概率分布（每行和=1）。权重高 = 当前 token 更关注该 token 的内容。';
    } else if (key === 'attn') {
      // 高亮 output[0] = attn[0] · V
      const hi = [{i:0, j:0}, {i:0, j:1}, {i:0, j:2}, {i:0, j:3}];
      main = this._matLabel('Attn Weights', '(3×3)', '#F97316') +
        this._matCells(this.Attn, '#F97316', { rowLabels: tokenLabels, colLabels: tokenLabels, highlight: [{i:0,j:0},{i:0,j:1},{i:0,j:2}], min: 0, max: 1 }) +
        '<div style="margin:10px 0;text-align:center;color:#666">× V</div>' +
        this._matLabel('V', '(3×4)', '#10B981') +
        this._matCells(this.V, '#10B981', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#666">=</div>' +
        this._matLabel('Attention Output', '(3×4)', '#06B6D4') +
        this._matCells(this.AttnOut, '#06B6D4', { rowLabels: tokenLabels, highlight: hi });
      const w = this.Attn[0];
      formula = 'AttnOut[0] = ' + w.map((p, j) => p.toFixed(2) + '·V[' + j + ']').join(' + ') + '   (加权求和)';
      narration = '每个 token 的输出是所有 V 的加权和，权重就是 softmax 后的注意力概率。';
    } else if (key === 'add1') {
      main = this._matLabel('Attention Output', '(3×4)', '#06B6D4') +
        this._matCells(this.AttnOut, '#06B6D4', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#F59E0B">+ X (残差)</div>' +
        this._matLabel('X (原输入+PE)', '(3×4)', '#3B82F6') +
        this._matCells(this.X, '#3B82F6', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#666">=</div>' +
        this._matLabel('Add₁', '(3×4)', '#10B981') +
        this._matCells(this.Add1, '#10B981', { rowLabels: tokenLabels });
      formula = 'Add₁ = AttentionOutput + X   (Residual Connection / Skip Connection)';
      narration = '残差连接让梯度可直接回传，同时保留原输入信息。ResNet 的思想被 Transformer 继承，是深层训练的关键。';
    } else if (key === 'ln1') {
      // 展开一行 LayerNorm 的计算
      const row = this.Add1[0];
      const mu = row.reduce((s, v) => s + v, 0) / row.length;
      const va = row.reduce((s, v) => s + (v - mu) ** 2, 0) / row.length;
      main = this._matLabel('Add₁', '(3×4)', '#10B981') +
        this._matCells(this.Add1, '#10B981', { rowLabels: tokenLabels, highlight: [0,1,2,3].map(j=>({i:0,j})) }) +
        '<div style="margin:10px 0;text-align:center;color:#666">↓ 对每一行独立 LayerNorm</div>' +
        this._matLabel('LN₁', '(3×4)', '#10B981') +
        this._matCells(this.LN1, '#10B981', { rowLabels: tokenLabels });
      formula = '对 token 0 一行：μ = ' + mu.toFixed(3) + ', σ² = ' + va.toFixed(3) + ', x̂ = (x-μ)/√(σ²+ε)';
      narration = 'LayerNorm 在"特征维度"上归一化（不跨样本），稳定训练。γ/β 可学习（这里简化为 γ=1, β=0）。';
    } else if (key === 'ffn1') {
      main = this._matLabel('LN₁', '(3×4)', '#10B981') +
        this._matCells(this.LN1, '#10B981', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#666">× W₁ (4×8) + b₁</div>' +
        this._matLabel('FFN₁ 输出 (线性升维)', '(3×8)', '#F59E0B') +
        this._matCells(this.FFN1, '#F59E0B', { rowLabels: tokenLabels, colLabels: dimLabels8 });
      formula = 'z₁ = LN₁ · W₁ + b₁   |   shape: (3×4) · (4×8) = (3×8)';
      narration = 'FFN 第 1 层：把每个位置的 4 维向量独立升到 8 维（通常 d_ff = 4·d_model）。Position-wise：各位置共享权重。';
    } else if (key === 'relu') {
      main = this._matLabel('FFN₁ 输出 (未激活)', '(3×8)', '#F59E0B') +
        this._matCells(this.FFN1, '#F59E0B', { rowLabels: tokenLabels, colLabels: dimLabels8 }) +
        '<div style="margin:10px 0;text-align:center;color:#666">↓ ReLU</div>' +
        this._matLabel('ReLU(FFN₁)', '(3×8)', '#F59E0B') +
        this._matCells(this.ReLU, '#F59E0B', { rowLabels: tokenLabels, colLabels: dimLabels8 });
      const killed = this.FFN1.flat().filter(v => v < 0).length;
      formula = 'ReLU(x) = max(0, x)   |   ' + killed + ' / 24 个元素被 ReLU 置零';
      narration = 'ReLU 引入非线性，让 FFN 能表达非线性变换。x<0 的通道"关闭"，增强稀疏性。';
    } else if (key === 'ffn2') {
      main = this._matLabel('ReLU(FFN₁)', '(3×8)', '#F59E0B') +
        this._matCells(this.ReLU, '#F59E0B', { rowLabels: tokenLabels, colLabels: dimLabels8 }) +
        '<div style="margin:10px 0;text-align:center;color:#666">× W₂ (8×4) + b₂</div>' +
        this._matLabel('FFN₂ 输出 (线性降维)', '(3×4)', '#F59E0B') +
        this._matCells(this.FFN2, '#F59E0B', { rowLabels: tokenLabels });
      formula = 'z₂ = ReLU(z₁) · W₂ + b₂   |   shape: (3×8) · (8×4) = (3×4)';
      narration = 'FFN 第 2 层：从 8 维降回 4 维，保持与输入同形，方便接下一个残差连接。';
    } else if (key === 'add2') {
      main = this._matLabel('FFN₂ 输出', '(3×4)', '#F59E0B') +
        this._matCells(this.FFN2, '#F59E0B', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#F59E0B">+ LN₁ (残差)</div>' +
        this._matLabel('LN₁', '(3×4)', '#10B981') +
        this._matCells(this.LN1, '#10B981', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#666">=</div>' +
        this._matLabel('Add₂', '(3×4)', '#10B981') +
        this._matCells(this.Add2, '#10B981', { rowLabels: tokenLabels });
      formula = 'Add₂ = FFN(LN₁) + LN₁   (第二个残差连接)';
      narration = '第二次残差连接：保留 attention 分支的信息，再叠加 FFN 的非线性变换。';
    } else if (key === 'ln2') {
      main = this._matLabel('Add₂', '(3×4)', '#10B981') +
        this._matCells(this.Add2, '#10B981', { rowLabels: tokenLabels }) +
        '<div style="margin:10px 0;text-align:center;color:#666">↓ LayerNorm</div>' +
        this._matLabel('最终输出 Y = LN₂', '(3×4)', '#3B82F6') +
        this._matCells(this.LN2, '#3B82F6', { rowLabels: tokenLabels });
      formula = 'Y = LayerNorm(Add₂)   ∈ ℝ^(3×4)';
      narration = '编码器块输出 Y，与输入同形，可作为下一个 Encoder Block 的输入（原论文堆叠 N=6 层）。';
    }

    return { main, formula, narration };
  }

  // ---------- 主渲染 ----------
  render() {
    if (!this.container) return;
    const st = this.STEPS[this.currentStep];
    const det = this._stepDetail(st);

    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._teInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._teInstance.play()">▶ 播放</button>';

    // 步骤进度条
    const stepBar = this.STEPS.map((s, i) => {
      const isActive = i === this.currentStep;
      const isDone = i < this.currentStep;
      const bg = isActive ? '#3B82F6' : (isDone ? '#1e3a5f' : '#1a1a1a');
      const color = isActive ? '#fff' : (isDone ? '#60a5fa' : '#666');
      return '<div onclick="window._teInstance.goTo(' + i + ')" ' +
        'style="padding:5px 9px;border:1px solid #333;border-radius:4px;font-size:0.7rem;cursor:pointer;white-space:nowrap;background:' + bg + ';color:' + color + '" ' +
        'title="' + s.name + '">' + (i + 1) + '. ' + s.short + '</div>';
    }).join('');

    this.container.innerHTML =
      '<div class="te-viz">' +
        '<style>' +
          '.te-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.te-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem}' +
          '.te-viz .ctrl-btn:hover{background:#252525;border-color:#3B82F6}' +
          '.te-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.te-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0;overflow-x:auto}' +
          '.te-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.te-viz .edu-panel{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px 16px;margin-bottom:14px}' +
          '.te-viz .edu-title{font-size:0.95rem;font-weight:700;color:#fff;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #21262d}' +
          '.te-viz .edu-row{margin-bottom:8px;font-size:0.82rem;color:#8b949e;line-height:1.65}' +
          '.te-viz .edu-row b{color:#58a6ff}' +
          '.te-viz .edu-row code{background:#161b22;padding:1px 5px;border-radius:3px;color:#f0883e;font-family:JetBrains Mono,monospace;font-size:0.78rem}' +
          '.te-viz .two-col{display:grid;grid-template-columns:1.6fr 1fr;gap:14px}' +
          '@media(max-width:1100px){.te-viz .two-col{grid-template-columns:1fr}}' +
          '.te-viz .pipe-cols{display:grid;grid-template-columns:220px 1fr;gap:16px}' +
          '@media(max-width:900px){.te-viz .pipe-cols{grid-template-columns:1fr}}' +
        '</style>' +

        '<div style="font-size:1.4rem;font-weight:700;margin-bottom:4px">Transformer Encoder Block</div>' +
        '<div style="color:#888;margin-bottom:14px;font-size:0.88rem">X → (+PE) → Self-Attention → Add&Norm → FFN → Add&Norm → Y  &nbsp;|&nbsp;  d_model=4, d_ff=8, seq_len=3</div>' +

        // 控制
        '<div class="section">' +
          '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">' +
            '<button class="ctrl-btn" onclick="window._teInstance.stepBack()">⏮ 上一步</button>' +
            playBtn +
            '<button class="ctrl-btn" onclick="window._teInstance.stepForward()">下一步 ⏭</button>' +
            '<button class="ctrl-btn" onclick="window._teInstance.reset()">↻ 重置</button>' +
            '<select class="ctrl-btn" onchange="window._teInstance.setSpeed(this.value)" style="padding:6px 12px">' +
              '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option>' +
            '</select>' +
            '<span style="color:#888;font-size:0.82rem;margin-left:8px">步骤 ' + (this.currentStep + 1) + ' / ' + this.STEPS.length + '</span>' +
          '</div>' +
          '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:12px">' + stepBar + '</div>' +
        '</div>' +

        '<div class="pipe-cols">' +
          '<div class="section" style="display:flex;flex-direction:column;align-items:stretch">' +
            '<div style="font-weight:600;margin-bottom:10px;text-align:center">架构流水线</div>' +
            this._archDiagram(st.key) +
          '</div>' +

          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px;color:#60a5fa">▸ 第 ' + (this.currentStep + 1) + ' 步: ' + st.name + '</div>' +
            '<div class="formula-box">' + det.formula + '</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-bottom:10px">' + det.narration + '</div>' +
            '<div style="overflow-x:auto;padding:4px">' + det.main + '</div>' +
          '</div>' +
        '</div>' +

        // 教学面板
        '<div class="two-col">' +
          '<div class="edu-panel">' +
            '<div class="edu-title">📖 核心概念精讲</div>' +
            '<div class="edu-row"><b>Q / K / V 的本质</b>：给同一个序列做<b>三种不同的线性投影</b>。Q 代表"我想找什么"，K 代表"我能被谁找到"，V 代表"我真正的内容"。注意力 = 按"Q 和 K 的相关度"加权平均 V。</div>' +
            '<div class="edu-row"><b>为什么除 √d_k</b>：假设 Q、K 各维度 ~ N(0,1)，那么 Q·K 是 d_k 个独立项之和，方差为 d_k。除以 √d_k 让方差回到 1，softmax 就不会进入饱和区导致梯度消失。</div>' +
            '<div class="edu-row"><b>残差连接 (Add)</b>：输出 = <code>Sublayer(x) + x</code>。好处：①梯度直接通过恒等映射回传，不被 Sublayer 的 Jacobian 压缩；②初始化时子层近似恒等，训练更稳定。</div>' +
            '<div class="edu-row"><b>LayerNorm 不同于 BatchNorm</b>：LayerNorm 在"<b>单样本</b>的特征维度"上归一；BatchNorm 跨 batch。LN 对 batch size 不敏感，适合变长序列（NLP）。</div>' +
            '<div class="edu-row"><b>FFN 为什么"升维又降维"</b>：Attention 本质是线性加权，表达力不足；FFN 用 <code>Linear→ReLU→Linear</code> 的 MLP 注入非线性。升维到 4×d_model 是增加参数容量、学习更丰富特征的关键。</div>' +
          '</div>' +

          '<div class="edu-panel">' +
            '<div class="edu-title">📐 LaTeX 公式对照</div>' +
            '<div class="formula-box">Attention(Q,K,V) = softmax(QKᵀ / √d_k) V</div>' +
            '<div class="formula-box">FFN(x) = max(0, xW₁ + b₁) W₂ + b₂</div>' +
            '<div class="formula-box">LN(x) = γ · (x - μ) / √(σ² + ε) + β</div>' +
            '<div class="formula-box">EncoderBlock(x) = LN(FFN(LN(x + Attn(x))) + LN(x + Attn(x)))</div>' +
            '<div class="edu-row" style="margin-top:8px"><b>固定参数</b>：d_model=4, seq_len=3, d_ff=8（为了可视化简化；原论文 d_model=512, d_ff=2048, N=6 层堆叠）。</div>' +
            '<div class="edu-row"><b>Encoder 堆叠</b>：本模块是<b>一层</b>的视角。实际 Transformer 通常 6-24 层，输出依次作为下一层的输入，每层的 Q/K/V 和 FFN 权重是独立的。</div>' +
            '<div class="edu-row"><b>Post-LN vs Pre-LN</b>：本模块展示 <code>Post-LN</code>（原论文）即 <code>LN(x + Sublayer(x))</code>。GPT 等现代模型多用 <code>Pre-LN</code>：<code>x + Sublayer(LN(x))</code>，训练更稳定、不需要 warmup。</div>' +
          '</div>' +
        '</div>' +

        // 整体流程总结
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">🔄 整块流程 (12 步) 速览</div>' +
          '<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#aaa;line-height:1.9">' +
            '<div><span style="color:#3B82F6">1. X</span> (3×4) → <span style="color:#8B5CF6">+ PE</span> → <span style="color:#3B82F6">X\'</span></div>' +
            '<div><span style="color:#3B82F6">X\'</span> → <span style="color:#666">三路投影</span> → Q, K, V  (各 3×4)</div>' +
            '<div>Score = Q · Kᵀ  (3×3)  →  / √d_k  →  <span style="color:#F97316">softmax</span>  →  <span style="color:#F97316">Attn</span></div>' +
            '<div>AttnOut = Attn · V  (3×4)  →  <span style="color:#F59E0B">+ X\'</span>  →  <span style="color:#10B981">LN₁</span>  (3×4)</div>' +
            '<div>FFN: LN₁ → W₁ (4→8) → <span style="color:#F59E0B">ReLU</span> → W₂ (8→4) → <span style="color:#F59E0B">+ LN₁</span> → <span style="color:#10B981">LN₂</span>  (3×4)</div>' +
            '<div style="color:#60a5fa;margin-top:6px">↳ 输出 Y ∈ ℝ^(3×4) 送入下一层 Encoder / Decoder 的 Cross-Attention</div>' +
          '</div>' +
        '</div>' +
      '</div>';

    window._teInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._teInstance === this) {
      try { delete window._teInstance; } catch (e) { window._teInstance = null; }
    }
    this.container = null;
  }
}

window.TransformerEncoder = TransformerEncoder;
