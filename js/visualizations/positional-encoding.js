// Positional Encoding 可视化 (Sinusoidal)
// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
class PositionalEncodingVisualization {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // 可调参数
    this.seq_len = 16;
    this.d_model = 16;

    // 交互：选中的 (pos, i) 用于分解计算
    this.selPos = 5;
    this.selDim = 6; // i=3 对应 dim=6 (偶数 -> sin)

    // 用于"叠加到 token embedding"的 3-token 例子
    // 句子: "I love cats"
    this.tokens = ['I', 'love', 'cats'];
    this.tokenEmbeddings = [
      [1.0, 0.5, 0.3, 0.2, 0.1, -0.2, 0.4, 0.6, -0.3, 0.2, 0.0, 0.5, 0.1, -0.4, 0.7, 0.2],
      [0.8, 1.0, 0.5, 0.1, -0.1, 0.3, -0.2, 0.4, 0.2, 0.0, 0.6, -0.3, 0.5, 0.1, -0.2, 0.4],
      [0.3, 0.7, 1.0, 0.4, 0.2, -0.3, 0.1, -0.1, 0.5, 0.4, -0.1, 0.2, -0.4, 0.3, 0.6, 0.0]
    ];

    this.STEPS = [
      '问题：Transformer 为什么需要位置编码？',
      '公式：sin / cos 的交替定义',
      '生成 PE 矩阵 (seq_len × d_model)',
      '热力图可视化',
      '不同维度的波形（频率差异）',
      '选定单元格的计算分解',
      '叠加到 Token Embedding：X + PE',
      '关键性质（相对位置 / 外推）'
    ];

    this._recompute();
  }

  // ============================================
  // 核心计算
  // ============================================
  init(container) {
    this.container = container;
    this.currentStep = 0;
    this.render();
    return this;
  }

  _recompute() {
    this.PE = [];
    for (let pos = 0; pos < this.seq_len; pos++) {
      const row = [];
      for (let i = 0; i < this.d_model; i++) {
        const twoI = 2 * Math.floor(i / 2);
        const angle = pos / Math.pow(10000, twoI / this.d_model);
        const v = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
        row.push(v);
      }
      this.PE.push(row);
    }
    // 3-token 对应的 PE（前 3 行，前 d_model 列，这里假设 token 嵌入也是 d_model=16）
    this.PE_small = this.PE.slice(0, 3).map(r => r.slice(0, this.d_model));
    // 叠加
    this.X_plus_PE = this.tokenEmbeddings.map((row, p) =>
      row.map((v, j) => v + this.PE_small[p][j])
    );
  }

  // ============================================
  // 控制 API
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
      this.timer = setTimeout(() => this._auto(), 1600 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  // 参数滑块回调
  setSeqLen(v) {
    this.seq_len = Math.max(4, Math.min(32, parseInt(v, 10) || 16));
    this._recompute();
    this.render();
  }
  setDModel(v) {
    this.d_model = Math.max(4, Math.min(32, parseInt(v, 10) || 16));
    // d_model 变化时要保证偶数（sin/cos 成对）
    if (this.d_model % 2 !== 0) this.d_model += 1;
    this._recompute();
    this.render();
  }
  selectCell(pos, dim) {
    this.selPos = pos;
    this.selDim = dim;
    if (this.currentStep < 5) this.currentStep = 5;
    this.render();
  }

  // ============================================
  // 辅助渲染：颜色、格式化
  // ============================================
  _cellColor(v) {
    // -1..1 => 红 (负) / 蓝 (正) 热力
    const clamp = Math.max(-1, Math.min(1, v));
    if (clamp >= 0) {
      const a = 0.15 + clamp * 0.55;
      return `rgba(56,132,232,${a.toFixed(3)})`; // 蓝
    } else {
      const a = 0.15 + Math.abs(clamp) * 0.55;
      return `rgba(229,78,78,${a.toFixed(3)})`; // 红
    }
  }

  _fmt(v, d = 3) { return (v >= 0 ? ' ' : '') + v.toFixed(d); }

  // 渲染 PE 热力图（SVG 风格的 HTML 表格）
  _renderHeatmap(highlightCell = null, size = 26) {
    const rows = this.seq_len;
    const cols = this.d_model;
    let html = '<div style="overflow:auto;padding:4px;background:#0a0a0a;border:1px solid #222;border-radius:6px">';
    html += '<table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.62rem">';
    // 表头：维度索引
    html += '<tr><th style="padding:2px 6px;color:#666;font-weight:400">pos\\dim</th>';
    for (let j = 0; j < cols; j++) {
      html += `<th style="padding:2px 4px;color:#888;font-weight:400;text-align:center">${j}</th>`;
    }
    html += '</tr>';
    for (let i = 0; i < rows; i++) {
      html += `<tr><td style="padding:2px 6px;color:#888;text-align:right">${i}</td>`;
      for (let j = 0; j < cols; j++) {
        const v = this.PE[i][j];
        const bg = this._cellColor(v);
        const isSel = highlightCell && highlightCell[0] === i && highlightCell[1] === j;
        const br = isSel ? '#F97316' : '#2a2a2a';
        const bw = isSel ? 2 : 1;
        const sv = v.toFixed(2).replace('-0.00', '0.00');
        html += `<td onclick="window._peInstance.selectCell(${i},${j})" style="cursor:pointer;width:${size}px;height:${size}px;padding:0;border:${bw}px solid ${br};background:${bg};color:${Math.abs(v) > 0.55 ? '#fff' : '#e5e5e5'};text-align:center;vertical-align:middle;font-size:0.58rem" title="pos=${i}, dim=${j}, v=${v.toFixed(4)}">${sv}</td>`;
      }
      html += '</tr>';
    }
    html += '</table></div>';
    return html;
  }

  // 波形图（SVG）：绘制 4 个有代表性的维度
  _renderWaves() {
    const W = 540, H = 220, pad = 30;
    // 选 dim = 0 (sin 低 i), 1 (cos 低 i), d/2 (sin 中 i), d-2 (sin 高 i)
    const pick = [0, 1, Math.max(4, Math.floor(this.d_model / 4) * 2), Math.max(6, this.d_model - 2)];
    const colors = ['#3B82F6', '#8B5CF6', '#F59E0B', '#10B981'];
    const names = pick.map(d => (d % 2 === 0 ? 'sin' : 'cos') + ` i=${Math.floor(d / 2)} (dim ${d})`);

    const plotX = v => pad + (v / (this.seq_len - 1)) * (W - 2 * pad);
    const plotY = v => H / 2 - v * (H / 2 - pad);

    let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="background:#0a0a0a;border:1px solid #222;border-radius:6px">`;
    // axes
    svg += `<line x1="${pad}" y1="${H / 2}" x2="${W - pad}" y2="${H / 2}" stroke="#333" stroke-dasharray="2 3"/>`;
    svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="#333"/>`;
    // labels
    svg += `<text x="${W - pad}" y="${H / 2 + 12}" text-anchor="end" fill="#666" font-size="10" font-family="JetBrains Mono,monospace">pos →</text>`;
    svg += `<text x="${pad - 4}" y="${pad - 4}" text-anchor="end" fill="#666" font-size="10" font-family="JetBrains Mono,monospace">+1</text>`;
    svg += `<text x="${pad - 4}" y="${H - pad + 10}" text-anchor="end" fill="#666" font-size="10" font-family="JetBrains Mono,monospace">-1</text>`;

    pick.forEach((dim, idx) => {
      const color = colors[idx];
      let path = '';
      // 采样细粒度曲线
      const pts = 200;
      for (let k = 0; k <= pts; k++) {
        const pos = (k / pts) * (this.seq_len - 1);
        const twoI = 2 * Math.floor(dim / 2);
        const angle = pos / Math.pow(10000, twoI / this.d_model);
        const v = dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
        const x = plotX(pos), y = plotY(v);
        path += (k === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
      }
      svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2" opacity="0.9"/>`;
      // 数据点
      for (let pos = 0; pos < this.seq_len; pos++) {
        const v = this.PE[pos][dim];
        svg += `<circle cx="${plotX(pos).toFixed(1)}" cy="${plotY(v).toFixed(1)}" r="2.5" fill="${color}"/>`;
      }
      // 图例
      svg += `<rect x="${W - pad - 170}" y="${pad + idx * 18}" width="12" height="3" fill="${color}"/>`;
      svg += `<text x="${W - pad - 154}" y="${pad + idx * 18 + 4}" fill="${color}" font-size="10" font-family="JetBrains Mono,monospace">${names[idx]}</text>`;
    });

    svg += '</svg>';
    return svg;
  }

  // 单元格计算分解
  _renderCellBreakdown() {
    const pos = this.selPos, dim = this.selDim;
    const i = Math.floor(dim / 2);
    const twoI = 2 * i;
    const denom = Math.pow(10000, twoI / this.d_model);
    const angle = pos / denom;
    const val = dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    const fn = dim % 2 === 0 ? 'sin' : 'cos';
    return '<div class="formula-box">目标：PE(pos=' + pos + ', dim=' + dim + ') ，对应 i = ⌊' + dim + '/2⌋ = ' + i + ' ，偶/奇 → <b>' + fn + '</b></div>' +
      '<div class="formula-box">1)&nbsp;&nbsp;分母 = 10000^(2i/d_model) = 10000^(' + twoI + '/' + this.d_model + ') ≈ <span style="color:#F59E0B">' + denom.toFixed(4) + '</span></div>' +
      '<div class="formula-box">2)&nbsp;&nbsp;角度 θ = pos / 分母 = ' + pos + ' / ' + denom.toFixed(4) + ' ≈ <span style="color:#F59E0B">' + angle.toFixed(4) + '</span> rad</div>' +
      '<div class="formula-box">3)&nbsp;&nbsp;' + fn + '(θ) = ' + fn + '(' + angle.toFixed(4) + ') ≈ <span style="color:#10B981">' + val.toFixed(4) + '</span></div>' +
      '<div style="color:#888;font-size:0.8rem;margin-top:6px">点击热力图的任意单元格查看对应的分解过程。</div>';
  }

  // 矩阵表（用于叠加展示）
  _miniMatrix(M, color, label, highlight = false) {
    const cols = M[0].length;
    let html = '<div style="margin:4px 0"><div style="color:' + color + ';font-size:0.76rem;margin-bottom:4px;font-family:JetBrains Mono,monospace">' + label + ' (' + M.length + '×' + cols + ')</div>';
    html += '<table style="border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.58rem">';
    M.forEach(row => {
      html += '<tr>';
      row.forEach(v => {
        const bg = this._cellColor(v);
        const bd = highlight ? color : '#2a2a2a';
        html += '<td style="width:36px;height:22px;padding:0;border:1px solid ' + bd + ';background:' + bg + ';color:#e5e5e5;text-align:center">' + v.toFixed(2) + '</td>';
      });
      html += '</tr>';
    });
    html += '</table></div>';
    return html;
  }

  // ============================================
  // 主渲染
  // ============================================
  render() {
    if (!this.container) return;
    const st = this.currentStep;

    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._peInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._peInstance.play()">▶ 播放</button>';

    const stepBar = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff' : (i < st ? 'background:#1e3a5f;color:#60a5fa' : 'background:#1a1a1a;color:#666');
      return '<div onclick="window._peInstance.goTo(' + i + ')" style="padding:6px 10px;border:1px solid #333;border-radius:14px;font-size:0.7rem;cursor:pointer;white-space:nowrap;' + active + '">' + (i + 1) + '. ' + s + '</div>';
    }).join('');

    // ---- 每步主面板 ----
    let stepDetail = '';
    if (st === 0) {
      stepDetail =
        '<div class="formula-box">Self-Attention 本身是<b>置换不变</b>的：如果打乱 token 顺序，输出也只是行顺序打乱，但每行结果不变。</div>' +
        '<div class="formula-box">→ 这意味着纯 Attention 无法区分 "I love cats" 和 "cats love I"！</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:8px">所以我们必须把"位置信息"<b>加到</b>输入里。作者选择的方案：<b>正余弦位置编码</b>——' +
        '让每个位置产生一个固定的、与 pos 相关的向量 PE(pos) ∈ ℝ^d，和 token embedding 直接相加。</div>';
    } else if (st === 1) {
      stepDetail =
        '<div class="formula-box">PE(pos, 2i)&nbsp;&nbsp;= sin( pos / 10000^(2i/d_model) )</div>' +
        '<div class="formula-box">PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:8px">' +
          '• <b>pos</b>：token 在序列中的位置（0, 1, 2, …）<br>' +
          '• <b>i</b>：维度对索引（d_model 个维度被成对分成 d_model/2 对）<br>' +
          '• <b>偶数维用 sin、奇数维用 cos</b>：成对出现，能用三角恒等式解出"相对位置"<br>' +
          '• 分母 10000^(2i/d_model) 让不同维度有<b>指数级变化的频率</b>' +
        '</div>';
    } else if (st === 2) {
      stepDetail =
        '<div class="formula-box">形状：PE ∈ ℝ^(' + this.seq_len + ' × ' + this.d_model + ')</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">在 Transformer 中，这个矩阵是<b>预计算的、不参与梯度更新的常量</b>。每个 batch、每个样本都重复使用它的前 seq_len 行。</div>' +
        '<div class="formula-box" style="margin-top:10px">提示：可拖动下方滑块改变 seq_len / d_model，观察 PE 图案如何变化。</div>';
    } else if (st === 3) {
      stepDetail =
        '<div class="formula-box">颜色编码：<span style="color:#3B82F6">蓝</span> = 正值 / <span style="color:#EF4444">红</span> = 负值 / 深色 ≈ 0</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">观察要点：<br>' +
          '• <b>左侧列（dim 小）</b>：颜色沿 pos 方向快速震荡（高频）<br>' +
          '• <b>右侧列（dim 大）</b>：颜色几乎不变（低频，周期远超 seq_len）<br>' +
          '• 整张图形成"彩虹条纹"式指纹，每一行都是唯一的<br>' +
          '• 点击任意单元格可看到详细计算' +
        '</div>';
    } else if (st === 4) {
      stepDetail =
        '<div class="formula-box">每个维度 dim 对应一条固定的正弦/余弦曲线，频率 = 1 / 10000^(2i/d_model)</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">' +
          '• dim=0 (sin, i=0) 频率最高 —— 周期 ≈ 2π ≈ 6.28，在 ' + this.seq_len + ' 个 token 里快速震荡<br>' +
          '• dim=' + (this.d_model - 2) + ' (sin, i=' + (this.d_model / 2 - 1) + ') 频率最低 —— 周期 ≈ 2π × 10000 ≈ 62832，几乎看不到变化<br>' +
          '• 多频率叠加让每个位置的整体向量都是<b>唯一的指纹</b>，且邻近位置向量相似（连续性）' +
        '</div>';
    } else if (st === 5) {
      stepDetail = this._renderCellBreakdown();
    } else if (st === 6) {
      stepDetail =
        '<div class="formula-box">Input = TokenEmbedding(X) + PE</div>' +
        '<div class="formula-box">形状：(seq_len, d_model) + (seq_len, d_model) = (seq_len, d_model)</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px"><b>直接相加</b>而非拼接。看起来"混合"了信息，但因为不同维度的频率不同，模型可以通过线性变换把位置和内容分离出来。</div>';
    } else if (st === 7) {
      const demoAngle1 = 3 / Math.pow(10000, 4 / this.d_model);
      const demoAngle2 = 5 / Math.pow(10000, 4 / this.d_model);
      stepDetail =
        '<div class="formula-box">关键性质 1 — <b>相对位置线性可导</b></div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">PE(pos+k) 可以写成 PE(pos) 的<b>线性变换</b>（k 只与旋转矩阵相关，和 pos 无关）：</div>' +
        '<div class="formula-box" style="font-size:0.78rem">[sin(θ(pos+k)), cos(θ(pos+k))] = R(kθ) · [sin(θpos), cos(θpos)]</div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">所以 Attention 只需学一组与 k 相关的权重，就能感知"距离 k"，而非硬编码每一对 (pos, pos+k)。</div>' +
        '<div class="formula-box" style="margin-top:14px">关键性质 2 — <b>外推到更长序列</b></div>' +
        '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">公式对任意 pos 都有定义，所以训练 512 长度、推理 1024 也能直接给出 PE。但实际效果依赖网络学到的函数形态（现代模型用 RoPE/ALiBi 外推更好）。</div>';
    }

    // ---- 热力图 + 波形 ----
    const highlightCell = st >= 5 ? [this.selPos, this.selDim] : null;
    const heatmap = this._renderHeatmap(highlightCell);
    const waves = this._renderWaves();

    // ---- 叠加面板（第 6 步） ----
    let overlayHTML = '';
    if (st >= 6) {
      // 显示 3 token 例子（用前 8 列避免太宽）
      const take = 8;
      const X_sub = this.tokenEmbeddings.map(r => r.slice(0, take));
      const PE_sub = this.PE_small.map(r => r.slice(0, take));
      const Y_sub = this.X_plus_PE.map(r => r.slice(0, take));
      overlayHTML =
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">3-token 例子：X + PE (显示前 ' + take + ' 维)</div>' +
          '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
            '<div>' +
              '<div style="color:#888;font-size:0.76rem;margin-bottom:4px">Token Embedding X</div>' +
              this._miniMatrix(X_sub, '#3B82F6', '["I", "love", "cats"]') +
            '</div>' +
            '<div style="font-size:1.5rem;color:#666;margin:0 4px">+</div>' +
            '<div>' +
              '<div style="color:#888;font-size:0.76rem;margin-bottom:4px">Positional Encoding PE</div>' +
              this._miniMatrix(PE_sub, '#8B5CF6', 'PE[0:3, :' + take + ']') +
            '</div>' +
            '<div style="font-size:1.5rem;color:#666;margin:0 4px">=</div>' +
            '<div>' +
              '<div style="color:#888;font-size:0.76rem;margin-bottom:4px">Input to Encoder</div>' +
              this._miniMatrix(Y_sub, '#06B6D4', 'X + PE', true) +
            '</div>' +
          '</div>' +
        '</div>';
    }

    // ---- 参数滑块 ----
    const sliderHTML =
      '<div style="display:flex;gap:18px;margin-top:10px;flex-wrap:wrap;align-items:center">' +
        '<label style="font-size:0.82rem;color:#aaa">seq_len = <span style="color:#60a5fa">' + this.seq_len + '</span>' +
          ' <input type="range" min="8" max="32" step="1" value="' + this.seq_len + '" oninput="window._peInstance.setSeqLen(this.value)" style="vertical-align:middle;margin-left:6px"></label>' +
        '<label style="font-size:0.82rem;color:#aaa">d_model = <span style="color:#60a5fa">' + this.d_model + '</span>' +
          ' <input type="range" min="8" max="32" step="2" value="' + this.d_model + '" oninput="window._peInstance.setDModel(this.value)" style="vertical-align:middle;margin-left:6px"></label>' +
      '</div>';

    // ---- 公式侧边栏 ----
    const formulaSide =
      '<div class="section">' +
        '<div style="font-weight:600;margin-bottom:8px">核心公式</div>' +
        '<div class="formula-box">PE(pos, 2i)&nbsp;&nbsp;= sin(pos / 10000^(2i/d_model))</div>' +
        '<div class="formula-box">PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</div>' +
        '<div style="color:#aaa;font-size:0.8rem;line-height:1.7;margin-top:8px">' +
          '• pos ∈ [0, seq_len)<br>' +
          '• i ∈ [0, d_model/2)，控制"维度对"索引<br>' +
          '• 10000 是常数（论文选的基数）<br>' +
          '• 频率范围：2π ~ 2π×10000（指数分布）' +
        '</div>' +
      '</div>';

    // ---- 教育面板（两栏） ----
    const eduTwoCol =
      '<div class="two-col">' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">为什么是 sin/cos，而不是学习型 PE？</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '• <b>无参数</b>：不引入额外可学习参数，不增加过拟合风险<br>' +
            '• <b>可外推</b>：公式对任意 pos 有定义（学习型受训练长度限制）<br>' +
            '• <b>相对位置线性关系</b>：PE(pos+k) 能由 PE(pos) 线性变换得到（三角恒等式）<br>' +
            '• <b>唯一性</b>：多频率叠加保证每个 pos 向量都不同<br>' +
            '• <b>平滑性</b>：邻近位置向量相似，远距离位置差异大' +
          '</div>' +
        '</div>' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">为什么基数是 10000？</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '• 目的：让<b>最低频</b>维度的周期远大于实际 seq_len，避免"折回"<br>' +
            '• 10000^(2i/d_model) 在 i=0 时 = 1（周期 2π ≈ 6），i=d/2 时 = 10000（周期 ≈ 62832）<br>' +
            '• 多头注意力的每个头能关注不同尺度的距离模式<br>' +
            '• 太小：长距离无分辨力；太大：短距离分辨不足 — 10000 是经验折中' +
          '</div>' +
        '</div>' +
      '</div>';

    const eduTwoCol2 =
      '<div class="two-col">' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">相对位置的三角恒等式</div>' +
          '<div class="formula-box" style="font-size:0.78rem">sin(a+b) = sin(a)cos(b) + cos(a)sin(b)</div>' +
          '<div class="formula-box" style="font-size:0.78rem">cos(a+b) = cos(a)cos(b) − sin(a)sin(b)</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7;margin-top:6px">把 a=pos·ω, b=k·ω 代入即可证明 PE(pos+k) 是 PE(pos) 的线性变换（旋转矩阵 R(kω)）。<br>→ 这让 Attention 可以学到"相对距离"模式。</div>' +
        '</div>' +
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">现代改进 (RoPE / ALiBi)</div>' +
          '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
            '• <b>RoPE</b>（LLaMA、GPT-NeoX）：把位置信息作为<b>旋转</b>嵌入到 Q/K 中，保留绝对+相对位置优点，外推更好<br>' +
            '• <b>ALiBi</b>（BLOOM）：不加 PE，而在 Attention score 里加<b>线性偏置</b> −m·|i−j|，外推极强<br>' +
            '• <b>学习型 PE</b>（BERT、GPT-2）：参数化查表，表达力稍强但不能外推<br>' +
            '• 本可视化展示的是 "Attention is All You Need" 原版 Sinusoidal PE' +
          '</div>' +
        '</div>' +
      '</div>';

    // ---- 最终 HTML ----
    this.container.innerHTML =
      '<div class="pe-viz">' +
        '<style>' +
          '.pe-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.pe-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem}' +
          '.pe-viz .ctrl-btn:hover{background:#252525}' +
          '.pe-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.pe-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.pe-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.pe-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '.pe-viz .three-col{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}' +
          '@media(max-width:900px){.pe-viz .two-col, .pe-viz .three-col{grid-template-columns:1fr}}' +
          '.pe-viz input[type=range]{accent-color:#3B82F6}' +
        '</style>' +

        '<div style="font-size:1.4rem;font-weight:600;margin-bottom:4px">Positional Encoding（正余弦位置编码）</div>' +
        '<div style="color:#888;margin-bottom:16px;font-size:0.88rem">Transformer 原论文版本：用固定频率的正余弦给每个位置生成唯一指纹</div>' +

        // 控制条
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">控制</div>' +
          '<button class="ctrl-btn" onclick="window._peInstance.stepBack()">⏮ 上一步</button>' +
          playBtn +
          '<button class="ctrl-btn" onclick="window._peInstance.stepForward()">⏭ 下一步</button>' +
          '<button class="ctrl-btn" onclick="window._peInstance.reset()">↻ 重置</button>' +
          '<select onchange="window._peInstance.setSpeed(this.value)" style="background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 10px;border-radius:6px;font-size:0.85rem;margin-left:4px">' +
            '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="3">3×</option>' +
          '</select>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepBar + '</div>' +
          sliderHTML +
        '</div>' +

        // 当前步骤详情
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">当前步骤：' + (st + 1) + '. ' + this.STEPS[st] + '</div>' +
          stepDetail +
        '</div>' +

        // 主视化：热力图 + 公式
        '<div class="three-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">PE 热力图（' + this.seq_len + ' × ' + this.d_model + '）' +
              (highlightCell ? ' <span style="font-size:0.78rem;color:#F97316">· 选中 (pos=' + highlightCell[0] + ', dim=' + highlightCell[1] + ')</span>' : '') +
              '</div>' +
            '<div style="color:#888;font-size:0.76rem;margin-bottom:6px">行 = 位置 (0 → ' + (this.seq_len - 1) + ')，列 = 维度 (0 → ' + (this.d_model - 1) + ')，点击单元格查看计算</div>' +
            heatmap +
          '</div>' +
          formulaSide +
        '</div>' +

        // 波形图
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">维度波形：不同维度沿 pos 的震荡曲线</div>' +
          '<div style="color:#888;font-size:0.76rem;margin-bottom:6px">低维（小 dim）高频，高维（大 dim）低频 — 这是"频率指纹"的来源</div>' +
          waves +
        '</div>' +

        // 叠加（仅后段）
        overlayHTML +

        // 教育面板
        eduTwoCol +
        eduTwoCol2 +

        // 维度小结
        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">维度速查</div>' +
          '<div class="formula-box">X (Token Embedding)：(' + this.seq_len + ' × ' + this.d_model + ')</div>' +
          '<div class="formula-box">PE (Positional Encoding)：(' + this.seq_len + ' × ' + this.d_model + ')，<b>不可训练</b></div>' +
          '<div class="formula-box">Input = X + PE：(' + this.seq_len + ' × ' + this.d_model + ')，送入 Encoder/Decoder</div>' +
        '</div>' +
      '</div>';

    window._peInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._peInstance === this) {
      try { delete window._peInstance; } catch (e) { window._peInstance = null; }
    }
    this.container = null;
  }
}

// 导出
window.PositionalEncodingVisualization = PositionalEncodingVisualization;
