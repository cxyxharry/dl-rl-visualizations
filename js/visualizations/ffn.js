// Position-wise Feed-Forward Network 可视化
// FFN(x) = max(0, x·W1 + b1)·W2 + b2
class FFNVisualization {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // d_model = 4, d_ff = 8
    // 输入向量 (单个 token，d_model=4)
    this.x = [1.0, 0.5, 0.3, 0.2];

    // W1: 4×8
    this.W1 = [
      [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.0, 0.4],
      [0.2, 0.7, -0.4, 0.5, 0.3, -0.1, 0.9, -0.3],
      [-0.6, 0.1, 0.3, -0.2, 0.8, 0.4, -0.5, 0.2],
      [0.4, 0.2, -0.1, 0.7, -0.3, 0.5, 0.6, -0.4]
    ];
    this.b1 = [0.1, 0.0, -0.2, 0.0, 0.1, -0.1, 0.0, 0.2];

    // W2: 8×4
    this.W2 = [
      [0.3, 0.1, -0.2, 0.4],
      [-0.5, 0.6, 0.2, -0.1],
      [0.2, -0.3, 0.5, 0.0],
      [0.4, 0.1, -0.4, 0.3],
      [-0.1, 0.3, 0.2, 0.5],
      [0.5, -0.2, 0.1, -0.3],
      [0.0, 0.4, -0.5, 0.2],
      [0.3, 0.0, 0.4, -0.1]
    ];
    this.b2 = [0.0, 0.1, 0.0, -0.1];

    this.STEPS = [
      '输入 x (d=4)',
      'z₁ = x·W₁ + b₁  (d=8)',
      'h = ReLU(z₁)  (d=8)',
      'z₂ = h·W₂ + b₂  (d=4)',
      '输出 y (d=4)'
    ];
  }

  init(container) {
    this.container = container;
    this.currentStep = 0;
    this._compute();
    this.render();
    return this;
  }

  _dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
  _vecMatmul(v, M) {
    return M[0].map((_, j) => this._dot(v, M.map(r => r[j])));
  }
  _add(a, b) { return a.map((v, i) => v + b[i]); }
  _relu(v) { return v.map(x => Math.max(0, x)); }

  _compute() {
    this.z1 = this._add(this._vecMatmul(this.x, this.W1), this.b1);
    this.h = this._relu(this.z1);
    this.z2 = this._add(this._vecMatmul(this.h, this.W2), this.b2);
    this.y = this.z2;
  }

  reset() { this.currentStep = 0; this.isPlaying = false; clearTimeout(this.timer); this.render(); }
  play() { this.isPlaying = true; this._auto(); }
  pause() { this.isPlaying = false; clearTimeout(this.timer); }
  setSpeed(s) { this.speed = s; }
  goTo(i) { this.currentStep = Math.max(0, Math.min(i, this.STEPS.length - 1)); this.render(); }
  stepForward() { if (this.currentStep < this.STEPS.length - 1) { this.currentStep++; this.render(); } }
  stepBack() { if (this.currentStep > 0) { this.currentStep--; this.render(); } }

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

  _vecCells(vec, color, highlightIdx = -1) {
    return vec.map((v, i) => {
      const hl = i === highlightIdx;
      const bg = v === 0 ? '#111' : (v > 0 ? 'rgba(16,185,129,' + Math.min(0.15 + Math.abs(v) * 0.3, 0.5) + ')' : 'rgba(239,68,68,' + Math.min(0.15 + Math.abs(v) * 0.3, 0.5) + ')');
      return '<div style="min-width:52px;padding:6px 8px;margin:2px;background:' + bg + ';border:' + (hl ? '2px' : '1px') + ' solid ' + (hl ? '#F97316' : color) + ';border-radius:4px;color:' + color + ';font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:center">' + v.toFixed(3) + '</div>';
    }).join('');
  }

  render() {
    if (!this.container) return;
    const st = this.currentStep;
    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._ffnInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._ffnInstance.play()">▶ 播放</button>';

    // 每步的详细公式框
    let detailHTML = '';
    if (st === 0) {
      detailHTML = '<div class="formula-box">输入向量 x ∈ ℝ⁴ = [' + this.x.map(v => v.toFixed(1)).join(', ') + ']</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:8px">Position-wise FFN 对序列中每个位置的向量独立执行（共享权重）。</div>';
    } else if (st === 1) {
      const j = 0;
      const terms = this.x.map((xi, i) => xi.toFixed(1) + '×' + this.W1[i][j].toFixed(2));
      detailHTML = '<div class="formula-box">z₁ = x·W₁ + b₁   (1×4)·(4×8) + (8,) = (1×8)</div>' +
        '<div class="formula-box" style="font-size:0.78rem">z₁[0] = ' + terms.join(' + ') + ' + ' + this.b1[j].toFixed(2) + ' = <span style="color:#F59E0B">' + this.z1[0].toFixed(3) + '</span></div>' +
        '<div style="color:#888;font-size:0.8rem;margin-top:6px">d_ff=8 是 d_model=4 的 2 倍（原论文为 4 倍）。FFN 的"宽"使其可学习丰富的非线性变换。</div>';
    } else if (st === 2) {
      const neg = this.z1.filter(v => v < 0).length;
      detailHTML = '<div class="formula-box">h = ReLU(z₁) = max(0, z₁)</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">' + neg + ' / ' + this.z1.length + ' 维被 ReLU 置零。ReLU 引入非线性，是 FFN 能逼近任意函数的关键。</div>';
    } else if (st === 3) {
      detailHTML = '<div class="formula-box">z₂ = h·W₂ + b₂   (1×8)·(8×4) + (4,) = (1×4)</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">第二个线性层把 d_ff 投影回 d_model，输出与输入同形，便于残差连接。</div>';
    } else if (st === 4) {
      detailHTML = '<div class="formula-box">FFN(x) = max(0, xW₁+b₁)W₂+b₂</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">最终输出 y ∈ ℝ⁴，维度与输入一致。Transformer 中对每个位置独立应用同一个 FFN。</div>';
    }

    // 步骤进度条
    const stepBar = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff' : (i < st ? 'background:#1e3a5f;color:#60a5fa' : 'background:#1a1a1a;color:#666');
      return '<div onclick="window._ffnInstance.goTo(' + i + ')" style="padding:6px 10px;border:1px solid #333;border-radius:4px;font-size:0.75rem;cursor:pointer;' + active + '">' + (i + 1) + '. ' + s + '</div>';
    }).join('');

    // 数据流可视化
    const xRow = this._vecCells(this.x, '#3B82F6');
    const z1Row = st >= 1 ? this._vecCells(this.z1, '#F59E0B') : this.z1.map(() => '<div style="min-width:52px;padding:6px 8px;margin:2px;background:#0a0a0a;border:1px dashed #333;border-radius:4px;color:#444;font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:center">?</div>').join('');
    const hRow = st >= 2 ? this._vecCells(this.h, '#10B981') : this.h.map(() => '<div style="min-width:52px;padding:6px 8px;margin:2px;background:#0a0a0a;border:1px dashed #333;border-radius:4px;color:#444;font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:center">?</div>').join('');
    const yRow = st >= 3 ? this._vecCells(this.y, '#06B6D4') : this.y.map(() => '<div style="min-width:52px;padding:6px 8px;margin:2px;background:#0a0a0a;border:1px dashed #333;border-radius:4px;color:#444;font-family:JetBrains Mono,monospace;font-size:0.78rem;text-align:center">?</div>').join('');

    this.container.innerHTML =
      '<div class="ffn-viz">' +
        '<style>' +
          '.ffn-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.ffn-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem}' +
          '.ffn-viz .ctrl-btn:hover{background:#252525}' +
          '.ffn-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.ffn-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.ffn-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.ffn-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '@media(max-width:900px){.ffn-viz .two-col{grid-template-columns:1fr}}' +
        '</style>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">控制</div>' +
          playBtn +
          '<button class="ctrl-btn" onclick="window._ffnInstance.reset()">↻ 重置</button>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepBar + '</div>' +
        '</div>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">当前步骤: ' + this.STEPS[st] + '</div>' +
          detailHTML +
        '</div>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">数据流可视化</div>' +

          '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">' +
            '<span style="color:#3B82F6;min-width:80px;font-family:JetBrains Mono,monospace;font-size:0.82rem">x (1×4)</span>' +
            '<div style="display:flex">' + xRow + '</div>' +
          '</div>' +

          '<div style="text-align:center;color:#666;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ × W₁ (4×8) + b₁</div>' +

          '<div style="display:flex;align-items:center;gap:8px;margin:8px 0">' +
            '<span style="color:#F59E0B;min-width:80px;font-family:JetBrains Mono,monospace;font-size:0.82rem">z₁ (1×8)</span>' +
            '<div style="display:flex;flex-wrap:wrap">' + z1Row + '</div>' +
          '</div>' +

          '<div style="text-align:center;color:#666;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ ReLU(·)</div>' +

          '<div style="display:flex;align-items:center;gap:8px;margin:8px 0">' +
            '<span style="color:#10B981;min-width:80px;font-family:JetBrains Mono,monospace;font-size:0.82rem">h (1×8)</span>' +
            '<div style="display:flex;flex-wrap:wrap">' + hRow + '</div>' +
          '</div>' +

          '<div style="text-align:center;color:#666;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ × W₂ (8×4) + b₂</div>' +

          '<div style="display:flex;align-items:center;gap:8px;margin:8px 0">' +
            '<span style="color:#06B6D4;min-width:80px;font-family:JetBrains Mono,monospace;font-size:0.82rem">y (1×4)</span>' +
            '<div style="display:flex">' + yRow + '</div>' +
          '</div>' +
        '</div>' +

        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">维度变化</div>' +
            '<div class="formula-box">(seq_len, d_model) → (seq_len, d_ff) → (seq_len, d_model)</div>' +
            '<div class="formula-box">(3, 4) → (3, 8) → (3, 4)</div>' +
            '<div style="color:#888;font-size:0.82rem;margin-top:6px">本示例只演示单个 token 的前向过程。实际 Transformer 中 FFN 对 batch×seq_len 个位置并行、权重共享。</div>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">为何需要 FFN？</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
              '• 自注意力本质是<b>线性加权</b>，缺乏非线性表达力<br>' +
              '• FFN 通过 ReLU 注入非线性，让网络能表示复杂函数<br>' +
              '• "升维再降维"提供更大参数容量，存储世界知识<br>' +
              '• 每个 token 独立处理（Position-wise），不跨位置交互' +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>';

    window._ffnInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._ffnInstance === this) {
      try { delete window._ffnInstance; } catch (e) { window._ffnInstance = null; }
    }
    this.container = null;
  }
}

window.FFNVisualization = FFNVisualization;
