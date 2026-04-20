// Layer Normalization + Residual Connection 可视化
// y = LayerNorm(x + Sublayer(x))   <- Post-LN 版本
// y = x + Sublayer(LayerNorm(x))   <- Pre-LN 版本
class LayerNormVisualization {
  constructor() {
    this.container = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1;
    this.timer = null;

    // 输入向量 (d_model=4)，分布不均
    this.x = [2.0, -1.0, 3.5, 0.5];
    // 假设 sublayer 输出
    this.sublayerOut = [0.3, 0.8, -0.4, 0.6];
    // LayerNorm 参数
    this.gamma = [1.0, 1.0, 1.0, 1.0];
    this.beta = [0.0, 0.0, 0.0, 0.0];
    this.eps = 1e-5;

    this.STEPS = [
      '输入 x',
      '子层计算 Sublayer(x)',
      '残差相加 r = x + Sublayer(x)',
      '计算均值 μ',
      '计算方差 σ²',
      '归一化 x̂ = (r - μ) / √(σ²+ε)',
      '仿射变换 y = γ·x̂ + β'
    ];
  }

  init(container) {
    this.container = container;
    this.currentStep = 0;
    this._compute();
    this.render();
    return this;
  }

  _compute() {
    this.r = this.x.map((v, i) => v + this.sublayerOut[i]);
    this.mu = this.r.reduce((s, v) => s + v, 0) / this.r.length;
    const vari = this.r.reduce((s, v) => s + (v - this.mu) ** 2, 0) / this.r.length;
    this.variance = vari;
    this.std = Math.sqrt(vari + this.eps);
    this.xHat = this.r.map(v => (v - this.mu) / this.std);
    this.y = this.xHat.map((v, i) => this.gamma[i] * v + this.beta[i]);
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
      this.timer = setTimeout(() => this._auto(), 1500 / this.speed);
    } else {
      this.isPlaying = false;
      this.render();
    }
  }

  _vecRow(label, vec, color, visible = true) {
    const cells = vec.map(v => {
      if (!visible) return '<div style="min-width:64px;padding:8px;margin:2px;background:#0a0a0a;border:1px dashed #333;border-radius:4px;color:#444;font-family:JetBrains Mono,monospace;font-size:0.82rem;text-align:center">?</div>';
      const intensity = Math.min(Math.abs(v) * 0.25, 0.5);
      const bg = v === 0 ? '#111' : (v > 0 ? 'rgba(16,185,129,' + (0.15 + intensity) + ')' : 'rgba(239,68,68,' + (0.15 + intensity) + ')');
      return '<div style="min-width:64px;padding:8px;margin:2px;background:' + bg + ';border:1px solid ' + color + ';border-radius:4px;color:' + color + ';font-family:JetBrains Mono,monospace;font-size:0.82rem;text-align:center">' + v.toFixed(3) + '</div>';
    }).join('');
    return '<div style="display:flex;align-items:center;gap:8px;margin:6px 0">' +
      '<span style="color:' + color + ';min-width:110px;font-family:JetBrains Mono,monospace;font-size:0.82rem">' + label + '</span>' +
      '<div style="display:flex">' + cells + '</div>' +
      '</div>';
  }

  render() {
    if (!this.container) return;
    const st = this.currentStep;
    const playBtn = this.isPlaying
      ? '<button class="ctrl-btn active" onclick="window._lnInstance.pause()">⏸ 暂停</button>'
      : '<button class="ctrl-btn" onclick="window._lnInstance.play()">▶ 播放</button>';

    const stepBar = this.STEPS.map((s, i) => {
      const active = i === st ? 'background:#3B82F6;color:#fff' : (i < st ? 'background:#1e3a5f;color:#60a5fa' : 'background:#1a1a1a;color:#666');
      return '<div onclick="window._lnInstance.goTo(' + i + ')" style="padding:6px 10px;border:1px solid #333;border-radius:4px;font-size:0.72rem;cursor:pointer;white-space:nowrap;' + active + '">' + (i + 1) + '. ' + s + '</div>';
    }).join('');

    let detailHTML = '';
    if (st === 0) {
      detailHTML = '<div class="formula-box">输入 x = [' + this.x.map(v => v.toFixed(2)).join(', ') + ']</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">注意各维度数值悬殊（-1 到 3.5），若不做归一化，会让深层网络的梯度爆炸/消失。</div>';
    } else if (st === 1) {
      detailHTML = '<div class="formula-box">Sublayer(x) = [' + this.sublayerOut.map(v => v.toFixed(2)).join(', ')  + ']</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">Sublayer 可以是 Multi-Head Attention 或 FFN。这里用假设的输出演示。</div>';
    } else if (st === 2) {
      detailHTML = '<div class="formula-box">r = x + Sublayer(x)  &nbsp;(残差连接)</div>' +
        '<div class="formula-box" style="font-size:0.78rem">r = [' + this.r.map(v => v.toFixed(2)).join(', ') + ']</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">残差连接让梯度可以直接回传到浅层，解决深层网络训练难题（ResNet 思想）。</div>';
    } else if (st === 3) {
      const sum = this.r.reduce((a, b) => a + b, 0);
      detailHTML = '<div class="formula-box">μ = (1/d)·Σᵢ rᵢ = (' + this.r.map(v => v.toFixed(2)).join(' + ') + ') / 4 = <span style="color:#F97316">' + this.mu.toFixed(4) + '</span></div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">LayerNorm 在<b>单个样本的特征维度</b>上计算均值（不同于 BatchNorm 跨 batch）。</div>';
    } else if (st === 4) {
      detailHTML = '<div class="formula-box">σ² = (1/d)·Σᵢ (rᵢ - μ)² = <span style="color:#F97316">' + this.variance.toFixed(4) + '</span></div>' +
        '<div class="formula-box">√(σ² + ε) = <span style="color:#F97316">' + this.std.toFixed(4) + '</span></div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">ε = 1e-5 防止除零。</div>';
    } else if (st === 5) {
      detailHTML = '<div class="formula-box">x̂ᵢ = (rᵢ - μ) / √(σ²+ε)</div>' +
        '<div class="formula-box">x̂ = [' + this.xHat.map(v => v.toFixed(3)).join(', ') + ']</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">归一化后 E[x̂]=0, Var[x̂]=1。各维度数值被拉到同一尺度。</div>';
    } else if (st === 6) {
      detailHTML = '<div class="formula-box">y = γ ⊙ x̂ + β</div>' +
        '<div class="formula-box">当前 γ = [1,1,1,1], β = [0,0,0,0]，恒等变换</div>' +
        '<div style="color:#888;font-size:0.82rem;margin-top:6px">γ 和 β 是<b>可学习参数</b>，让模型可以恢复任何分布（不强制归一化到 N(0,1)）。</div>';
    }

    const row_x = this._vecRow('x (输入)', this.x, '#3B82F6');
    const row_sub = this._vecRow('Sublayer(x)', this.sublayerOut, '#8B5CF6', st >= 1);
    const row_r = this._vecRow('r = x + S(x)', this.r, '#F59E0B', st >= 2);
    const row_xhat = this._vecRow('x̂ (归一化)', this.xHat, '#10B981', st >= 5);
    const row_y = this._vecRow('y (输出)', this.y, '#06B6D4', st >= 6);

    // 统计信息
    const statsBefore = '<div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#F59E0B">μ_r=' + this.mu.toFixed(3) + '  σ²_r=' + this.variance.toFixed(3) + '</div>';
    const muHat = this.xHat.reduce((a, b) => a + b, 0) / this.xHat.length;
    const varHat = this.xHat.reduce((s, v) => s + (v - muHat) ** 2, 0) / this.xHat.length;
    const statsAfter = '<div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#10B981">μ_x̂=' + muHat.toFixed(3) + '  σ²_x̂=' + varHat.toFixed(3) + '</div>';

    this.container.innerHTML =
      '<div class="ln-viz">' +
        '<style>' +
          '.ln-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.ln-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem}' +
          '.ln-viz .ctrl-btn:hover{background:#252525}' +
          '.ln-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.ln-viz .formula-box{background:#111;border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#a5f3fc;margin:6px 0}' +
          '.ln-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.ln-viz .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}' +
          '@media(max-width:900px){.ln-viz .two-col{grid-template-columns:1fr}}' +
        '</style>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:8px">控制</div>' +
          playBtn +
          '<button class="ctrl-btn" onclick="window._lnInstance.reset()">↻ 重置</button>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">' + stepBar + '</div>' +
        '</div>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">当前步骤: ' + this.STEPS[st] + '</div>' +
          detailHTML +
        '</div>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">数据流</div>' +
          row_x +
          '<div style="text-align:center;color:#8B5CF6;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ Sublayer (MHA / FFN)</div>' +
          row_sub +
          '<div style="text-align:center;color:#F59E0B;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ 残差相加</div>' +
          row_r +
          (st >= 3 ? '<div style="text-align:center;margin:4px 0">' + statsBefore + '</div>' : '') +
          '<div style="text-align:center;color:#10B981;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ LayerNorm</div>' +
          row_xhat +
          (st >= 5 ? '<div style="text-align:center;margin:4px 0">' + statsAfter + '</div>' : '') +
          '<div style="text-align:center;color:#06B6D4;font-family:JetBrains Mono,monospace;font-size:0.8rem">↓ γ⊙x̂ + β</div>' +
          row_y +
        '</div>' +

        '<div class="two-col">' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">LayerNorm vs BatchNorm</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
              '• <b style="color:#10B981">LayerNorm</b>：在单个样本的<b>特征维度</b>做归一化<br>' +
              '  ↳ 不依赖 batch size，适合 NLP（变长序列）<br>' +
              '• <b style="color:#F59E0B">BatchNorm</b>：跨 batch 在同一特征通道归一化<br>' +
              '  ↳ batch 小时不稳定；推理需移动平均<br>' +
              '• Transformer 几乎只用 LayerNorm' +
            '</div>' +
          '</div>' +
          '<div class="section">' +
            '<div style="font-weight:600;margin-bottom:8px">残差连接的作用</div>' +
            '<div style="color:#aaa;font-size:0.85rem;line-height:1.7">' +
              '• 梯度直通：∂L/∂x 可绕过 Sublayer 直达浅层<br>' +
              '• 解决深网络退化：初始化时近似恒等映射<br>' +
              '• Post-LN: <code>LayerNorm(x + Sublayer(x))</code><br>' +
              '• Pre-LN: <code>x + Sublayer(LayerNorm(x))</code>（更稳定，GPT 使用）' +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>';

    window._lnInstance = this;
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._lnInstance === this) {
      try { delete window._lnInstance; } catch (e) { window._lnInstance = null; }
    }
    this.container = null;
  }
}

window.LayerNormVisualization = LayerNormVisualization;
