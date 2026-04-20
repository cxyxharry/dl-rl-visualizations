// 激活函数对比可视化：ReLU / Sigmoid / Tanh / GELU / LeakyReLU
class ActivationsVisualization {
  constructor() {
    this.container = null;
    this.hoverX = 1.0;           // 当前悬停的 x 值
    this.showGrad = true;        // 是否显示导数曲线
    this.enabled = { relu: true, sigmoid: true, tanh: true, gelu: true, leaky: false };
  }

  init(container) {
    this.container = container;
    this.render();
    this._bindSvg();
    return this;
  }

  reset() { this.hoverX = 1.0; this.showGrad = true; this.enabled = { relu: true, sigmoid: true, tanh: true, gelu: true, leaky: false }; this.render(); this._bindSvg(); }
  play() {}
  pause() {}
  setSpeed() {}
  stepForward() { this.hoverX = Math.min(5, this.hoverX + 0.5); this.render(); this._bindSvg(); }
  stepBack() { this.hoverX = Math.max(-5, this.hoverX - 0.5); this.render(); this._bindSvg(); }
  goTo() {}

  toggle(key) { this.enabled[key] = !this.enabled[key]; this.render(); this._bindSvg(); }
  toggleGrad() { this.showGrad = !this.showGrad; this.render(); this._bindSvg(); }

  // 激活函数及其导数
  _fns() {
    return {
      relu: {
        name: 'ReLU',
        color: '#3B82F6',
        f: x => Math.max(0, x),
        df: x => x > 0 ? 1 : 0,
        formula: 'f(x) = max(0, x)',
        dformula: "f'(x) = 1 if x>0 else 0",
        pros: '计算快、稀疏激活、缓解梯度消失',
        cons: 'x<0 时梯度为 0（dying ReLU）',
        use: 'CNN、FFN 默认选择'
      },
      sigmoid: {
        name: 'Sigmoid',
        color: '#8B5CF6',
        f: x => 1 / (1 + Math.exp(-x)),
        df: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
        formula: 'f(x) = 1 / (1 + e^(-x))',
        dformula: "f'(x) = f(x)(1 - f(x))",
        pros: '输出 (0,1)，可解释为概率',
        cons: '饱和区梯度≈0、非零均值',
        use: '二分类输出层、门控（LSTM/GRU）'
      },
      tanh: {
        name: 'Tanh',
        color: '#10B981',
        f: x => Math.tanh(x),
        df: x => 1 - Math.tanh(x) ** 2,
        formula: 'f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)',
        dformula: "f'(x) = 1 - tanh²(x)",
        pros: '输出 (-1,1)、零均值、收敛更快',
        cons: '仍存在饱和梯度消失',
        use: 'RNN 隐藏层、早期 DNN'
      },
      gelu: {
        name: 'GELU',
        color: '#F97316',
        f: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
        df: x => {
          const c = Math.sqrt(2 / Math.PI);
          const u = c * (x + 0.044715 * x ** 3);
          const tanh_u = Math.tanh(u);
          const sech2 = 1 - tanh_u ** 2;
          const du_dx = c * (1 + 3 * 0.044715 * x ** 2);
          return 0.5 * (1 + tanh_u) + 0.5 * x * sech2 * du_dx;
        },
        formula: 'f(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³)))',
        dformula: '近似光滑的 x·Φ(x)',
        pros: '平滑、非单调、实证效果好',
        cons: '计算稍贵',
        use: 'Transformer（BERT/GPT）FFN 内部'
      },
      leaky: {
        name: 'LeakyReLU',
        color: '#EC4899',
        f: x => x > 0 ? x : 0.1 * x,
        df: x => x > 0 ? 1 : 0.1,
        formula: 'f(x) = x if x>0 else αx (α=0.1)',
        dformula: "f'(x) = 1 if x>0 else α",
        pros: '解决 dying ReLU',
        cons: '多一个超参；边际收益有限',
        use: 'GAN 判别器、某些 CV 模型'
      }
    };
  }

  _plot(w, h, xMin, xMax, yMin, yMax) {
    const pad = 40;
    const sx = x => pad + (x - xMin) / (xMax - xMin) * (w - 2 * pad);
    const sy = y => h - pad - (y - yMin) / (yMax - yMin) * (h - 2 * pad);
    return { sx, sy, pad };
  }

  _curve(fn, sx, sy, xMin, xMax, step = 0.05) {
    let d = '';
    let first = true;
    for (let x = xMin; x <= xMax + 1e-6; x += step) {
      const yv = fn(x);
      if (!isFinite(yv)) { first = true; continue; }
      const cmd = first ? 'M' : 'L';
      d += cmd + sx(x).toFixed(2) + ',' + sy(yv).toFixed(2) + ' ';
      first = false;
    }
    return d;
  }

  render() {
    if (!this.container) return;
    const fns = this._fns();
    const xMin = -5, xMax = 5, yMin = -1.5, yMax = 2.0;
    const w = 720, h = 360;
    const { sx, sy, pad } = this._plot(w, h, xMin, xMax, yMin, yMax);

    // 网格
    let grid = '';
    for (let gx = Math.ceil(xMin); gx <= xMax; gx++) {
      grid += '<line x1="' + sx(gx) + '" y1="' + pad + '" x2="' + sx(gx) + '" y2="' + (h - pad) + '" stroke="#1f1f1f" stroke-width="1"/>';
    }
    for (let gy = Math.ceil(yMin); gy <= yMax; gy++) {
      grid += '<line x1="' + pad + '" y1="' + sy(gy) + '" x2="' + (w - pad) + '" y2="' + sy(gy) + '" stroke="#1f1f1f" stroke-width="1"/>';
    }
    // 轴
    const axis = '<line x1="' + pad + '" y1="' + sy(0) + '" x2="' + (w - pad) + '" y2="' + sy(0) + '" stroke="#444" stroke-width="1.5"/>' +
                 '<line x1="' + sx(0) + '" y1="' + pad + '" x2="' + sx(0) + '" y2="' + (h - pad) + '" stroke="#444" stroke-width="1.5"/>';

    // 坐标轴标签
    let ticks = '';
    for (let gx = Math.ceil(xMin); gx <= xMax; gx++) {
      if (gx === 0) continue;
      ticks += '<text x="' + sx(gx) + '" y="' + (sy(0) + 14) + '" fill="#666" font-size="11" text-anchor="middle" font-family="JetBrains Mono">' + gx + '</text>';
    }
    for (let gy = Math.ceil(yMin); gy <= yMax; gy++) {
      if (gy === 0) continue;
      ticks += '<text x="' + (sx(0) - 6) + '" y="' + (sy(gy) + 4) + '" fill="#666" font-size="11" text-anchor="end" font-family="JetBrains Mono">' + gy + '</text>';
    }

    // 曲线
    let curves = '';
    Object.keys(fns).forEach(k => {
      if (!this.enabled[k]) return;
      const fn = fns[k];
      // f(x) 主曲线
      curves += '<path d="' + this._curve(fn.f, sx, sy, xMin, xMax) + '" fill="none" stroke="' + fn.color + '" stroke-width="2"/>';
      // f'(x) 虚线
      if (this.showGrad) {
        curves += '<path d="' + this._curve(fn.df, sx, sy, xMin, xMax) + '" fill="none" stroke="' + fn.color + '" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.6"/>';
      }
    });

    // 悬停线与点
    const hX = this.hoverX;
    let hoverLine = '<line x1="' + sx(hX) + '" y1="' + pad + '" x2="' + sx(hX) + '" y2="' + (h - pad) + '" stroke="#F97316" stroke-width="1" stroke-dasharray="3,3"/>';
    let hoverDots = '';
    let hoverTable = '';
    Object.keys(fns).forEach(k => {
      if (!this.enabled[k]) return;
      const fn = fns[k];
      const y = fn.f(hX);
      const dy = fn.df(hX);
      hoverDots += '<circle cx="' + sx(hX) + '" cy="' + sy(y) + '" r="4" fill="' + fn.color + '" stroke="#fff" stroke-width="1"/>';
      hoverTable += '<tr>' +
        '<td style="padding:4px 8px;color:' + fn.color + ';font-weight:600">' + fn.name + '</td>' +
        '<td style="padding:4px 8px;font-family:JetBrains Mono,monospace;color:#a5f3fc">' + y.toFixed(4) + '</td>' +
        '<td style="padding:4px 8px;font-family:JetBrains Mono,monospace;color:#fbbf24">' + dy.toFixed(4) + '</td>' +
      '</tr>';
    });

    // 图例 & 切换
    let legend = '';
    Object.keys(fns).forEach(k => {
      const fn = fns[k];
      const on = this.enabled[k];
      legend += '<div onclick="window._actInstance.toggle(\'' + k + '\')" style="display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border:1px solid ' + (on ? fn.color : '#333') + ';border-radius:6px;cursor:pointer;font-size:0.82rem;background:' + (on ? 'rgba(0,0,0,0.4)' : '#0a0a0a') + ';color:' + (on ? fn.color : '#555') + ';margin-right:8px;margin-bottom:6px">' +
        '<span style="display:inline-block;width:18px;height:2px;background:' + fn.color + ';opacity:' + (on ? 1 : 0.3) + '"></span>' +
        fn.name +
      '</div>';
    });

    // 卡片
    let cards = '';
    Object.keys(fns).forEach(k => {
      if (!this.enabled[k]) return;
      const fn = fns[k];
      cards += '<div class="section" style="border-left:3px solid ' + fn.color + '">' +
        '<div style="font-weight:600;color:' + fn.color + ';margin-bottom:6px">' + fn.name + '</div>' +
        '<div class="formula-box">' + fn.formula + '</div>' +
        '<div class="formula-box" style="color:#fbbf24">' + fn.dformula + '</div>' +
        '<div style="font-size:0.82rem;color:#aaa;line-height:1.7;margin-top:6px">' +
          '<div>✅ <b>优点</b>：' + fn.pros + '</div>' +
          '<div>⚠️ <b>缺点</b>：' + fn.cons + '</div>' +
          '<div>🎯 <b>典型场景</b>：' + fn.use + '</div>' +
        '</div>' +
      '</div>';
    });

    this.container.innerHTML =
      '<div class="act-viz">' +
        '<style>' +
          '.act-viz{font-family:Inter,sans-serif;color:#e5e5e5}' +
          '.act-viz .ctrl-btn{background:#1a1a1a;border:1px solid #333;color:#e5e5e5;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:8px;font-size:0.85rem}' +
          '.act-viz .ctrl-btn:hover{background:#252525}' +
          '.act-viz .ctrl-btn.active{background:#3B82F6;border-color:#3B82F6;color:#fff}' +
          '.act-viz .formula-box{background:#111;border-radius:6px;padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#a5f3fc;margin:4px 0}' +
          '.act-viz .section{background:#1a1a1a;border-radius:8px;padding:14px;border:1px solid #333;margin-bottom:14px}' +
          '.act-viz .grid-2{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}' +
        '</style>' +

        '<div class="section">' +
          '<div style="font-weight:600;margin-bottom:10px">激活函数（实线）与导数（虚线）对比</div>' +
          '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">' + legend + '</div>' +
          '<button class="ctrl-btn ' + (this.showGrad ? 'active' : '') + '" onclick="window._actInstance.toggleGrad()">显示导数 ' + (this.showGrad ? '✓' : '✗') + '</button>' +
          '<button class="ctrl-btn" onclick="window._actInstance.reset()">↻ 重置</button>' +
          '<div style="color:#888;font-size:0.8rem;margin-top:8px">💡 移动鼠标到图表上查看任一 x 值的函数值和梯度</div>' +
        '</div>' +

        '<div class="section">' +
          '<svg id="act-svg" viewBox="0 0 ' + w + ' ' + h + '" style="width:100%;max-width:' + w + 'px;display:block;margin:0 auto;background:#0a0a0a;border-radius:6px;cursor:crosshair">' +
            grid + axis + ticks + curves + hoverLine + hoverDots +
            '<text x="' + (w - pad + 4) + '" y="' + (sy(0) + 4) + '" fill="#666" font-size="11" font-family="JetBrains Mono">x</text>' +
            '<text x="' + (sx(0) + 4) + '" y="' + (pad - 4) + '" fill="#666" font-size="11" font-family="JetBrains Mono">y</text>' +
          '</svg>' +

          '<div style="margin-top:12px;text-align:center">' +
            '<div style="display:inline-block;background:#0a0a0a;border:1px solid #333;border-radius:8px;padding:10px 16px">' +
              '<div style="color:#888;font-size:0.78rem;margin-bottom:6px">当前 x = <span style="color:#F97316;font-family:JetBrains Mono,monospace;font-weight:600">' + hX.toFixed(3) + '</span></div>' +
              '<table style="border-collapse:collapse;font-size:0.82rem;margin:0 auto">' +
                '<thead><tr>' +
                  '<th style="padding:4px 8px;color:#666;font-weight:500">函数</th>' +
                  '<th style="padding:4px 8px;color:#666;font-weight:500">f(x)</th>' +
                  '<th style="padding:4px 8px;color:#666;font-weight:500">f\'(x)</th>' +
                '</tr></thead><tbody>' + hoverTable + '</tbody>' +
              '</table>' +
            '</div>' +
          '</div>' +
        '</div>' +

        '<div class="grid-2">' + cards + '</div>' +
      '</div>';

    window._actInstance = this;
  }

  _bindSvg() {
    const svg = document.getElementById('act-svg');
    if (!svg) return;
    const self = this;
    const xMin = -5, xMax = 5;
    const w = 720, pad = 40;
    svg.addEventListener('mousemove', e => {
      const rect = svg.getBoundingClientRect();
      const px = (e.clientX - rect.left) * (w / rect.width);
      const x = xMin + (px - pad) / (w - 2 * pad) * (xMax - xMin);
      self.hoverX = Math.max(xMin, Math.min(xMax, x));
      self.render();
      self._bindSvg();
    });
  }

  cleanup() {
    this.isPlaying = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (typeof window !== 'undefined' && window._actInstance === this) {
      try { delete window._actInstance; } catch (e) { window._actInstance = null; }
    }
    this.container = null;
  }
}

window.ActivationsVisualization = ActivationsVisualization;
