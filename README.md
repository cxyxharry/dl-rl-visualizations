# DL/RL Visualizations

Interactive visualizations of Deep Learning & Reinforcement Learning algorithms — built for teaching, self-study, and live demos.

**[→ Live Demo](http://localhost:8000)**

> Note: Since this is a static site using dynamic module loading, it must be served over HTTP. See [Getting Started](#getting-started) below.

---

## What is this?

A pure frontend visualization platform covering three core areas:

- **🔬 Transformer Series** — Self-Attention, Multi-Head Attention, Positional Encoding, FFN, LayerNorm, Encoder & Decoder
- **🧠 Neural Network Basics** — Forward/backward pass, activation functions
- **🎮 Reinforcement Learning** — MDP, Q-Learning, Policy Gradient, DQN

Each module gives you **formula + concrete numbers (e.g. d_model=4, seq_len=3)** instead of abstract diagrams — and lets you step through the computation one tick at a time. Perfect for classroom walkthroughs and screen recordings.

---

## ✨ Features

- **Step-by-step playback** — Step forward / backward through every computation
- **Concrete numerical examples** — Every formula is instantiated with real values, not symbolic notation alone
- **Dark theme** — Optimized for presentations and long study sessions
- **Pure frontend** — No build tools, no dependencies to install, no framework
- **Keyboard-first** — Space to play/pause, ← → to step, R to reset, Esc to return home

---

## 📚 Modules

### Transformer Series
| Module | Title | What's Visualized |
|--------|-------|-------------------|
| `attention` | Self-Attention | Q/K/V → Score → Scale → Softmax → Output |
| `multi-head-attention` | Multi-Head Attention | Multi-head parallel → Concat → Project |
| `positional-encoding` | Positional Encoding | sin/cos PE + heatmap |
| `ffn` | Feed-Forward Network | Linear → ReLU → Linear |
| `layer-norm` | Layer Norm & Residuals | LayerNorm + residual connection |
| `transformer-encoder` | Transformer Encoder | MHA → Add&Norm → FFN → Add&Norm |
| `transformer-decoder` | Transformer Decoder | Masked MHA → Cross-Attention → FFN |

### Neural Network Basics
| Module | Title | What's Visualized |
|--------|-------|-------------------|
| `forward-backward` | Forward & Backward Pass | 2-layer net forward + gradients + chain rule |
| `activations` | Activation Functions | ReLU / Sigmoid / Tanh / GELU / LeakyReLU |

### Reinforcement Learning
| Module | Title | What's Visualized |
|--------|-------|-------------------|
| `rl-mdp` | MDP State Graph | States / actions / rewards / transition probabilities |
| `q-learning` | Q-Learning | Q-table update + ε-greedy policy |
| `policy-gradient` | Policy Gradient | REINFORCE + trajectory sampling |
| `dqn` | DQN | Experience replay + target network |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Structure | HTML5 (single-page app, hash routing) |
| Styling | CSS3 with CSS variables, dark theme |
| Logic | Vanilla JS (ES6+, no framework, no bundler) |
| Charts | D3.js v7 (via CDN) |
| Math | KaTeX (via CDN) |
| Fonts | Google Fonts — Inter + JetBrains Mono |

All dependencies are served from CDN. No `package.json`, no `node_modules`, no build output.

---

## 🚀 Getting Started

```bash
# Python 3
python3 -m http.server 8000

# Node.js (npx, no global install needed)
npx http-server -p 8000 .

# VS Code — right-click index.html → "Open with Live Server"
```

Then open **http://localhost:8000** in your browser.

### Navigation

- Click any module card on the home page
- Use the left sidebar
- Or go directly via URL hash: `#attention`, `#q-learning`, etc.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `→` | Step forward |
| `←` | Step backward |
| `R` | Reset |
| `Esc` | Back to home |

---

## 📁 Project Structure

```
dl-rl-visualizations/
├── index.html              # Main entry point
├── css/
│   ├── style.css           # Global styles + dark theme
│   └── components.css      # Component styles
├── js/
│   ├── main.js             # Router, module loader, global controls
│   ├── utils/              # Shared utilities (matrix ops, activation, etc.)
│   └── visualizations/     # One file per module
├── references/
│   └── math-notation.md   # Math symbol cheat sheet
└── docs/
    ├── module-spec.md      # Module development guide
    └── module-status.md    # Module health tracker
```

---

## 🎯 Roadmap

| Phase | Focus |
|-------|-------|
| Phase 1 | Architecture cleanup — unified module lifecycle, consistent controls |
| Phase 2 | UX improvements — responsive layout, accessibility, search |
| Phase 3 | Content polish — consistent token examples, shape annotations |
| Phase 4 | New modules — training loops, modern LLMs, advanced RL |
| Phase 5 | Distribution — packaging, CI, contribution guide |