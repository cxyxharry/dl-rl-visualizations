/**
 * Utils.math — 共享数学工具
 *
 * 所有矩阵操作、激活、softmax、layer norm、位置编码等统一放在这里。
 * 模块请调用 `Utils.matmul(A, B)` 等，不要在模块内部再实现 `_matmul`。
 *
 * 约定：
 *  - 矩阵用 `number[][]`（行主序），向量用 `number[]`
 *  - 函数纯函数，不修改入参
 *  - 维度不匹配时 JS 引擎会自然报错（NaN 或越界），我们不做额外校验
 */

(function () {
    'use strict';

    // ============================================
    // 全局教学数值例子（全项目统一）
    // ============================================
    const GLOBAL_CONFIG = {
        d_model: 4,
        seq_len: 3,
        batch_size: 1,
        num_heads: 2,
        d_k: 4,
        d_v: 4,
        X: [
            [1.0, 0.5, 0.3, 0.2],
            [0.8, 1.0, 0.5, 0.1],
            [0.3, 0.7, 1.0, 0.4]
        ],
        WQ: [[1,0,0,0],[0,1,0,0],[1,1,0,0],[0,0,1,1]],
        WK: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        WV: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        Q_table: [[0.0, 0.0],[0.5, 0.3],[0.2, 0.6]],
        alpha: 0.1,
        gamma: 0.9
    };

    // ============================================
    // 矩阵运算
    // ============================================

    function matmul(A, B) {
        const m = A.length, n = A[0].length, p = B[0].length;
        const C = Array.from({ length: m }, () => new Array(p).fill(0));
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                let s = 0;
                for (let k = 0; k < n; k++) s += A[i][k] * B[k][j];
                C[i][j] = s;
            }
        }
        return C;
    }

    function transpose(A) {
        return A[0].map((_, j) => A.map(r => r[j]));
    }

    function scale(A, s) {
        return A.map(r => r.map(v => v / s));
    }

    function matAdd(A, B) {
        return A.map((row, i) => row.map((v, j) => v + B[i][j]));
    }

    // ============================================
    // 激活函数
    // ============================================

    function relu(x) { return Math.max(0, x); }
    function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    function tanh(x) { return Math.tanh(x); }
    function reluMatrix(M) { return M.map(r => r.map(v => Math.max(0, v))); }

    // ============================================
    // Softmax
    // ============================================

    function softmaxRow(row) {
        const mx = Math.max(...row);
        const ex = row.map(v => Math.exp(v - mx));
        const s = ex.reduce((a, b) => a + b, 0);
        return ex.map(v => v / s);
    }

    function softmax(M) {
        return M.map(softmaxRow);
    }

    // ============================================
    // Layer Normalization
    // ============================================

    function layerNorm(v, eps = 1e-5) {
        const m = v.reduce((s, x) => s + x, 0) / v.length;
        const va = v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length;
        const std = Math.sqrt(va + eps);
        return v.map(x => (x - m) / std);
    }

    function layerNormMatrix(M, eps = 1e-5) {
        return M.map(row => layerNorm(row, eps));
    }

    // ============================================
    // 向量 / 其他
    // ============================================

    function dotProduct(a, b) {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    function lerp(start, end, t) {
        return start + (end - start) * t;
    }

    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    // ============================================
    // 位置编码
    // ============================================

    function positionalEncoding(pos, i, d_model) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / d_model);
        return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }

    function generatePositionalEncoding(seq_len, d_model) {
        const PE = [];
        for (let pos = 0; pos < seq_len; pos++) {
            const row = [];
            for (let i = 0; i < d_model; i++) {
                row.push(positionalEncoding(pos, i, d_model));
            }
            PE.push(row);
        }
        return PE;
    }

    // ============================================
    // 导出到 window.Utils（聚合在三个 utils 文件之间）
    // ============================================

    window.Utils = Object.assign(window.Utils || {}, {
        // 常量
        GLOBAL_CONFIG,

        // 推荐新名字
        matmul,
        transpose,
        scale,
        matAdd,
        softmaxRow,
        softmax,
        layerNorm,
        layerNormMatrix,
        reluMatrix,
        dotProduct,
        relu,
        sigmoid,
        tanh,
        lerp,
        easeInOutCubic,
        positionalEncoding,
        generatePositionalEncoding,

        // 兼容原 utils.js 的长名（此前已经对外暴露过）
        matrixMultiply: matmul,
        matrixTranspose: transpose,
        matrixScale: scale,
        matrixAdd: matAdd
    });
})();
