/**
 * Utils.color — 颜色、热力图、Canvas / D3 渲染辅助
 *
 * 颜色常量（Q/K/V/Score/Output/FFN）与 css/style.css 的 --x-color 变量保持对齐。
 * 模块优先用 CSS 变量，JS 中需要程序化赋色（如 heatmap）时再用这里。
 */

(function () {
    'use strict';

    const COLORS = {
        Q: '#3B82F6',
        K: '#8B5CF6',
        V: '#10B981',
        Score: '#F97316',
        Output: '#06B6D4',
        FFN: '#F59E0B',
        Positive: '#10B981',
        Negative: '#EF4444',
        Zero: '#374151',
        Background: '#1F2937',
        Text: '#F9FAFB'
    };

    function getHeatmapColor(value, min, max) {
        const range = (max - min) || 1;
        const normalized = (value - min) / range;
        if (value === 0) return COLORS.Zero;
        if (value < 0) {
            const intensity = Math.min(1, Math.abs(normalized));
            return `rgba(239, 68, 68, ${intensity})`;
        }
        const intensity = Math.min(1, normalized);
        return `rgba(16, 185, 129, ${intensity})`;
    }

    function clearCanvas(ctx, width, height) {
        ctx.fillStyle = COLORS.Background;
        ctx.fillRect(0, 0, width, height);
    }

    /**
     * D3 箭头 marker。
     * @param {d3.Selection} svg  - SVG 选择器
     * @param {string} id         - marker id（需唯一）
     * @param {string} color      - 填充色
     */
    function createArrowMarker(svg, id, color) {
        svg.append('defs')
            .append('marker')
            .attr('id', id)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 8)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', color);
    }

    /**
     * Canvas 上绘制一个矩阵（含值、边框、维度标注）。
     * 依赖 Utils.formatNumber / Utils.fmt，需在 format.js 之后加载。
     */
    function drawMatrix(ctx, matrix, x, y, cellSize, color, label = '') {
        const rows = matrix.length;
        const cols = matrix[0].length;

        if (label) {
            ctx.fillStyle = COLORS.Text;
            ctx.font = '14px JetBrains Mono';
            ctx.fillText(label, x, y - 10);
        }

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const value = matrix[i][j];
                const cellX = x + j * cellSize;
                const cellY = y + i * cellSize;

                ctx.fillStyle = getHeatmapColor(value, -1, 1);
                ctx.fillRect(cellX, cellY, cellSize - 2, cellSize - 2);

                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.strokeRect(cellX, cellY, cellSize - 2, cellSize - 2);

                ctx.fillStyle = COLORS.Text;
                ctx.font = '12px JetBrains Mono';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const fmtFn = (window.Utils && window.Utils.fmt) || ((v, d) => Number(v).toFixed(d));
                ctx.fillText(
                    fmtFn(value, 2),
                    cellX + (cellSize - 2) / 2,
                    cellY + (cellSize - 2) / 2
                );
            }
        }

        ctx.fillStyle = COLORS.Text;
        ctx.font = '12px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(`(${rows} × ${cols})`, x + cols * cellSize + 10, y + rows * cellSize / 2);
    }

    /**
     * KaTeX 公式渲染。KaTeX 还未加载时退化为纯文本显示。
     */
    function renderLatex(latex, container, displayMode = true) {
        if (typeof katex !== 'undefined') {
            katex.render(latex, container, { displayMode, throwOnError: false });
        } else {
            container.textContent = latex;
        }
    }

    window.Utils = Object.assign(window.Utils || {}, {
        COLORS,
        getHeatmapColor,
        clearCanvas,
        createArrowMarker,
        drawMatrix,
        renderLatex
    });
})();
