/**
 * Utils.format — 数值与字符串格式化
 *
 * 三种风格对应现有模块的需求：
 *  - fmt(v, d)        ：最常用。`|v|<1e-9` 视为 0，NaN/Infinity 视为 0，保留 d 位小数
 *  - formatNumber(v,d)：最朴素。直接 `v.toFixed(d)`，不做任何清理
 *  - fmtSigned(v, d)  ：定宽。正数前补一个空格，负数保留负号，用于对齐表格列
 */

(function () {
    'use strict';

    function fmt(v, d) {
        const decimals = (d == null) ? 2 : d;
        const safe = (!Number.isFinite(v) || Math.abs(v) < 1e-9) ? 0 : v;
        return safe.toFixed(decimals);
    }

    function formatNumber(v, d) {
        const decimals = (d == null) ? 3 : d;
        return Number(v).toFixed(decimals);
    }

    function fmtSigned(v, d) {
        const decimals = (d == null) ? 3 : d;
        const n = Number(v);
        return (n >= 0 ? ' ' : '') + n.toFixed(decimals);
    }

    window.Utils = Object.assign(window.Utils || {}, {
        fmt,
        formatNumber,
        fmtSigned
    });
})();
