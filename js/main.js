/**
 * main.js - 主入口文件
 * 负责路由、导航、模块加载和动画控制
 */

// ============================================
// 全局状态管理
// ============================================

const AppState = {
    currentModule: null,
    isPlaying: false,
    currentStep: 0,
    totalSteps: 0,
    speed: 1.0,
    animationId: null,
    moduleInstance: null,
    syncTimer: null  // 控制栏状态轮询 interval id
};

/**
 * 已适配全局控制栏的模块白名单。
 * 这些模块的 DOM 容器会被打上 `.uses-global-controls` 类，
 * 配合 CSS 规则隐藏其内嵌的 play/pause/step/reset/speed 控件，
 * 避免与全局控制栏重复。未在此名单中的模块暂时保留自己的内嵌控件，
 * 但同一个实例方法（play/pause/stepForward/stepBack/reset）会被双方复用，
 * 点击内嵌按钮或全局按钮效果一致，不会冲突。
 */
const ADAPTED_MODULES = new Set([
    'attention',
    'multi-head-attention',
    'transformer-encoder',
    'forward-backward'
]);

// ============================================
// DOM 元素引用
// ============================================

const DOM = {
    sidebar: null,
    sidebarToggle: null,
    mainContent: null,
    homeView: null,
    moduleView: null,
    moduleTitle: null,
    moduleContent: null,
    controls: null,
    playBtn: null,
    pauseBtn: null,
    stepBtn: null,
    resetBtn: null,
    backBtn: null,
    speedSelect: null,
    stepIndicator: null,
    moduleCards: null
};

// ============================================
// 初始化
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initDOM();
    initEventListeners();
    initRouter();
    console.log('App initialized');
});

/**
 * 初始化 DOM 元素引用
 */
function initDOM() {
    DOM.sidebar = document.getElementById('sidebar');
    DOM.sidebarToggle = document.getElementById('sidebar-toggle');
    DOM.mainContent = document.getElementById('main-content');
    DOM.homeView = document.getElementById('home-view');
    DOM.moduleView = document.getElementById('module-view');
    DOM.moduleTitle = document.getElementById('module-title');
    DOM.moduleContent = document.getElementById('module-content');
    DOM.controls = document.getElementById('controls');
    DOM.playBtn = document.getElementById('play-btn');
    DOM.pauseBtn = document.getElementById('pause-btn');
    DOM.stepBtn = document.getElementById('step-btn');
    DOM.resetBtn = document.getElementById('reset-btn');
    DOM.backBtn = document.getElementById('back-btn');
    DOM.speedSelect = document.getElementById('speed-select');
    DOM.stepIndicator = document.getElementById('step-indicator');
    DOM.moduleCards = document.querySelectorAll('.module-card');

    // 首页筛选/搜索相关
    DOM.searchInput = document.getElementById('module-search');
    DOM.clearFilters = document.getElementById('clear-filters');
    DOM.emptyState = document.getElementById('empty-state');
    DOM.emptyClear = document.getElementById('empty-clear');
    DOM.filterStatus = document.getElementById('filter-status');
    DOM.filterCategoryChips = document.querySelectorAll('.filter-chip[data-filter-cat]');
    DOM.filterDifficultyChips = document.querySelectorAll('.filter-chip[data-filter-diff]');
    DOM.moduleSections = document.querySelectorAll('.module-section');
    DOM.pathStepLinks = document.querySelectorAll('.path-steps a[data-module]');
}

/**
 * 初始化事件监听器
 */
function initEventListeners() {
    // 侧边栏折叠
    if (DOM.sidebarToggle) {
        DOM.sidebarToggle.addEventListener('click', toggleSidebar);
    }

    // 模块卡片点击：只改 hash，视图切换由 hashchange 驱动
    DOM.moduleCards.forEach(card => {
        card.addEventListener('click', (e) => {
            e.preventDefault();
            const moduleName = card.dataset.module;
            navigateTo(moduleName);
        });
    });

    // 侧边栏导航链接：同上
    const navLinks = document.querySelectorAll('.sidebar a[data-module]');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const moduleName = link.dataset.module;
            navigateTo(moduleName);
        });
    });

    // 学习路径步骤链接：和侧边栏链接一致
    if (DOM.pathStepLinks) {
        DOM.pathStepLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                navigateTo(link.dataset.module);
            });
        });
    }

    // 首页筛选与搜索
    initHomeFilters();

    // 返回按钮：回首页
    if (DOM.backBtn) {
        DOM.backBtn.addEventListener('click', () => {
            navigateTo('');
        });
    }

    // 控制按钮
    if (DOM.playBtn) {
        DOM.playBtn.addEventListener('click', play);
    }
    if (DOM.pauseBtn) {
        DOM.pauseBtn.addEventListener('click', pause);
    }
    if (DOM.stepBtn) {
        DOM.stepBtn.addEventListener('click', step);
    }
    if (DOM.resetBtn) {
        DOM.resetBtn.addEventListener('click', reset);
    }

    // 速度控制：同步到 AppState 与当前模块实例
    if (DOM.speedSelect) {
        DOM.speedSelect.addEventListener('change', (e) => {
            const s = parseFloat(e.target.value) || 1;
            AppState.speed = s;
            const inst = AppState.moduleInstance;
            if (inst && typeof inst.setSpeed === 'function') {
                try { inst.setSpeed(s); } catch (err) { console.error('setSpeed failed:', err); }
            }
        });
    }

    // 键盘快捷键
    document.addEventListener('keydown', handleKeyboard);
}

/**
 * 已知模块白名单（与 getModuleInfo 保持同步）
 * 用于识别合法的 hash 路由，未知 hash 会被 replaceState 清掉并回到首页。
 */
const KNOWN_MODULES = new Set([
    'attention',
    'multi-head-attention',
    'positional-encoding',
    'ffn',
    'layer-norm',
    'transformer-encoder',
    'transformer-decoder',
    'forward-backward',
    'activations',
    'rl-mdp',
    'q-learning',
    'policy-gradient',
    'dqn'
]);

/**
 * 初始化路由
 * 策略：纯 hash 路由
 *  - URL 的 hash 是「想看哪个模块」的唯一数据源
 *  - 用户动作（点卡片/侧边栏/返回/Esc）只改 hash，不直接切换视图
 *  - hashchange 事件是视图切换的唯一入口
 *  - 浏览器前进/后退天然触发 hashchange，无需额外 popstate 监听
 */
function initRouter() {
    applyRoute();
    window.addEventListener('hashchange', applyRoute);
}

/**
 * 根据当前 URL hash 渲染对应视图。
 * 唯一会直接调用 loadModule() 或 showHomeView() 的地方。
 */
function applyRoute() {
    const hash = (window.location.hash || '').replace(/^#\/?/, '');

    if (!hash) {
        showHomeView();
        return;
    }

    if (!KNOWN_MODULES.has(hash)) {
        // 未知 hash：静默清掉，不产生历史记录
        console.warn(`Unknown route: #${hash}, falling back to home`);
        const cleanUrl = window.location.pathname + window.location.search;
        window.history.replaceState(null, '', cleanUrl);
        showHomeView();
        return;
    }

    loadModule(hash);
}

/**
 * 导航到指定模块（或回到首页）。
 * 所有用户触发的导航都应走此函数，它只负责改 URL，视图切换由 hashchange 驱动。
 * @param {string} moduleName - 模块名，空串/null/undefined 表示回首页
 */
function navigateTo(moduleName) {
    if (moduleName) {
        const desired = `#${moduleName}`;
        if (window.location.hash === desired) {
            // 幂等：已在目标模块，不改 URL，不重新渲染
            return;
        }
        window.location.hash = desired; // 触发 hashchange → applyRoute
    } else {
        if (!window.location.hash) {
            // 已经在首页，幂等
            return;
        }
        // 回首页：pushState 清掉 hash，保留干净的 URL（/ 而不是 /#）
        // pushState 不会触发 hashchange，需要手动调 applyRoute
        const cleanUrl = window.location.pathname + window.location.search;
        window.history.pushState(null, '', cleanUrl);
        applyRoute();
    }
}

// ============================================
// 首页：筛选与搜索
// ============================================
//
// 状态全部在这个局部对象里。URL 不参与（深链接本轮只做模块入口，
// 后续可以考虑支持 ?cat=transformer 这类 query，但不在本轮范围）。

const HomeFilter = {
    category: 'all',
    difficulty: 'all',
    query: ''
};

function initHomeFilters() {
    if (DOM.searchInput) {
        DOM.searchInput.addEventListener('input', () => {
            HomeFilter.query = DOM.searchInput.value.trim().toLowerCase();
            applyHomeFilters();
        });
    }

    if (DOM.filterCategoryChips) {
        DOM.filterCategoryChips.forEach(chip => {
            chip.addEventListener('click', () => {
                DOM.filterCategoryChips.forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                HomeFilter.category = chip.dataset.filterCat;
                applyHomeFilters();
            });
        });
    }

    if (DOM.filterDifficultyChips) {
        DOM.filterDifficultyChips.forEach(chip => {
            chip.addEventListener('click', () => {
                DOM.filterDifficultyChips.forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                HomeFilter.difficulty = chip.dataset.filterDiff;
                applyHomeFilters();
            });
        });
    }

    if (DOM.clearFilters) DOM.clearFilters.addEventListener('click', resetHomeFilters);
    if (DOM.emptyClear)   DOM.emptyClear.addEventListener('click', resetHomeFilters);

    applyHomeFilters(); // 初次渲染一下 status 行
}

function resetHomeFilters() {
    HomeFilter.category = 'all';
    HomeFilter.difficulty = 'all';
    HomeFilter.query = '';
    if (DOM.searchInput) DOM.searchInput.value = '';
    if (DOM.filterCategoryChips) {
        DOM.filterCategoryChips.forEach(c => c.classList.toggle('active', c.dataset.filterCat === 'all'));
    }
    if (DOM.filterDifficultyChips) {
        DOM.filterDifficultyChips.forEach(c => c.classList.toggle('active', c.dataset.filterDiff === 'all'));
    }
    applyHomeFilters();
}

/**
 * 根据当前筛选状态切换每张卡片的 .hidden；
 * 若某个分类下所有卡片都被筛掉，整个 .module-section 也隐藏；
 * 命中 0 条时显示 .empty-state 并更新 aria-live 的状态文本。
 */
function applyHomeFilters() {
    if (!DOM.moduleCards) return;
    const q = HomeFilter.query;
    const cat = HomeFilter.category;
    const diff = HomeFilter.difficulty;

    let totalVisible = 0;

    DOM.moduleCards.forEach(card => {
        const cardCat = card.dataset.category || '';
        const cardDiff = card.dataset.difficulty || '';
        const text = card.textContent.toLowerCase() + ' ' + (card.dataset.module || '').toLowerCase();

        const matchCat = cat === 'all' || cardCat === cat;
        const matchDiff = diff === 'all' || cardDiff === diff;
        const matchQuery = !q || text.includes(q);

        const visible = matchCat && matchDiff && matchQuery;
        card.classList.toggle('hidden', !visible);
        if (visible) totalVisible++;
    });

    // 整个分类都没卡片时，把章节头也隐藏
    if (DOM.moduleSections) {
        DOM.moduleSections.forEach(section => {
            const visibleCards = section.querySelectorAll('.module-card:not(.hidden)');
            section.classList.toggle('hidden', visibleCards.length === 0);
        });
    }

    // 空态与状态文案
    if (DOM.emptyState) DOM.emptyState.hidden = totalVisible > 0;
    if (DOM.filterStatus) {
        const hasFilter = q || cat !== 'all' || diff !== 'all';
        DOM.filterStatus.textContent = hasFilter
            ? `显示 ${totalVisible} / ${DOM.moduleCards.length} 个模块`
            : '';
    }
}

// ============================================
// 视图切换
// ============================================

/**
 * 显示主页视图（纯渲染，不改 URL）
 * URL 的同步由 navigateTo / applyRoute 负责。
 */
function showHomeView() {
    if (DOM.homeView) DOM.homeView.style.display = 'block';
    if (DOM.moduleView) DOM.moduleView.style.display = 'none';
    if (DOM.controls) DOM.controls.style.display = 'none';

    // 停止状态轮询 + 清理当前模块
    stopControlsSync();
    cleanupModule();

    // 清除侧边栏激活态
    document.querySelectorAll('.sidebar a[data-module]').forEach(a => a.classList.remove('active'));
}

/**
 * 显示模块视图
 * 全局控制栏在进入模块视图时显示，由 syncControls() 负责把状态同步到当前模块实例。
 * 已适配模块会通过 CSS 隐藏内嵌控件，未适配模块保留其内嵌控件但行为与全局一致。
 */
function showModuleView() {
    if (DOM.homeView) DOM.homeView.style.display = 'none';
    if (DOM.moduleView) DOM.moduleView.style.display = 'block';
    if (DOM.controls) DOM.controls.style.display = 'flex';
    // 进入模块视图后，轮询当前模块实例的状态同步到全局控制栏
    startControlsSync();
}

// ============================================
// 模块加载
// ============================================

/**
 * 加载并渲染模块（纯渲染，不改 URL）
 * URL 的同步由 navigateTo / applyRoute 负责。
 * @param {string} moduleName - 模块名称
 */
async function loadModule(moduleName) {
    console.log(`Loading module: ${moduleName}`);

    // 清理之前的模块
    cleanupModule();

    // 更新状态
    AppState.currentModule = moduleName;

    // 显示模块视图
    showModuleView();

    // 更新标题
    const moduleInfo = getModuleInfo(moduleName);
    if (DOM.moduleTitle) {
        DOM.moduleTitle.textContent = moduleInfo.title;
    }

    // 更新侧边栏激活态
    document.querySelectorAll('.sidebar a[data-module]').forEach(a => {
        a.classList.toggle('active', a.dataset.module === moduleName);
    });

    // 标记容器：已适配全局控制栏的模块，用 CSS 隐藏其内嵌控件
    if (DOM.moduleContent) {
        DOM.moduleContent.classList.toggle('uses-global-controls', ADAPTED_MODULES.has(moduleName));
    }

    // 每次加载模块时将全局速度重置为 1×，与绝大多数模块的初始 speed 一致
    if (DOM.speedSelect) {
        DOM.speedSelect.value = '1';
        AppState.speed = 1;
    }

    // 清空内容区
    if (DOM.moduleContent) {
        DOM.moduleContent.innerHTML = '<p style="text-align: center; color: #9CA3AF; padding: 40px;">模块开发中...</p>';
    }

    // 如果已加载，直接初始化，避免重复 <script> 导致 class 重复声明
    const expectedClass = moduleInfo.className;
    if (expectedClass && window[expectedClass]) {
        initializeModule(moduleName);
    } else {
        try {
            const script = document.createElement('script');
            script.src = `js/visualizations/${moduleName}.js`;
            script.dataset.module = moduleName;
            script.onload = () => {
                console.log(`Module ${moduleName} loaded`);
                // 防御：如果用户在脚本加载期间已切换到别的模块，不要把旧模块塞进容器
                if (AppState.currentModule !== moduleName) return;
                initializeModule(moduleName);
            };
            script.onerror = () => {
                console.warn(`Module ${moduleName} not found`);
                if (AppState.currentModule !== moduleName) return;
                showModulePlaceholder(moduleInfo);
            };
            document.body.appendChild(script);
        } catch (error) {
            console.error(`Error loading module ${moduleName}:`, error);
            showModulePlaceholder(moduleInfo);
        }
    }
}

/**
 * 初始化模块
 * @param {string} moduleName - 模块名称
 */
function initializeModule(moduleName) {
    const moduleInfo = getModuleInfo(moduleName);
    const vizName = moduleInfo.className || toPascalCase(moduleName) + 'Visualization';
    const target = window[vizName];

    if (!target) {
        console.warn(`Module ${moduleName}: ${vizName} not found in window`);
        showModulePlaceholder(moduleInfo);
        return;
    }

    try {
        if (typeof target.init === 'function') {
            // Object style: { init(container) }
            const instance = target.init(DOM.moduleContent);
            AppState.moduleInstance = instance;
            console.log(`Module ${moduleName} (${vizName}) initialized`);
        } else if (typeof target === 'function') {
            // Class constructor: new ClassName(container)
            const instance = new target(DOM.moduleContent);
            if (typeof instance.init === 'function') {
                instance.init(DOM.moduleContent);
            }
            AppState.moduleInstance = instance;
            console.log(`Module ${moduleName} (class ${vizName}) initialized`);
        } else {
            console.warn(`Module ${moduleName}: ${vizName} has no init function or constructor`);
            showModulePlaceholder(moduleInfo);
        }
    } catch (error) {
        console.error(`Error initializing ${moduleName}:`, error);
        showModulePlaceholder(moduleInfo);
    }
}

/**
 * 清理当前模块
 */
function cleanupModule() {
    // 停止控制栏轮询，避免访问已被 cleanup 的实例
    stopControlsSync();

    // 停止残留的 rAF（历史字段，保留兼容）
    if (AppState.animationId) {
        cancelAnimationFrame(AppState.animationId);
        AppState.animationId = null;
    }

    // 清理模块实例
    if (AppState.moduleInstance && typeof AppState.moduleInstance.cleanup === 'function') {
        try {
            AppState.moduleInstance.cleanup();
        } catch (error) {
            console.error('Error cleaning up module:', error);
        }
    }

    // 重置状态
    AppState.currentModule = null;
    AppState.isPlaying = false;
    AppState.currentStep = 0;
    AppState.totalSteps = 0;
    AppState.moduleInstance = null;

    // 重置控制按钮到「播放可见，暂停隐藏」的默认状态
    if (DOM.playBtn)  DOM.playBtn.style.display  = '';
    if (DOM.pauseBtn) DOM.pauseBtn.style.display = 'none';
    if (DOM.stepIndicator) DOM.stepIndicator.textContent = '步骤: 0/0';
}

/**
 * 显示模块占位符
 * @param {Object} moduleInfo - 模块信息
 */
function showModulePlaceholder(moduleInfo) {
    if (DOM.moduleContent) {
        DOM.moduleContent.innerHTML = `
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 48px; margin-bottom: 20px;">${moduleInfo.icon}</div>
                <h3 style="color: #F9FAFB; margin-bottom: 10px;">${moduleInfo.title}</h3>
                <p style="color: #9CA3AF; margin-bottom: 20px;">${moduleInfo.description}</p>
                <p style="color: #6B7280; font-size: 14px;">此模块正在开发中...</p>
            </div>
        `;
    }
}

// ============================================
// 控制栏 ←→ 模块实例 代理层
// ============================================
//
// 设计：
//   全局控制栏不再维护自己的动画循环。它只做一件事：把用户的意图
//   （播放/暂停/下一步/重置/改速度）转发给当前模块实例。
//   模块实例是状态的单一来源（`inst.isPlaying`、`inst.currentStep`、
//   `inst.STEPS.length`），我们用 setInterval 低频轮询把这几个字段
//   同步到全局按钮与步骤指示器上。
//
//   这样做的好处是模块可以继续用自己的 setTimeout 驱动自动播放，
//   也可以通过内嵌按钮（未适配模块）触发 play/pause，
//   全局控制栏始终显示最新状态。

/**
 * 播放
 */
function play() {
    const inst = AppState.moduleInstance;
    if (inst && typeof inst.play === 'function') {
        try { inst.play(); } catch (err) { console.error('play failed:', err); }
    }
    syncControls();
}

/**
 * 暂停
 */
function pause() {
    const inst = AppState.moduleInstance;
    if (inst && typeof inst.pause === 'function') {
        try { inst.pause(); } catch (err) { console.error('pause failed:', err); }
    }
    syncControls();
}

/**
 * 单步前进
 * 语义：若正在自动播放，先暂停，再前进一步。
 */
function step() {
    const inst = AppState.moduleInstance;
    if (!inst) return;
    if (inst.isPlaying && typeof inst.pause === 'function') {
        try { inst.pause(); } catch (err) { console.error('pause failed:', err); }
    }
    if (typeof inst.stepForward === 'function') {
        try { inst.stepForward(); } catch (err) { console.error('stepForward failed:', err); }
    }
    syncControls();
}

/**
 * 重置
 */
function reset() {
    const inst = AppState.moduleInstance;
    if (!inst) return;
    if (typeof inst.reset === 'function') {
        try { inst.reset(); } catch (err) { console.error('reset failed:', err); }
    }
    syncControls();
}

/**
 * 启动控制栏状态轮询
 * 用 setInterval 低频轮询当前模块实例的 isPlaying / currentStep，
 * 把状态写回全局按钮与步骤指示器。
 */
function startControlsSync() {
    stopControlsSync();
    syncControls();
    AppState.syncTimer = setInterval(syncControls, 150);
}

/**
 * 停止控制栏状态轮询
 */
function stopControlsSync() {
    if (AppState.syncTimer) {
        clearInterval(AppState.syncTimer);
        AppState.syncTimer = null;
    }
}

/**
 * 把当前模块实例的状态同步到全局控制栏。
 * 约定的读取字段：
 *   inst.isPlaying  : boolean —— 是否正在自动播放
 *   inst.currentStep: number  —— 当前步骤（0-based）
 *   inst.STEPS      : array   —— 步骤列表（用 .length 算总数）
 * 读不到时降级显示「步骤: –」，不抛错。
 */
function syncControls() {
    const inst = AppState.moduleInstance;
    if (!inst) return;

    // 播放/暂停按钮可见性
    const playing = !!inst.isPlaying;
    AppState.isPlaying = playing;
    if (DOM.playBtn)  DOM.playBtn.style.display  = playing ? 'none' : '';
    if (DOM.pauseBtn) DOM.pauseBtn.style.display = playing ? ''     : 'none';

    // 步骤指示器
    let text = '步骤: –';
    if (Array.isArray(inst.STEPS) && inst.STEPS.length > 0) {
        const total = inst.STEPS.length;
        const cur = (typeof inst.currentStep === 'number' ? inst.currentStep : 0) + 1;
        AppState.currentStep = cur;
        AppState.totalSteps = total;
        text = `步骤: ${Math.min(cur, total)}/${total}`;
    }
    if (DOM.stepIndicator) DOM.stepIndicator.textContent = text;
}

// ============================================
// 侧边栏控制
// ============================================

/**
 * 切换侧边栏
 */
function toggleSidebar() {
    if (DOM.sidebar) {
        DOM.sidebar.classList.toggle('collapsed');
    }
}

// ============================================
// 键盘快捷键
// ============================================

/**
 * 处理键盘事件
 * @param {KeyboardEvent} e - 键盘事件
 */
function handleKeyboard(e) {
    // 忽略输入框中的按键
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
        return;
    }

    const inst = AppState.moduleInstance;
    switch (e.key) {
        case ' ':
            e.preventDefault();
            if (inst) {
                if (inst.isPlaying && typeof inst.pause === 'function') inst.pause();
                else if (typeof inst.play === 'function') inst.play();
                syncControls();
            }
            break;
        case 'ArrowRight':
            e.preventDefault();
            if (inst && typeof inst.stepForward === 'function') {
                inst.stepForward();
                syncControls();
            } else {
                step();
            }
            break;
        case 'ArrowLeft':
            e.preventDefault();
            if (inst && typeof inst.stepBack === 'function') {
                inst.stepBack();
                syncControls();
            }
            break;
        case 'r':
        case 'R':
            e.preventDefault();
            if (inst && typeof inst.reset === 'function') {
                inst.reset();
                syncControls();
            } else {
                reset();
            }
            break;
        case 'Escape':
            e.preventDefault();
            navigateTo('');
            break;
    }
}

// ============================================
// 工具函数
// ============================================

/**
 * 获取模块信息
 * @param {string} moduleName - 模块名称
 * @returns {Object} - 模块信息
 */
function getModuleInfo(moduleName) {
    const moduleMap = {
        'attention': { title: 'Self-Attention', icon: '🔍', description: 'Q/K/V → Score → Softmax → Output', className: 'AttentionVisualization' },
        'multi-head-attention': { title: 'Multi-Head Attention', icon: '🎯', description: '多 Head 并行 + Concat + 投影', className: 'MultiHeadAttention' },
        'positional-encoding': { title: 'Positional Encoding', icon: '📍', description: 'sin/cos 位置编码热力图', className: 'PositionalEncodingVisualization' },
        'ffn': { title: 'Feed-Forward Network', icon: '🔗', description: 'Position-wise: Linear → ReLU → Linear', className: 'FFNVisualization' },
        'layer-norm': { title: 'Layer Norm & Residuals', icon: '📏', description: 'LayerNorm + 残差连接的作用', className: 'LayerNormVisualization' },
        'transformer-encoder': { title: 'Transformer Encoder', icon: '📦', description: 'MHA → Add&Norm → FFN → Add&Norm', className: 'TransformerEncoder' },
        'transformer-decoder': { title: 'Transformer Decoder', icon: '📤', description: 'Masked MHA → Cross MHA → FFN', className: 'TransformerDecoder' },
        'forward-backward': { title: '前向/反向传播', icon: '⚡', description: '2层网络 + 梯度计算 + 链式法则', className: 'ForwardBackward' },
        'activations': { title: '激活函数对比', icon: '📈', description: 'ReLU / Sigmoid / Tanh 曲线与梯度', className: 'ActivationsVisualization' },
        'rl-mdp': { title: 'MDP 状态图', icon: '🎲', description: '状态图 + 动作 + 奖励 + 转移概率', className: 'RLMDPTutorial' },
        'q-learning': { title: 'Q-Learning', icon: '📊', description: 'Q-Table 更新 + ε-greedy', className: 'QLearning' },
        'policy-gradient': { title: 'Policy Gradient', icon: '🎯', description: 'REINFORCE + 策略网络 + 轨迹', className: 'PolicyGradient' },
        'dqn': { title: 'DQN', icon: '🧠', description: '经验回放 + 目标网络 + Bellman', className: 'DQN' }
    };

    return moduleMap[moduleName] || { title: moduleName, icon: '📦', description: '模块描述' };
}

/**
 * 转换为 PascalCase
 * @param {string} str - 输入字符串
 * @returns {string} - PascalCase 字符串
 */
function toPascalCase(str) {
    return str
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join('');
}

// ============================================
// 导出全局对象
// ============================================

window.App = {
    AppState,
    navigateTo,
    applyRoute,
    loadModule,     // 低层 API：仅渲染，不改 URL
    showHomeView,   // 低层 API：仅渲染，不改 URL
    play,
    pause,
    step,
    reset,
    syncControls,
    startControlsSync,
    stopControlsSync
};
