/* ═══════════════════════════════════════════════════════════════
   Deriv Bot Dashboard — app.js
   Polling automático + lógica das abas + Chart.js
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── Estado ────────────────────────────────────────────────────────────────

const state = {
  activeTab:    'overview',
  botRunning:   false,
  histFilter:   'all',
  allTrades:    [],

  // Chart instances
  charts: {
    equity: null,
    price:  null,
    rsi:    null,
    macd:   null,
  },

  // Dados locais para RSI/MACD (série temporal de indicadores)
  rsiSeries:  [],
  macdSeries: [],
};

// ─── Polling intervals (ms) ──────────────────────────────────────────────────
const POLL_STATUS   = 4_000;
const POLL_SUMMARY  = 5_000;
const POLL_TICKS    = 4_000;
const POLL_TRADES   = 15_000;
const POLL_IND      = 5_000;
const POLL_MODEL    = 30_000;
const POLL_LOGS     = 2_000;

// ─── Helpers ────────────────────────────────────────────────────────────────

function fmt(v, dec = 2)   { return v == null ? '—' : Number(v).toFixed(dec); }
function fmtPct(v)          { return v == null ? '—' : fmt(v, 2) + '%'; }
function fmtUsd(v)          { return v == null ? '—' : '$' + fmt(v, 2); }
function fmtUptime(sec)     {
  if (!sec) return '—';
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return `${h}h ${m}m ${s}s`;
}
function el(id)             { return document.getElementById(id); }
function setClass(elem, cls, flag) {
  if (typeof elem === 'string') elem = el(elem);
  elem?.classList[flag ? 'add' : 'remove'](cls);
}

// ─── Toast ──────────────────────────────────────────────────────────────────

function showToast(msg, type = '') {
  const t = el('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = 'toast'; }, 3500);
}

// ─── Tabs ───────────────────────────────────────────────────────────────────

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const tab = item.dataset.tab;
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    item.classList.add('active');
    el('tab-' + tab)?.classList.add('active');
    el('topbarTitle').textContent = item.querySelector('.nav-label').textContent;
    state.activeTab = tab;

    // Força refresh de charts ao trocar aba
    if (tab === 'technical') {
      setTimeout(() => {
        state.charts.price?.resize();
        state.charts.rsi?.resize();
        state.charts.macd?.resize();
      }, 50);
    }
  });
});

// ─── Chart.js defaults ──────────────────────────────────────────────────────

Chart.defaults.color            = '#8b949e';
Chart.defaults.borderColor      = '#30363d';
Chart.defaults.font.family      = "'JetBrains Mono', monospace";
Chart.defaults.font.size        = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation.duration     = 400;

const baseChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { tooltip: { mode: 'index', intersect: false } },
  scales: {
    x: { grid: { color: '#1e2737' }, ticks: { maxTicksLimit: 8 } },
    y: { grid: { color: '#1e2737' }, ticks: { maxTicksLimit: 6 } },
  },
};

// ─── Equity Chart ───────────────────────────────────────────────────────────

function initEquityChart() {
  const ctx = el('equityChart').getContext('2d');
  state.charts.equity = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Saldo',
        data:  [],
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0,255,136,0.06)',
        tension: 0.3,
        fill: true,
        pointRadius: 2,
        borderWidth: 2,
        pointBackgroundColor: ctx => {
          const d = ctx.dataset.data[ctx.dataIndex];
          return d?._result === 'WIN' ? '#00ff88' : '#ff4757';
        },
      }],
    },
    options: { ...baseChartOptions, scales: {
      x: { ...baseChartOptions.scales.x, ticks: { maxTicksLimit: 6, maxRotation: 0 } },
      y: { ...baseChartOptions.scales.y },
    }},
  });
}

function updateEquityChart(data) {
  const chart = state.charts.equity;
  if (!chart) return;

  chart.data.labels = data.map(d => d.timestamp?.slice(0, 16) || '');
  chart.data.datasets[0].data = data.map(d => ({ x: d.timestamp, y: d.balance_after, _result: d.result }));
  chart.update('none');
}

// ─── Price Chart ────────────────────────────────────────────────────────────

function initPriceChart() {
  const ctx = el('priceChart').getContext('2d');
  state.charts.price = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Preço', data: [], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.04)',
          tension: 0.1, fill: true, pointRadius: 0, borderWidth: 1.5, order: 3 },
        { label: 'EMA 9', data: [], borderColor: '#00ff88', borderWidth: 1.5,
          pointRadius: 0, tension: 0.3, order: 2 },
        { label: 'EMA 21', data: [], borderColor: '#ffb830', borderWidth: 1.5,
          pointRadius: 0, tension: 0.3, order: 1 },
      ],
    },
    options: {
      ...baseChartOptions,
      scales: {
        x: { ...baseChartOptions.scales.x, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { ...baseChartOptions.scales.y, position: 'right' },
      },
    },
  });
}

function updatePriceChart(ticks) {
  const chart = state.charts.price;
  if (!chart || !ticks.length) return;

  const labels = ticks.map(t => t.epoch);
  chart.data.datasets[0].data = ticks.map(t => ({ x: t.epoch, y: t.price }));

  // EMAs calculados localmente (simples, só para visualização)
  const prices = ticks.map(t => t.price);
  chart.data.datasets[1].data = calcEma(prices, 9).map((v, i) => ({ x: ticks[i]?.epoch, y: v }));
  chart.data.datasets[2].data = calcEma(prices, 21).map((v, i) => ({ x: ticks[i]?.epoch, y: v }));

  chart.update('none');
}

// ─── RSI Chart ───────────────────────────────────────────────────────────────

function initRsiChart() {
  const ctx = el('rsiChart').getContext('2d');
  state.charts.rsi = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'RSI', data: [], borderColor: '#a371f7', borderWidth: 1.5,
          fill: false, pointRadius: 0, tension: 0.3 },
      ],
    },
    options: {
      ...baseChartOptions,
      scales: {
        x: { ...baseChartOptions.scales.x, display: false },
        y: { ...baseChartOptions.scales.y, min: 0, max: 100,
             ticks: { values: [35, 50, 65] },
             afterDataLimits: () => {},
        },
      },
    },
  });
}

// ─── MACD Chart ──────────────────────────────────────────────────────────────

function initMacdChart() {
  const ctx = el('macdChart').getContext('2d');
  state.charts.macd = new Chart(ctx, {
    type: 'bar',
    data: {
      datasets: [{
        label: 'MACD Hist',
        data: [],
        backgroundColor: d => {
          const v = d.raw?.y ?? d.raw;
          return v >= 0 ? 'rgba(0,255,136,0.7)' : 'rgba(255,71,87,0.7)';
        },
        borderRadius: 2,
      }],
    },
    options: {
      ...baseChartOptions,
      scales: {
        x: { ...baseChartOptions.scales.x, display: false },
        y: { ...baseChartOptions.scales.y, ticks: { maxTicksLimit: 4 } },
      },
    },
  });
}

// ─── Indicador EMA local (para gráfico de preço) ─────────────────────────────

function calcEma(prices, period) {
  if (prices.length < period) return new Array(prices.length).fill(null);
  const k = 2 / (period + 1);
  const result = new Array(prices.length).fill(null);
  let ema = prices[0];
  for (let i = 0; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
    if (i >= period - 1) result[i] = ema;
  }
  return result;
}

// ─── Polling: Bot Status ──────────────────────────────────────────────────────

async function pollBotStatus() {
  try {
    const r = await fetch('/api/bot/status');
    const d = await r.json();

    state.botRunning = d.running;

    const led  = el('statusLed');
    const text = el('statusText');
    const btnStart = el('btnStart');
    const btnStop  = el('btnStop');

    if (d.running) {
      led.className  = 'status-led running';
      text.textContent = 'Rodando' + (d.pid ? ` (PID ${d.pid})` : '');
      btnStart.style.display = 'none';
      btnStop.style.display  = '';
      el('logBotStatus').textContent = '● Rodando';
      el('logBotStatus').className   = 'log-stat log-bot-indicator running';
      el('cardStatus').textContent = 'RODANDO';
      el('cardStatus').className   = 'card-value positive';
      el('cardStatusSub').textContent = d.uptime_sec ? fmtUptime(d.uptime_sec) : '—';
    } else {
      led.className  = 'status-led stopped';
      text.textContent = 'Parado';
      btnStart.style.display = '';
      btnStop.style.display  = 'none';
      el('logBotStatus').textContent = '● Parado';
      el('logBotStatus').className   = 'log-stat log-bot-indicator stopped';
      el('cardStatus').textContent = 'PARADO';
      el('cardStatus').className   = 'card-value';
      el('cardStatusSub').textContent = '—';
    }
  } catch (_) {}
}

// ─── Polling: Summary ─────────────────────────────────────────────────────────

async function pollSummary() {
  try {
    const r = await fetch('/api/summary');
    const d = await r.json();

    // Saldo
    el('cardBalance').textContent = fmtUsd(d.balance);
    el('cardBalanceSub').textContent = `Inicial: ${fmtUsd(d.balance_initial)}`;

    // P&L
    const pnl = d.pnl_today ?? 0;
    el('cardPnl').textContent = (pnl >= 0 ? '+' : '') + fmtUsd(pnl);
    el('cardPnl').className   = 'card-value ' + (pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '');
    el('cardPnlPct').textContent = (d.pnl_pct >= 0 ? '+' : '') + fmtPct(d.pnl_pct);

    // Win rate
    el('cardWinRate').textContent = fmtPct(d.win_rate);
    el('cardWinRateSub').textContent = `${d.wins} / ${d.total_trades} trades`;

    // IA & Risco — drawdown
    const dd = Math.min(Math.abs(d.drawdown_pct ?? 0) / 25 * 100, 100);
    el('drawdownBar').style.width = dd + '%';
    el('drawdownVal').textContent = fmtPct(d.drawdown_pct ?? 0);

    // Losses consecutivos
    const cl = d.consec_losses ?? 0;
    el('consecVal').textContent = `${cl} / 3`;
    const dotsEl = el('consecDots');
    dotsEl.innerHTML = '';
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('div');
      dot.className = 'consec-dot' + (i < cl ? ' filled' : '');
      dotsEl.appendChild(dot);
    }
    el('consecHint').textContent = d.is_paused ? '⚠ Pausado' : cl > 0 ? `${cl} loss(es)` : 'Normal';

    // Drift alert
    const drift = (d.win_rate_recent > 0) && (d.win_rate_recent < 40);
    el('driftAlert').style.display = drift ? '' : 'none';

    // Win rate dots (últimas 20)
    updateRecentDots('recentWinDots');

    // Timestamp
    el('lastUpdate').textContent = 'Atualizado: ' + new Date().toLocaleTimeString('pt-BR');
  } catch (_) {}
}

// ─── Polling: Ticks ──────────────────────────────────────────────────────────

async function pollTicks() {
  try {
    const r = await fetch('/api/ticks?n=300');
    const ticks = await r.json();

    if (ticks.length) updatePriceChart(ticks);
  } catch (_) {}
}

// ─── Polling: Equity ─────────────────────────────────────────────────────────

async function pollEquity() {
  try {
    const r = await fetch('/api/equity');
    const data = await r.json();
    updateEquityChart(data);
  } catch (_) {}
}

// ─── Polling: Indicadores ────────────────────────────────────────────────────

async function pollIndicators() {
  try {
    const r = await fetch('/api/indicators');
    const d = await r.json();
    if (!Object.keys(d).length) return;

    // Tabela de indicadores
    el('indEma9').textContent  = fmt(d.ema9, 4);
    el('indEma21').textContent = fmt(d.ema21, 4);
    el('indRsi').textContent   = fmt(d.rsi, 1);
    el('indAdx').textContent   = fmt(d.adx, 1);
    el('indMacd').textContent  = (d.macd_hist >= 0 ? '+' : '') + fmt(d.macd_hist, 5);

    // ADX bar
    const adxPct = Math.min((d.adx / 100) * 100, 100);
    el('adxBar').style.width = adxPct + '%';
    el('adxValue').textContent = fmt(d.adx, 1);
    const adxBadge = el('adxBadge');
    if (d.adx >= 20) {
      adxBadge.textContent = 'Tendência';
      adxBadge.style.background = 'rgba(0,255,136,0.15)';
      adxBadge.style.color = '#00ff88';
    } else {
      adxBadge.textContent = 'Lateral';
      adxBadge.style.background = 'rgba(255,184,48,0.15)';
      adxBadge.style.color = '#ffb830';
    }

    // RSI value
    el('rsiValue').textContent = fmt(d.rsi, 1);
    el('rsiValue').style.color = d.rsi > 65 ? '#ff4757' : d.rsi < 35 ? '#00ff88' : '#e6edf3';

    // MACD value
    el('macdValue').textContent = (d.macd_hist >= 0 ? '+' : '') + fmt(d.macd_hist, 5);
    el('macdValue').style.color = d.macd_hist >= 0 ? '#00ff88' : '#ff4757';

    // Market condition badge
    const mc = d.market_condition || 'unknown';
    const mcEl = el('marketCondBadge');
    mcEl.textContent = mc.charAt(0).toUpperCase() + mc.slice(1);
    mcEl.className = 'market-cond-badge ' + (mc === 'trending' ? 'trending' : 'lateral');

    // IA gauge (confiança)
    const conf = d.ai_confidence ?? 0;
    const gaugePerc = Math.min(conf, 1);
    // circumference do semicírculo = ~157
    const dashOffset = 157 - (gaugePerc * 157);
    el('gaugeFill').style.strokeDashoffset = dashOffset;
    el('gaugeFill').style.stroke = conf >= 0.58 ? '#00ff88' : conf >= 0.45 ? '#ffb830' : '#ff4757';
    el('gaugeValue').textContent = fmtPct(conf * 100);
    el('gaugeLabel').textContent = conf >= 0.58 ? 'Confiável' : 'Baixa confiança';

    // AI score bar
    const score = d.ai_score ?? 0;
    el('aiScoreBar').style.width = (score * 100) + '%';
    el('aiScoreVal').textContent = fmt(score, 3);

    // Adiciona ao histórico de séries para RSI/MACD chart
    const now = Date.now();
    state.rsiSeries.push({ x: now, y: d.rsi });
    state.macdSeries.push({ x: now, y: d.macd_hist });
    if (state.rsiSeries.length > 60) state.rsiSeries.shift();
    if (state.macdSeries.length > 60) state.macdSeries.shift();

    // Atualiza gráficos de RSI e MACD
    state.charts.rsi.data.datasets[0].data = [...state.rsiSeries];
    state.charts.rsi.update('none');
    state.charts.macd.data.datasets[0].data = [...state.macdSeries];
    state.charts.macd.update('none');

  } catch (_) {}
}

// ─── Polling: Model info ──────────────────────────────────────────────────────

async function pollModel() {
  try {
    const r = await fetch('/api/model');
    const d = await r.json();

    el('infoTicks').textContent   = d.ticks_count?.toLocaleString('pt-BR') ?? '—';
    el('infoDataset').textContent = d.dataset_rows?.toLocaleString('pt-BR') ?? '—';
    el('infoModel').textContent   = d.model_exists ? '✓ OK' : '✗ Ausente';
    el('infoModel').style.color   = d.model_exists ? '#00ff88' : '#ff4757';
    el('infoTft').textContent     = d.tft_exists   ? '✓ OK' : '✗ Ausente';
    el('infoTft').style.color     = d.tft_exists   ? '#00ff88' : '#ff4757';

    if (d.model_mtime) {
      el('infoModelMtime').textContent = new Date(d.model_mtime * 1000).toLocaleString('pt-BR');
    } else {
      el('infoModelMtime').textContent = '—';
    }
  } catch (_) {}
}

// ─── Histórico de trades ──────────────────────────────────────────────────────

async function loadHistory() {
  const n = el('histCount')?.value || 50;
  try {
    const r = await fetch(`/api/trades?n=${n}`);
    state.allTrades = await r.json();
    renderHistory();
  } catch (_) {}
}

function renderHistory() {
  const filter = state.histFilter;
  let trades = state.allTrades;

  if (filter === 'BUY')  trades = trades.filter(t => t.direction === 'BUY');
  if (filter === 'SELL') trades = trades.filter(t => t.direction === 'SELL');
  if (filter === 'WIN')  trades = trades.filter(t => t.result === 'WIN');
  if (filter === 'LOSS') trades = trades.filter(t => t.result === 'LOSS');

  const tbody = el('histBody');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="empty-msg">Nenhuma operação encontrada.</td></tr>';
    el('histSummary').textContent = '';
    return;
  }

  const totalProfit = trades.reduce((s, t) => s + (t.profit || 0), 0);
  const wins  = trades.filter(t => t.result === 'WIN').length;
  const wr    = trades.length ? (wins / trades.length * 100).toFixed(1) : 0;

  tbody.innerHTML = trades.map(t => `
    <tr>
      <td class="td-mono">${t.timestamp?.slice(0, 19) || '—'}</td>
      <td><span class="badge badge-${t.direction?.toLowerCase()}">${t.direction || '—'}</span></td>
      <td class="td-mono">${fmtUsd(t.stake)}</td>
      <td class="td-mono">${t.duration ?? '—'}t</td>
      <td><span class="badge badge-${t.result?.toLowerCase()}">${t.result || '—'}</span></td>
      <td class="td-mono ${t.profit >= 0 ? 'positive' : 'negative'}">${(t.profit >= 0 ? '+' : '') + fmtUsd(t.profit)}</td>
      <td class="td-mono">${fmtUsd(t.balance_after)}</td>
      <td class="td-mono">${fmt(t.rsi, 1)}</td>
      <td class="td-mono">${fmt(t.adx, 1)}</td>
      <td class="td-mono">${fmtPct(t.ai_confidence * 100)}</td>
      <td>${t.market_condition || '—'}</td>
    </tr>
  `).join('');

  el('histSummary').textContent =
    `${trades.length} operações | Win Rate: ${wr}% | Lucro total: ${(totalProfit >= 0 ? '+' : '') + fmtUsd(totalProfit)}`;

  // Recent dots no overview
  updateRecentDots('recentTradesDots');
}

function updateRecentDots(containerId) {
  const container = el(containerId);
  if (!container) return;
  const recent = state.allTrades.slice(0, 20);
  container.innerHTML = recent.length
    ? recent.map(t => `<div class="tdot ${t.result === 'WIN' ? 'win' : 'loss'}" title="${t.result} — ${fmtUsd(t.profit)}"></div>`).join('')
    : '<span style="color:var(--text-dim);font-size:12px">Sem operações</span>';
}

// ─── Filtros do histórico ─────────────────────────────────────────────────────

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.histFilter = btn.dataset.filter;
    renderHistory();
  });
});

// ─── Exportar CSV ─────────────────────────────────────────────────────────────

function exportCsv() {
  const trades = state.allTrades;
  if (!trades.length) { showToast('Sem dados para exportar.', 'err'); return; }

  const keys = Object.keys(trades[0]);
  const csv  = [keys.join(',')].concat(
    trades.map(t => keys.map(k => JSON.stringify(t[k] ?? '')).join(','))
  ).join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `operacoes_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('CSV exportado!', 'ok');
}

// ─── Controle do bot ──────────────────────────────────────────────────────────

function botStart() {
  el('startModal').style.display = 'flex';
}

function closeStartModal() {
  el('startModal').style.display = 'none';
}
function closeStartupPanel() {
  el('startupPanel').style.display = 'none';
  stopStartupPolling();
}

// ─── Startup panel: fases do pipeline ─────────────────────────────────

// Keywords presentes no stdout do pipeline.py para cada fase
const PHASE_KEYWORDS = {
  'phase-scan':    ['TEND\u00caNCIA', 'Scan prim\u00e1rio', 'Scan secund\u00e1rio', 'score='],
  'phase-collect': ['HIST\u00d3RICO', 'ticks hist\u00f3ricos gravados', 'Buscando os \u00faltimos'],
  'phase-build':   ['DATASET', 'dataset_builder', 'features', 'dataset.csv'],
  'phase-train':   ['TREINO', 'Treinando', 'model.pkl', 'accuracy', 'train_model'],
  'phase-bot':     ['BOT', 'executor', 'Aguardando', 'pipeline', 'Pipeline'],
};

const PHASE_ORDER = ['phase-scan', 'phase-collect', 'phase-build', 'phase-train', 'phase-bot'];

let _startupPollingId = null;
let _currentActivePhase = null;

function _detectPhaseFromLines(lines) {
  let lastActive = null;
  for (const line of lines) {
    for (const [phaseId, keywords] of Object.entries(PHASE_KEYWORDS)) {
      if (keywords.some(k => line.includes(k))) lastActive = phaseId;
    }
  }
  return lastActive;
}

function _updatePhaseUI(activePhaseId) {
  if (activePhaseId === _currentActivePhase) return;
  _currentActivePhase = activePhaseId;
  const activeIdx = PHASE_ORDER.indexOf(activePhaseId);

  PHASE_ORDER.forEach((id, i) => {
    const ph = el(id);
    if (!ph) return;
    const icon = ph.querySelector('.phase-icon');
    if (i < activeIdx) {
      ph.className = 'startup-phase done';
      icon.textContent = '\u2713';
    } else if (i === activeIdx) {
      ph.className = 'startup-phase active';
      icon.textContent = '\u25cf';
    } else {
      ph.className = 'startup-phase';
      icon.textContent = '\u25cb';
    }
  });
}

async function pollStartupLogs() {
  try {
    const r = await fetch('/api/bot/logs');
    const d = await r.json();
    const lines = d.lines || [];

    // Atualiza caixa de log (m\u00e1x 10 linhas mais recentes)
    const logEl = el('startupLog');
    if (logEl && lines.length) {
      logEl.textContent = lines.slice(-10).join('\n');
      logEl.scrollTop = logEl.scrollHeight;
    }

    // Detecta e atualiza fase visual
    const phase = _detectPhaseFromLines(lines);
    if (phase) _updatePhaseUI(phase);

    // Verifica se bot j\u00e1 est\u00e1 confirm\u00e1velmente rodando
    const isRunning = lines.some(l =>
      l.includes('[BOT]') || l.includes('executor') ||
      l.includes('Pipeline') || l.includes('Aguardando')
    );
    if (isRunning) {
      _updatePhaseUI('phase-bot');
      // Fecha o painel ap\u00f3s 2s para o usu\u00e1rio ver a \u00faltima fase
      setTimeout(closeStartupPanel, 2000);
    }
  } catch (_) {}
}

function startStartupPolling() {
  // Mostra overlay e reseta estado visual
  el('startupPanel').style.display = 'flex';
  _currentActivePhase = null;
  PHASE_ORDER.forEach(id => {
    const ph = el(id);
    if (!ph) return;
    ph.className = 'startup-phase';
    ph.querySelector('.phase-icon').textContent = '\u25cb';
    const detail = el(id + '-detail');
    if (detail) detail.textContent = '';
  });
  const logEl = el('startupLog');
  if (logEl) logEl.textContent = 'Aguardando sa\u00edda do pipeline...';

  if (_startupPollingId) clearInterval(_startupPollingId);
  _startupPollingId = setInterval(pollStartupLogs, 1200);
  // Primeira chamada imediata
  setTimeout(pollStartupLogs, 500);
}

function stopStartupPolling() {
  if (_startupPollingId) { clearInterval(_startupPollingId); _startupPollingId = null; }
}

// Quick Start: inicia no modo demo com valores padr\u00e3o, sem precisar da modal
async function quickStart() {
  closeStartModal();
  try {
    const r = await fetch('/api/bot/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'demo', balance: 1000, skip_collect: false }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(`Bot iniciado (PID ${d.pid}) \u2014 Demo $1000`, 'ok');
      startStartupPolling();
    } else {
      showToast('Erro: ' + d.msg, 'err');
    }
  } catch (e) {
    showToast('Falha: ' + e.message, 'err');
  }
}
async function confirmStart() {
  closeStartModal();
  const mode        = el('modeSelect').value;
  const balance     = parseFloat(el('balanceInput').value) || 1000;
  const skipCollect = el('skipCollect').checked;

  if (mode === 'real') {
    if (!confirm('⚠ Modo REAL usa dinheiro real. Confirma?')) return;
  }

  try {
    const r = await fetch('/api/bot/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, balance, skip_collect: skipCollect }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(`Bot iniciado (PID ${d.pid}) — modo ${mode}`, 'ok');
      startStartupPolling();
    } else {
      showToast('Erro: ' + d.msg, 'err');
    }
  } catch (e) {
    showToast('Falha ao iniciar: ' + e.message, 'err');
  }
}

async function botStop() {
  if (!confirm('Parar o bot?')) return;
  try {
    const r = await fetch('/api/bot/stop', { method: 'POST' });
    const d = await r.json();
    if (d.ok) {
      showToast('Bot encerrado.', 'ok');
    } else {
      showToast('Erro: ' + d.msg, 'err');
    }
  } catch (e) {
    showToast('Falha: ' + e.message, 'err');
  }
}

// ─── Pipeline Logs ───────────────────────────────────────────────────────────

let _allLogLines = [];

async function pollLogs() {
  try {
    const r = await fetch('/api/bot/logs');
    const d = await r.json();
    _allLogLines = d.lines || [];
    renderLogs();
    const logLastFetch = el('logLastFetch');
    if (logLastFetch) logLastFetch.textContent = 'Atualizado: ' + new Date().toLocaleTimeString('pt-BR');
    const logLineCount = el('logLineCount');
    if (logLineCount) logLineCount.textContent = `${_allLogLines.length} linhas`;
  } catch (_) {}
}

function renderLogs() {
  const terminal = el('logTerminal');
  if (!terminal) return;
  const filter = (el('logFilter')?.value || '').toLowerCase();
  const lines = filter
    ? _allLogLines.filter(l => l.toLowerCase().includes(filter))
    : _allLogLines;
  if (!lines.length) {
    terminal.innerHTML = '<span class="log-empty-msg">Nenhum log disponível. Inicie o bot para ver a saída do pipeline.</span>';
    return;
  }
  terminal.innerHTML = lines.map(line => {
    const cls = _logLineClass(line);
    const escaped = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<div class="log-line ${cls}">${escaped}</div>`;
  }).join('');
  if (el('logAutoScroll')?.checked) terminal.scrollTop = terminal.scrollHeight;
}

function _logLineClass(line) {
  if (/erro|error|falha|fail|exception|traceback/i.test(line)) return 'log-err';
  if (/warn|aviso/i.test(line))                                 return 'log-warn';
  if (/\[BOT\]|\[PIPELINE\]|\[EXECUTOR\]/i.test(line))         return 'log-bot';
  if (/\[TREINO\]|\[MODEL\]|train|model\.pkl|accuracy/i.test(line)) return 'log-train';
  if (/\[DATASET\]|dataset|features/i.test(line))              return 'log-dataset';
  if (/\[COLETOR\]|ticks|coletando/i.test(line))               return 'log-collect';
  if (/\[TEND|scan|score=/i.test(line))                        return 'log-scan';
  return '';
}

function clearLogs() {
  _allLogLines = [];
  renderLogs();
  const logLineCount = el('logLineCount');
  if (logLineCount) logLineCount.textContent = '0 linhas';
}

// ─── Inicialização ────────────────────────────────────────────────────────────

function init() {
  initEquityChart();
  initPriceChart();
  initRsiChart();
  initMacdChart();

  // Primeira carga
  pollBotStatus();
  pollSummary();
  pollTicks();
  pollEquity();
  pollIndicators();
  pollModel();
  loadHistory();
  pollCandlePatterns();

  // Polls contínuos
  setInterval(pollBotStatus,  POLL_STATUS);
  setInterval(pollSummary,    POLL_SUMMARY);
  setInterval(pollTicks,      POLL_TICKS);
  setInterval(pollEquity,     POLL_SUMMARY);
  setInterval(pollIndicators, POLL_IND);
  setInterval(pollModel,      POLL_MODEL);
  setInterval(loadHistory,    POLL_TRADES);
  setInterval(pollCandlePatterns, POLL_IND);

  pollLogs();
  setInterval(pollLogs, POLL_LOGS);
}


// ─── Candle Patterns ───────────────────────────────────────────

async function pollCandlePatterns() {
  try {
    const res = await fetch('/api/candle-patterns');
    const data = await res.json();
    const wrap = document.getElementById('candlePatternsWrap');
    if (!wrap) return;

    if (!Array.isArray(data) || data.length === 0) {
      wrap.innerHTML = '<div class="candle-pattern-empty">Nenhum padrão detectado</div>';
      return;
    }

    wrap.innerHTML = data.map(p => {
      const icon = { bullish: '🟢', bearish: '🔴', neutral: '🟡' }[p.direction] || '⚪';
      const strengthPct = Math.round(p.strength * 100);
      const barColor = p.direction === 'bullish' ? '#00ff88' : p.direction === 'bearish' ? '#ff4466' : '#ffb830';
      return `
        <div class="candle-pattern-item">
          <span class="cp-icon">${icon}</span>
          <div class="cp-info">
            <div class="cp-name">${p.name}</div>
            <div class="cp-meta">Força: ${strengthPct}% ${p.context ? '| ' + p.context : ''}</div>
            <div class="cp-bar"><div class="cp-bar-fill" style="width:${strengthPct}%;background:${barColor}"></div></div>
          </div>
          <span class="cp-price">${p.price}</span>
        </div>`;
    }).join('');
  } catch (e) {
    // silenciar erros de polling
  }
}

document.addEventListener('DOMContentLoaded', init);
