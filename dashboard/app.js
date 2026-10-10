/* ═══════════════════════════════════════════════════════════════
   Deriv Bot Dashboard — app.js
   Polling automático + lógica das abas + Chart.js
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── Backend URL ────────────────────────────────────────────────────────────
// Por padrão usa a mesma origem que serviu o dashboard. Para proxy/túnel,
// defina window.STANDERIV_API_BASE_URL antes de carregar este arquivo.
const API_BASE_URL = window.STANDERIV_API_BASE_URL || '';

// ─── Estado ────────────────────────────────────────────────────────────────

const state = {
  activeTab:    'overview',
  botRunning:   false,
  histFilter:   'all',
  allTrades:    [],
  activeContract: null,
  backtestRunId: null,
  backtestResultRunId: null,

  // Chart instances
  charts: {
    equity: null,
    price:  null,
    rsi:    null,
    macd:   null,
    rtContract: null,
    backtestEquity: null,
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
const POLL_MODEL    = 10_000;
const POLL_LOGS     = 2_000;
const POLL_CONTRACT = 2_000;
const POLL_BACKTEST = 2_000;
const POLL_PATTERNS = 15_000;

// Candlestick desativado por padrão para evitar renderização distorcida de velas.
const ENABLE_CANDLESTICK = false;

const _pollJobs = new Map();

function _startPoller(name, callback, interval) {
  if (_pollJobs.has(name)) return;
  const job = { active: true, timer: null, running: false, run: null };
  _pollJobs.set(name, job);

  const run = async () => {
    if (!job.active || document.hidden || job.running) return;
    job.running = true;
    try {
      await callback();
    } catch (error) {
      console.warn(`[dashboard:${name}]`, error);
    } finally {
      job.running = false;
      if (job.active && !document.hidden && _pollJobs.get(name) === job) {
        job.timer = setTimeout(run, interval);
      }
    }
  };

  job.run = run;
  run();
}

function _stopPoller(name) {
  const job = _pollJobs.get(name);
  if (!job) return;
  job.active = false;
  clearTimeout(job.timer);
  _pollJobs.delete(name);
}

function _syncTabPollers(tab = state.activeTab) {
  const tabJobs = {
    overview: [
      ['contract', pollContractRealtime, POLL_CONTRACT],
      ['equity', pollEquity, POLL_TRADES],
      ['history', loadHistory, POLL_TRADES],
      ['patterns', pollCandlePatterns, POLL_PATTERNS],
    ],
    technical: [
      ['ticks', pollTicks, POLL_TICKS],
      ['indicators', pollIndicators, POLL_IND],
      ['patterns', pollCandlePatterns, POLL_PATTERNS],
    ],
    ai: [['model-metrics', pollModelMetrics, POLL_MODEL]],
    history: [['history', loadHistory, POLL_TRADES]],
    stats: [['stats', pollStats, POLL_TRADES]],
    backtest: [['backtest', pollBacktestStatus, POLL_BACKTEST]],
    pipeline: [
      ['logs', pollLogs, POLL_LOGS],
      ['system', pollSystem, 5_000],
    ],
  };
  const desired = new Map((tabJobs[tab] || []).map(([name, callback, interval]) => [
    name,
    [callback, interval],
  ]));

  for (const name of _pollJobs.keys()) {
    if (name !== 'state' && !desired.has(name)) _stopPoller(name);
  }
  for (const [name, [callback, interval]] of desired) {
    _startPoller(name, callback, interval);
  }

  if (tab !== 'overview' && _rtContractTimerInterval) {
    clearInterval(_rtContractTimerInterval);
    _rtContractTimerInterval = null;
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    for (const job of _pollJobs.values()) {
      clearTimeout(job.timer);
      job.timer = null;
    }
    if (_rtContractTimerInterval) {
      clearInterval(_rtContractTimerInterval);
      _rtContractTimerInterval = null;
    }
    return;
  }

  for (const job of _pollJobs.values()) job.run();
});

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Formata número com `dec` casas decimais; retorna '—' para null/undefined. */
function fmt(v, dec = 2)   { return v == null ? '—' : Number(v).toFixed(dec); }
/** Formata valor como percentual com 2 casas (ex.: '12.34%'). */
function fmtPct(v)          { return v == null ? '—' : fmt(v, 2) + '%'; }
/** Formata valor em dólares com 2 casas (ex.: '$10.00'). */
function fmtUsd(v)          { return v == null ? '—' : '$' + fmt(v, 2); }
/** Converte segundos em texto 'Xh Ym Zs'; retorna '—' se vazio. */
function fmtUptime(sec)     {
  if (!sec) return '—';
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return `${h}h ${m}m ${s}s`;
}
/** Atalho para document.getElementById. */
function el(id)             { return document.getElementById(id); }
/**
 * Adiciona ou remove a classe `cls` do elemento conforme `flag`.
 * @param {Element|string} elem Elemento ou id do elemento.
 */
function setClass(elem, cls, flag) {
  if (typeof elem === 'string') elem = el(elem);
  elem?.classList[flag ? 'add' : 'remove'](cls);
}
/** Escapa caracteres especiais de HTML para inserção segura em innerHTML. */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

// ─── Toast ──────────────────────────────────────────────────────────────────

/**
 * Exibe o toast com a mensagem por 3,5 s.
 * @param {string} type Classe CSS adicional (ex.: 'ok', 'err').
 */
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
    _syncTabPollers(tab);

    // Close mobile sidebar when selecting a tab
    _closeMobileSidebar();

    // Força refresh de charts ao trocar aba
    if (tab === 'technical') {
      setTimeout(() => {
        state.charts.price?.resize();
        state.charts.rsi?.resize();
        state.charts.macd?.resize();
      }, 50);
    }
    if (tab === 'overview') {
      setTimeout(() => { state.charts.rtContract?.resize(); }, 50);
    }
    if (tab === 'stats') {
      setTimeout(() => { _profitDistChart?.resize(); }, 50);
    }
    if (tab === 'backtest') {
      setTimeout(() => { state.charts.backtestEquity?.resize(); }, 50);
    }
  });
});

// ─── Chart.js fallback/defaults ──────────────────────────────────────────────

if (typeof window.Chart === 'undefined') {
  window.Chart = class ChartFallback {
    static defaults = {
      color: '#8b949e',
      borderColor: '#30363d',
      font: {},
      plugins: { legend: {} },
      animation: {},
    };
    static registry = { controllers: { get: () => null } };

    constructor(_ctx, config = {}) {
      this.config = config;
      this.data = config.data || { labels: [], datasets: [] };
      this.options = config.options || {};
    }

    update() {}
    resize() {}
  };
  console.warn('[dashboard] Chart.js indisponível; gráficos desativados, controles e APIs continuam ativos.');
}

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

/** Cria o gráfico de linha da curva de saldo em state.charts.equity. */
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

/** Atualiza a curva de saldo com os pontos (timestamp, balance_after, result) recebidos. */
function updateEquityChart(data) {
  const chart = state.charts.equity;
  if (!chart) return;

  chart.data.labels = data.map(d => d.timestamp?.slice(0, 16) || '');
  chart.data.datasets[0].data = data.map(d => ({ x: d.timestamp?.slice(0, 16) || '', y: d.balance_after, _result: d.result }));
  chart.update('none');
}

// ─── Price Chart ────────────────────────────────────────────────────────────

/** Cria o gráfico de preço (candlestick se habilitado e disponível, senão linha) com datasets de EMA 9 e 21. */
function initPriceChart() {
  const ctx = el('priceChart').getContext('2d');
  // Usa candlestick apenas se habilitado e com plugin disponível.
  const hasCandlestick = typeof Chart !== 'undefined' &&
    Chart.registry && Chart.registry.controllers &&
    Chart.registry.controllers.get && Chart.registry.controllers.get('candlestick');

  if (ENABLE_CANDLESTICK && hasCandlestick) {
    state.charts.price = new Chart(ctx, {
      type: 'candlestick',
      data: {
        datasets: [
          { label: 'OHLC', data: [], color: { up: '#00ff88', down: '#ff4757', unchanged: '#8b949e' } },
          { type: 'line', label: 'EMA 9',  data: [], borderColor: '#00ff88', borderWidth: 1.5,
            pointRadius: 0, tension: 0.3, order: 1 },
          { type: 'line', label: 'EMA 21', data: [], borderColor: '#ffb830', borderWidth: 1.5,
            pointRadius: 0, tension: 0.3, order: 2 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { tooltip: { mode: 'index', intersect: false }, legend: { display: false } },
        scales: {
          x: { grid: { color: '#1e2737' }, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
          y: { grid: { color: '#1e2737' }, position: 'right' },
        },
      },
    });
    state._priceChartIsCandlestick = true;
  } else {
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
          x: {
            type: 'linear',
            ...baseChartOptions.scales.x,
            ticks: {
              maxTicksLimit: 8,
              maxRotation: 0,
              callback: v => {
                const d = new Date(v * 1000);
                return d.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
              },
            },
          },
          y: { ...baseChartOptions.scales.y, position: 'right' },
        },
      },
    });
    state._priceChartIsCandlestick = false;
  }
}

// Polling periódico de candles (para gráfico candlestick)
/** Busca /api/candles e atualiza OHLC e EMAs do gráfico de preço (apenas no modo candlestick). */
async function pollCandles() {
  try {
    const candles = await _fetchJson('/api/candles?n=100');
    if (!Array.isArray(candles) || !candles.length) return;
    const chart = state.charts.price;
    if (!chart || !state._priceChartIsCandlestick) return;

    chart.data.datasets[0].data = candles.map(c => ({ x: c.t, o: c.o, h: c.h, l: c.l, c: c.c }));
    // EMA sobre closes
    const closes = candles.map(c => c.c);
    const ema9d  = calcEma(closes, 9).map((v, i) => ({ x: candles[i]?.t, y: v }));
    const ema21d = calcEma(closes, 21).map((v, i) => ({ x: candles[i]?.t, y: v }));
    chart.data.datasets[1].data = ema9d;
    chart.data.datasets[2].data = ema21d;
    chart.update('none');
  } catch (error) {
    _recordPollError('candles', error);
  }
}

/** Atualiza o gráfico de preço em modo linha com os ticks (epoch, price) e suas EMAs 9/21. */
function updatePriceChart(ticks) {
  const chart = state.charts.price;
  if (!chart) return;
  if (state._priceChartIsCandlestick) return; // updated by pollCandles

  const validTicks = Array.isArray(ticks) ? ticks : [];
  chart.data.datasets[0].data = validTicks.map(t => ({ x: t.epoch, y: t.price }));
  const prices = validTicks.map(t => t.price);
  chart.data.datasets[1].data = calcEma(prices, 9).map((v, i) => ({ x: ticks[i]?.epoch, y: v }));
  chart.data.datasets[2].data = calcEma(prices, 21).map((v, i) => ({ x: ticks[i]?.epoch, y: v }));
  chart.update('none');
}

/** Cria o gráfico em tempo real do contrato (série de preço e marcador de entrada). */
function initRtContractChart() {
  const canvas = el('rtContractChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  state.charts.rtContract = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Preço',
          data: [],
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88,166,255,0.05)',
          pointRadius: 0,
          borderWidth: 1.6,
          tension: 0.18,
          fill: true,
          order: 2,
        },
        {
          label: 'Entrada',
          data: [],
          type: 'scatter',
          borderColor: '#ffb830',
          backgroundColor: '#ffb830',
          pointRadius: 5,
          pointHoverRadius: 6,
          pointBorderWidth: 2,
          pointBorderColor: '#0d1117',
          showLine: false,
          order: 1,
        },
      ],
    },
    options: {
      ...baseChartOptions,
      scales: {
        x: {
          type: 'linear',
          ...baseChartOptions.scales.x,
          ticks: {
            maxTicksLimit: 8,
            maxRotation: 0,
            callback: v => {
              const d = new Date(v * 1000);
              return d.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            },
          },
        },
        y: { ...baseChartOptions.scales.y, position: 'right' },
      },
    },
  });
}

let _rtContractTimerInterval = null;

/**
 * Atualiza textos, direção, barra de progresso e cronômetro do contrato (reinicia o setInterval de 1 s).
 * Sem `contract`, mostra o estado ocioso.
 */
function _setRtContractStatus(contract, entryTick, lastTick) {
  const statusEl = el('rtContractStatus');
  const metaEl   = el('rtContractMeta');
  const dirEl    = el('rtContractDirection');
  const timerEl  = el('rtContractTimer');
  const progEl   = el('rtContractProgress');

  if (_rtContractTimerInterval) { clearInterval(_rtContractTimerInterval); _rtContractTimerInterval = null; }

  if (!contract) {
    if (statusEl) { statusEl.textContent = 'Aguardando contrato'; statusEl.style.color = '#8b949e'; }
    if (metaEl)   metaEl.textContent = 'Sem contrato ativo no momento.';
    if (dirEl)    { dirEl.textContent = '—'; dirEl.style.background = '#21262d'; dirEl.style.color = '#8b949e'; }
    if (timerEl)  timerEl.textContent = '—';
    if (progEl)   progEl.style.width = '0%';
    return;
  }

  const isBuy   = contract.direction === 'BUY' || contract.direction === 'CALL';
  const isClosed = !!contract.closed;
  const side    = isBuy ? '▲ CALL / BUY' : '▼ PUT / SELL';
  const dirColor = isBuy ? '#00ff88' : '#ff4757';
  const dirBg    = isBuy ? 'rgba(0,255,136,0.12)' : 'rgba(255,71,87,0.12)';

  if (statusEl) {
    statusEl.textContent = `${isClosed ? 'Último contrato' : (contract.phase || 'Ativo')} • ${contract.symbol || '—'} • ${isBuy ? 'CALL/BUY' : 'PUT/SELL'}`;
    statusEl.style.color = dirColor;
  }
  if (dirEl)    { dirEl.textContent = side; dirEl.style.background = dirBg; dirEl.style.color = dirColor; }

  const entryTxt   = entryTick ? fmt(entryTick.price, 5) : fmt(contract.entry_price, 5);
  const durationSec = (Number(contract.duration) || 0) * 60;
  const startEpoch  = Number(contract.buy_timestamp) || Number(contract.entry_epoch) || 0;
  const endEpoch    = Number(contract.exit_epoch) || 0;

  /** Recalcula tempo decorrido/restante e redesenha timer, progresso e meta. */
  function _tick() {
    const now       = isClosed && endEpoch > 0 ? endEpoch : Date.now() / 1000;
    const elapsed   = startEpoch > 0 ? Math.max(0, now - startEpoch) : 0;
    const remaining = durationSec > 0 ? Math.max(0, durationSec - elapsed) : null;
    const pct       = durationSec > 0 ? Math.min(100, (elapsed / durationSec) * 100) : 0;
    const currentTxt = lastTick ? fmt(lastTick.price, 5) : '—';

    let timerTxt = '';
    if (isClosed) {
      const profit = Number(contract.profit);
      const result = contract.result || 'ENCERRADO';
      timerTxt = `${result}${Number.isFinite(profit) ? ` ${profit >= 0 ? '+' : ''}${fmt(profit, 2)} USD` : ''}`;
    } else if (remaining !== null) {
      const rm = Math.ceil(remaining);
      timerTxt = `⏱ ${Math.floor(rm / 60)}m ${rm % 60}s restantes de ${durationSec}s`;
    } else if (elapsed > 0) {
      timerTxt = `Decorrido: ${Math.floor(elapsed)}s`;
    }

    if (timerEl) timerEl.textContent = timerTxt;
    if (progEl)  { progEl.style.width = pct + '%'; progEl.style.background = pct > 80 ? '#ff4757' : isBuy ? '#00ff88' : '#58a6ff'; }
    if (metaEl)  metaEl.textContent = `Entrada: ${entryTxt} | Atual: ${currentTxt} | ${timerTxt}`;
  }

  _tick();
  _rtContractTimerInterval = setInterval(_tick, 1000);
}

/** Plota ticks válidos e o tick de entrada do contrato; sem contrato nem ticks, limpa o gráfico. */
function updateRtContractChart(ticks, contract) {
  const chart = state.charts.rtContract;
  if (!chart) return;
  const validTicks = Array.isArray(ticks)
    ? ticks
        .map(t => ({ epoch: Number(t.epoch), price: Number(t.price) }))
        .filter(t => Number.isFinite(t.epoch) && Number.isFinite(t.price))
    : [];

  if (!contract && !validTicks.length) {
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    chart.update('none');
    _setRtContractStatus(null);
    return;
  }

  const series = validTicks.map(t => ({ x: t.epoch, y: t.price }));
  chart.data.datasets[0].data = series;

  let entryTick = null;
  if (contract && contract.entry_epoch) {
    entryTick = validTicks.find(t => t.epoch >= Number(contract.entry_epoch));
    if (!entryTick && validTicks.length) entryTick = validTicks[0];
  }
  chart.data.datasets[1].data = entryTick ? [{ x: entryTick.epoch, y: entryTick.price }] : [];

  chart.update('active');
  if (contract) _setRtContractStatus(contract, entryTick, validTicks[validTicks.length - 1]);
}

// ─── RSI Chart ───────────────────────────────────────────────────────────────

/** Cria o gráfico de RSI (escala 0–100, linhas destacadas em 30 e 70). */
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
        x: { type: 'linear', ...baseChartOptions.scales.x, display: false },
        y: {
          ...baseChartOptions.scales.y,
          min: 0, max: 100,
          afterBuildTicks: axis => {
            axis.ticks = [0, 30, 50, 70, 100].map(v => ({ value: v }));
          },
          ticks: {
            maxTicksLimit: 8,
          },
          grid: {
            color: ctx => [30, 70].includes(ctx.tick.value) ? 'rgba(255,184,48,0.4)' : '#1e2737',
            lineWidth: ctx => [30, 70].includes(ctx.tick.value) ? 1.5 : 1,
          },
        },
      },
    },
  });
}

// ─── MACD Chart ────────────────────────────────────────────────────────────────────────────

/** Cria o gráfico de barras do histograma MACD (verde positivo, vermelho negativo). */
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
        x: { type: 'linear', ...baseChartOptions.scales.x, display: false },
        y: {
          ...baseChartOptions.scales.y,
          ticks: { maxTicksLimit: 4 },
          grid: {
            color: ctx => ctx.tick.value === 0 ? 'rgba(139,148,158,0.55)' : '#1e2737',
            lineWidth: ctx => ctx.tick.value === 0 ? 2 : 1,
          },
        },
      },
    },
  });
}

// ─── Indicador EMA local (para gráfico de preço) ──────────────────────────────────────────────────
/**
 * Calcula a EMA de `prices`; valores antes do período completo ficam null.
 * @returns {Array<number|null>} Série com o mesmo tamanho da entrada.
 */
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

// ─── Polling: Estado consolidado (/api/state) ────────────────────────────────

let _pollFailures = 0;

function _botOperationalCard(bot = {}, model = {}, riskState = {}, summary = {}, activeContract = {}) {
  const status = bot.status || (bot.running ? 'running' : 'idle');
  const pnlPct = Number(riskState.daily_pnl_pct ?? 0);
  const pauseRemaining = Number(riskState.pause_remaining_sec ?? 0);
  const isPaused = !!summary.is_paused || pauseRemaining > 0;
  const hasModelInfo = Object.prototype.hasOwnProperty.call(model, 'model_exists');
  const hasModel = hasModelInfo ? !!model.model_exists : null;
  const ticksCount = Number(model.ticks_count ?? 0);
  const datasetRows = Number(model.dataset_rows ?? 0);
  const minTicksTarget = Math.max(1, Number(model.min_ticks_target ?? 500));
  const candleCount = Number(bot.candle_count ?? 0);
  const minCandles = Math.max(1, Number(bot.min_candles ?? bot.model_feature_window ?? 0));
  const warmupRemaining = Math.max(0, Number(bot.warmup_remaining_candles ?? Math.max(0, minCandles - candleCount)));
  const hasActiveContract = !!activeContract.has_active;

  if (pnlPct <= -25) {
    return ['STOP LOSS', 'card-value negative', `PnL: ${pnlPct.toFixed(1)}% — Dia encerrado`];
  }
  if (pnlPct >= 50) {
    return ['TAKE PROFIT', 'card-value positive', `PnL: +${pnlPct.toFixed(1)}% — Meta atingida`];
  }
  if (isPaused) {
    return [
      'PAUSADO',
      'card-value warning',
      pauseRemaining > 0 ? `Retoma em ${Math.floor(pauseRemaining/60)}m ${pauseRemaining%60}s` : 'Losses consecutivos',
    ];
  }
  if (['collecting', 'starting'].includes(status)) {
    const suffix = minCandles > 0 ? ` | ${candleCount} / ${minCandles} candles` : '';
    return ['COLETANDO', 'card-value', `${ticksCount} / ${minTicksTarget} ticks coletados${suffix}`];
  }
  if (['training', 'waiting_model'].includes(status) || (hasModel === false && ticksCount >= minTicksTarget)) {
    const detail = warmupRemaining > 0
      ? `Warm-up: faltam ${warmupRemaining} candles`
      : `IA aprendendo (${datasetRows} linhas)`;
    return ['TREINANDO', 'card-value warning', detail];
  }
  if (hasModel === false && ticksCount < minTicksTarget) {
    return ['COLETANDO', 'card-value', `${ticksCount} / ${minTicksTarget} ticks coletados`];
  }
  if (hasModel !== true) {
    return ['RODANDO', 'card-value positive', fmtUptime(bot.uptime_sec)];
  }
  return [
    'OPERANDO',
    'card-value positive',
    hasActiveContract ? 'Contrato ativo em andamento' : 'Procurando entrada para o contrato',
  ];
}

function _renderBotCard(bot = {}, model = {}, riskState = {}, summary = {}, activeContract = {}) {
  if (!bot.running) {
    el('cardStatus').textContent = 'PARADO';
    el('cardStatus').className   = 'card-value';
    el('cardStatusSub').textContent = '—';
    return;
  }
  const [label, className, detail] = _botOperationalCard(bot, model, riskState, summary, activeContract);
  el('cardStatus').textContent = label;
  el('cardStatus').className   = className;
  el('cardStatusSub').textContent = detail;
}

/** Consulta /api/state e atualiza status do bot, cards, risco, modelo e métricas; mostra banner após 3 falhas seguidas. */
async function pollState() {
  try {
    const d = await _fetchJson('/api/state', { cache: 'no-store' });
    _pollFailures = 0;
    _updateConnectionBanner(false);

    // — Bot status —
    const running = d.bot?.running ?? false;
    state.botRunning = running;
    const pid = d.bot?.pid;
    const uptimeSec = d.bot?.uptime_sec;
    const led  = el('statusLed');
    const text = el('statusText');
    if (running) {
      led.className  = 'status-led running';
      text.textContent = 'Rodando' + (pid ? ` (PID ${pid})` : '');
      el('btnStart').style.display = 'none';
      el('btnStop').style.display  = '';
      el('logBotStatus').textContent = '● Rodando';
      el('logBotStatus').className   = 'log-stat log-bot-indicator running';
      _renderBotCard(d.bot, d.model, d.risk_state, d.summary, d.active_contract);
    } else {
      led.className  = 'status-led stopped';
      text.textContent = 'Parado';
      el('btnStart').style.display = '';
      el('btnStop').style.display  = 'none';
      el('logBotStatus').textContent = '● Parado';
      el('logBotStatus').className   = 'log-stat log-bot-indicator stopped';
      el('cardStatus').textContent = 'PARADO';
      el('cardStatus').className   = 'card-value';
      el('cardStatusSub').textContent = '—';
    }

    // — Símbolo ativo —
    const symEl = el('activeSymbol');
    if (symEl) symEl.textContent = d.active_symbol || '';

    // — Resumo —
    const s = d.summary ?? {};
    el('cardBalance').textContent = fmtUsd(s.balance);
    el('cardBalanceSub').textContent = `Inicial: ${fmtUsd(s.balance_initial)}`;
    const pnl = s.pnl_today ?? 0;
    el('cardPnl').textContent = (pnl >= 0 ? '+' : '') + fmtUsd(pnl);
    el('cardPnl').className   = 'card-value ' + (pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '');
    el('cardPnlPct').textContent = (s.pnl_pct >= 0 ? '+' : '') + fmtPct(s.pnl_pct);
    el('cardWinRate').textContent = fmtPct(s.win_rate);
    el('cardWinRateSub').textContent = `${s.wins ?? 0} / ${s.total_trades ?? 0} trades`;
    // Drawdown diário vem do risk_state (mais preciso que o campo do CSV)
    const ddPct = Math.abs(d.risk_state?.daily_pnl_pct ?? s.drawdown_pct ?? 0);
    const dd = Math.min(ddPct / 25 * 100, 100);
    el('drawdownBar').style.width = dd + '%';
    el('drawdownBar').style.background = ddPct > 20 ? 'var(--red)' : ddPct > 10 ? 'var(--yellow)' : 'var(--green)';
    el('drawdownVal').textContent = fmtPct(-(d.risk_state?.daily_pnl_pct ?? 0));
    const cl = s.consec_losses ?? 0;
    el('consecVal').textContent = `${cl} / 3`;
    const dotsEl = el('consecDots');
    if (dotsEl) {
      dotsEl.innerHTML = '';
      for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'consec-dot' + (i < cl ? ' filled' : '');
        dotsEl.appendChild(dot);
      }
    }
    el('consecHint').textContent = s.is_paused ? '⚠ Pausado' : cl > 0 ? `${cl} loss(es)` : 'Normal';
    const drift = (s.win_rate_recent > 0) && (s.win_rate_recent < 40);
    el('driftAlert').style.display = drift ? '' : 'none';

    // Win rate dots (aba IA & Risco)
    updateRecentDots('recentWinDots');

    el('lastUpdate').textContent = 'Atualizado: ' + new Date().toLocaleTimeString('pt-BR');

    // Notificações de eventos críticos (Phase 4.4)
    _checkCriticalEvents(s, d.risk_state ?? {});

    // — Modelo —
    const m = d.model ?? {};
    if (el('infoTicks')) el('infoTicks').textContent = m.ticks_count?.toLocaleString('pt-BR') ?? '—';
    if (el('infoDataset')) el('infoDataset').textContent = m.dataset_rows?.toLocaleString('pt-BR') ?? '—';
    const candidate = m.last_candidate_metrics ?? {};
    if (el('infoModel')) {
      el('infoModel').textContent = m.ready ? '✓ Aprovado' : candidate.timestamp ? 'Treinado; aguardando aprovação' : 'Aguardando treino';
      el('infoModel').style.color = m.ready ? '#00ff88' : '#ffb830';
      el('infoModel').title = (candidate.rejection_reasons ?? []).join('; ') || m.blocked_reason || '';
    }
    if (el('infoTft')) { el('infoTft').textContent = m.tft_exists ? '✓ OK' : '✗ Ausente'; el('infoTft').style.color = m.tft_exists ? '#00ff88' : '#ff4757'; }
    if (el('infoModelMtime') && m.model_mtime) el('infoModelMtime').textContent = new Date(m.model_mtime * 1000).toLocaleString('pt-BR');

    // — Métricas do último treino (via /api/state para atualização a cada 5s) —
    const mm = m.ready ? (m.last_metrics ?? {}) : candidate;
    if (el('mmStatus')) el('mmStatus').textContent = m.ready
      ? 'Modelo aprovado para operação.'
      : candidate.timestamp
        ? `Candidato treinado; operações bloqueadas. ${(candidate.rejection_reasons ?? []).join('; ') || m.blocked_reason || ''}`
        : 'Aguardando dados e primeiro treinamento.';
    if (el('mmBestModel'))  el('mmBestModel').textContent  = mm.best_model ? `${mm.best_model}${m.ready ? '' : ' (candidato)'}` : '—';
    if (el('mmAccuracy'))   el('mmAccuracy').textContent   = mm.accuracy != null ? fmtPct(mm.accuracy * 100) : '—';
    if (el('mmAuc'))        el('mmAuc').textContent        = mm.auc       != null ? fmt(mm.auc, 4)           : '—';
    if (el('mmF1'))         el('mmF1').textContent         = mm.f1        != null ? fmt(mm.f1, 4)            : '—';
    if (el('mmNTrain'))     el('mmNTrain').textContent     = mm.n_train   != null ? mm.n_train.toLocaleString('pt-BR') : '—';
    if (el('mmNTest'))      el('mmNTest').textContent      = mm.n_test    != null ? mm.n_test.toLocaleString('pt-BR')  : '—';
    if (el('mmTimestamp'))  el('mmTimestamp').textContent  = mm.timestamp ? new Date(mm.timestamp).toLocaleString('pt-BR') : '—';
    if (el('mmDiagnostics')) {
      const lines = [];
      for (const [key, label] of [['ai_only', 'IA'], ['technical_only', 'Sinais técnicos'], ['combined', 'Estratégia completa']]) {
        const result = mm.components?.[key];
        if (result) lines.push(`${label}: ${result.trades} contratos simulados, ${result.wins} acertos (${fmtPct(result.win_rate * 100)}).`);
      }
      const selected = (mm.walk_forward ?? []).find(row => mm.selected_recipe
        ? row.recipe_id === mm.selected_recipe : row.name === mm.best_model);
      if (selected) lines.push(`Validação em ${selected.folds.length} janelas futuras: ${selected.trades} contratos; AUC média ${fmt(selected.mean_auc, 4)}.`);
      if (mm.calibration) lines.push(`Qualidade das probabilidades: Brier ${fmt(mm.calibration.multiclass_brier, 4)}; log loss ${fmt(mm.calibration.log_loss, 4)} (menor é melhor).`);
      if (mm.calibration_method) lines.push(`Calibração selecionada: ${mm.calibration_method}; calendário da avaliação: ${mm.evaluation_risk_timezone ?? 'não informado'}.`);
      const observed = mm.observed_payout_evaluation;
      if (observed) lines.push(`Payouts observados: ${observed.trades} contratos avaliados; ${observed.blockers?.missing_observed_payout ?? 0} entradas sem cotação histórica adequada.`);
      for (const warning of mm.selection_warnings ?? []) lines.push(`Atenção: ${warning}.`);
      el('mmDiagnostics').replaceChildren(...lines.map(line => {
        const paragraph = document.createElement('p');
        paragraph.textContent = line;
        return paragraph;
      }));
    }
    if (el('mmAccuracy')) {
      el('mmAccuracy').style.color = mm.accuracy == null
        ? '#8b949e'
        : mm.accuracy >= 0.65 ? '#00ff88' : mm.accuracy >= 0.55 ? '#ffb830' : '#ff4757';
    }

    // — Risk state —
    const rs = d.risk_state ?? {};
    if (rs.pause_remaining_sec > 0) {
      const rem = rs.pause_remaining_sec;
      el('consecHint').textContent = `⚠ Pausado — ${Math.floor(rem/60)}m ${rem%60}s`;
    }
    if (d.indicators && Object.keys(d.indicators).length) {
      _applyIndicators(d.indicators);
    }
    if (d.contracts_source_error) {
      console.warn('[dashboard:contracts]', d.contracts_source_error);
    }
  } catch (error) {
    _pollFailures++;
    if (_pollFailures >= 3) _updateConnectionBanner(true);
    _recordPollError('state', error, 'lastUpdate');
  }
}


// ─── Limpar dados (operacoes + estados) ─────────────────────────────────────
/** Após confirmação, pede ao servidor para apagar dados de operações/estado e recarrega estado, histórico e gráfico de saldo. */
async function clearData() {
  try {
    const confirmMsg = 'Reiniciar a visualização da sessão?\n\nO histórico de contratos e o estado de risco serão preservados.';
    if (!confirm(confirmMsg)) return;
    showToast('Limpando dados...', '');

    const r = await fetch(API_BASE_URL + '/api/bot/data/clear', {
      method: 'POST',
      headers: _authHeaders(),
    });
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.ok) {
      showToast('Visualização da sessão reiniciada.', 'ok');
      // Atualiza estado e histórico na UI
      await pollState();
      await loadHistory();
      // Limpa gráficos locais
      try {
        if (state.charts.equity && state.charts.equity.data && state.charts.equity.data.datasets && state.charts.equity.data.datasets[0]) {
          state.charts.equity.data.datasets[0].data = [];
          if (typeof state.charts.equity.update === 'function') state.charts.equity.update();
        }
      } catch (e) {}
    } else {
      const msg = j.msg || r.statusText || 'Erro desconhecido';
      showToast('Falha ao limpar: ' + msg, 'err');
    }
  } catch (e) {
    showToast('Erro ao limpar dados: ' + (e && e.message ? e.message : e), 'err');
  }
}

// ─── Notificações de eventos críticos ────────────────────────────────────────

const _notifState = { paused: false, stopLoss: false, drift: false };

/** Dispara notificação/toast uma vez por ocorrência de pausa, stop diário e drift. */
function _checkCriticalEvents(summary, riskState) {
  if (!('Notification' in window)) return;
  const hasPerm = Notification.permission === 'granted';

  /** Mostra Notification (se permitido) e toast. */
  function _notify(title, body) {
    if (hasPerm) new Notification(title, { body });
    showToast(`🔔 ${title}: ${body}`, 'err');
  }

  const isPaused = summary.is_paused || (riskState.pause_remaining_sec > 0);
  if (isPaused && !_notifState.paused) {
    _notifState.paused = true;
    _notify('Bot Pausado', 'Losses consecutivos atingiram o limite.');
  } else if (!isPaused) {
    _notifState.paused = false;
  }

  const stopHit = (summary.pnl_pct ?? 0) <= -25;
  if (stopHit && !_notifState.stopLoss) {
    _notifState.stopLoss = true;
    _notify('Stop Diário Atingido', `PnL: ${fmtPct(summary.pnl_pct)}`);
  } else if (!stopHit) {
    _notifState.stopLoss = false;
  }

  const hasDrift = riskState.drift_detected;
  if (hasDrift && !_notifState.drift) {
    _notifState.drift = true;
    _notify('Drift Detectado', `Win rate recente abaixo do mínimo.`);
  } else if (!hasDrift) {
    _notifState.drift = false;
  }
}

// ─── Mobile sidebar toggle ──────────────────────────────────────────────────

/** Fecha a sidebar mobile, o overlay e o estado do botão hambúrguer. */
function _closeMobileSidebar() {
  const sidebar = el('sidebar');
  const overlay = el('sidebarOverlay');
  const btn = el('hamburgerBtn');
  if (sidebar) sidebar.classList.remove('open');
  if (overlay) overlay.classList.remove('visible');
  if (btn) btn.classList.remove('open');
}

/** Alterna a abertura da sidebar mobile, do overlay e do botão hambúrguer. */
function _toggleMobileSidebar() {
  const sidebar = el('sidebar');
  const overlay = el('sidebarOverlay');
  const btn = el('hamburgerBtn');
  const isOpen = sidebar?.classList.toggle('open');
  if (overlay) overlay.classList.toggle('visible', isOpen);
  if (btn) btn.classList.toggle('open', isOpen);
}

// Attach hamburger and overlay listeners once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  el('hamburgerBtn')?.addEventListener('click', _toggleMobileSidebar);
  el('sidebarOverlay')?.addEventListener('click', _closeMobileSidebar);
});

// Resize charts on orientation change
window.addEventListener('resize', () => {
  Object.values(state.charts).forEach(c => c?.resize?.());
  if (typeof _profitDistChart !== 'undefined' && _profitDistChart) _profitDistChart.resize();
});

// ─── Banner de conexão perdida ────────────────────────────────────────────────

/** Exibe ou oculta o banner de conexão perdida, criando-o na topbar se necessário. */
function _updateConnectionBanner(show) {
  let banner = el('connectionBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'connectionBanner';
    banner.className = 'connection-banner';
    banner.textContent = '⚠ Conexão perdida com o servidor';
    document.getElementById('topbar')?.appendChild(banner);
  }
  banner.style.display = show ? '' : 'none';
}

function _recordPollError(scope, error, targetId = null) {
  const message = error?.message || String(error || 'falha desconhecida');
  console.warn(`[dashboard:${scope}]`, error);
  if (targetId && el(targetId)) {
    el(targetId).textContent = `Falha ao atualizar: ${message}`;
  }
}

async function _fetchJson(path, options = {}) {
  const response = await fetch(API_BASE_URL + path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.msg || data?.error || response.statusText || `HTTP ${response.status}`);
  }
  return data;
}

// ─── Função compartilhada para aplicar indicadores aos elementos ─────────────

/** Aplica indicadores (EMA, RSI, ADX, MACD, condição de mercado, confiança e score da IA) aos elementos da DOM. */
function _applyIndicators(d) {
    el('indEma9').textContent  = fmt(d.ema9, 4);
    el('indEma21').textContent = fmt(d.ema21, 4);
    el('indRsi').textContent   = fmt(d.rsi, 1);
    el('indAdx').textContent   = fmt(d.adx, 1);
    el('indMacd').textContent  = (d.macd_hist >= 0 ? '+' : '') + fmt(d.macd_hist, 5);
    const adxPct = Math.min((d.adx / 100) * 100, 100);
    el('adxBar').style.width = adxPct + '%';
    el('adxValue').textContent = fmt(d.adx, 1);
    const adxBadge = el('adxBadge');
    if (d.adx >= 20) { adxBadge.textContent = 'Tendência'; adxBadge.style.background = 'rgba(0,255,136,0.15)'; adxBadge.style.color = '#00ff88'; }
    else { adxBadge.textContent = 'Lateral'; adxBadge.style.background = 'rgba(255,184,48,0.15)'; adxBadge.style.color = '#ffb830'; }
    el('rsiValue').textContent = fmt(d.rsi, 1);
    el('rsiValue').style.color = d.rsi > 65 ? '#ff4757' : d.rsi < 35 ? '#00ff88' : '#e6edf3';
    el('macdValue').textContent = (d.macd_hist >= 0 ? '+' : '') + fmt(d.macd_hist, 5);
    el('macdValue').style.color = d.macd_hist >= 0 ? '#00ff88' : '#ff4757';
    const mc = d.market_condition || 'unknown';
    const mcEl = el('marketCondBadge');
    mcEl.textContent = mc.charAt(0).toUpperCase() + mc.slice(1);
    mcEl.className = 'market-cond-badge ' + (mc === 'trending' ? 'trending' : 'lateral');
    const conf = d.ai_confidence ?? 0;
    const dashOffset = 157 - (conf * 157);
    el('gaugeFill').style.strokeDashoffset = dashOffset;
    el('gaugeFill').style.stroke = conf >= 0.58 ? '#00ff88' : conf >= 0.45 ? '#ffb830' : '#ff4757';
    el('gaugeValue').textContent = fmtPct(conf * 100);
    el('gaugeLabel').textContent = conf >= 0.58 ? 'Confiável' : 'Baixa confiança';
    const score = d.ai_score ?? 0;
    el('aiScoreBar').style.width = (score * 100) + '%';
    el('aiScoreVal').textContent = fmt(score, 3);
    // RSI/MACD chart são atualizados exclusivamente por pollIndicators()
}

/** Consulta /api/bot/status e atualiza LED, botões e card de status do bot. */
async function pollBotStatus() {
  try {
    const r = await fetch(API_BASE_URL + '/api/bot/status');
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
      _renderBotCard(d, {}, {}, {}, {});
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

/** Consulta /api/summary e atualiza saldo, P&L, win rate, drawdown, losses consecutivos e alerta de drift. */
async function pollSummary() {
  try {
    const r = await fetch(API_BASE_URL + '/api/summary');
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

    // IA & Risco — drawdown diário (risk_state)
    const ddPct = Math.abs(d.risk_state?.daily_pnl_pct ?? 0);
    const dd = Math.min(ddPct / 25 * 100, 100);
    el('drawdownBar').style.width = dd + '%';
    el('drawdownBar').style.background = ddPct > 20 ? 'var(--red)' : ddPct > 10 ? 'var(--yellow)' : 'var(--green)';
    el('drawdownVal').textContent = fmtPct(-(d.risk_state?.daily_pnl_pct ?? 0));

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

/** Busca os últimos 300 ticks e atualiza o gráfico de preço. */
async function pollTicks() {
  try {
    const symbol = el('activeSymbol')?.textContent?.trim();
    const params = new URLSearchParams({ n: '300' });
    if (symbol) params.set('symbol', symbol);
    const ticks = await _fetchJson('/api/ticks?' + params.toString(), { cache: 'no-store' });

    updatePriceChart(Array.isArray(ticks) ? ticks : []);
  } catch (error) {
    _recordPollError('ticks', error);
  }
}

/** Busca o contrato ativo (ou o último) e seus ticks, atualizando o gráfico em tempo real; sem contrato, opera como monitor de preço. */
async function pollContractRealtime() {
  try {
    const resp = await fetch(API_BASE_URL + '/api/contract-active', { cache: 'no-store' });
    const c = await resp.json();

    if (!c || !c.has_active) {
      if (c?.last_contract?.contract_id) {
        state.activeContract = null;
        const last = c.last_contract;
        const params = new URLSearchParams({ n: '600' });
        if (last.symbol) params.set('symbol', last.symbol);
        if (last.entry_epoch && last.entry_epoch > 0) params.set('from_epoch', String(last.entry_epoch));
        const rTicks = await fetch(API_BASE_URL + '/api/ticks?' + params.toString(), { cache: 'no-store' });
        const ticks = await rTicks.json();
        updateRtContractChart(Array.isArray(ticks) ? ticks : [], last);
        return;
      }
      // Sem contrato ativo: mostra últimos 150 ticks do símbolo atual (modo monitor)
      state.activeContract = null;
      const activeSymbol = el('activeSymbol')?.textContent?.trim();
      const params = new URLSearchParams({ n: '150' });
      if (activeSymbol && activeSymbol !== '—') params.set('symbol', activeSymbol);
      try {
        const rTicks = await fetch(API_BASE_URL + '/api/ticks?' + params.toString(), { cache: 'no-store' });
        let ticks = await rTicks.json();
        if ((!Array.isArray(ticks) || !ticks.length) && params.has('symbol')) {
          const fallbackTicks = await fetch(API_BASE_URL + '/api/ticks?n=150', { cache: 'no-store' });
          ticks = await fallbackTicks.json();
        }
        if (Array.isArray(ticks) && ticks.length) {
          updateRtContractChart(ticks, null);
          // Exibe mensagem de monitor mas não limpa o gráfico
          const statusEl = el('rtContractStatus');
          const metaEl = el('rtContractMeta');
          if (statusEl) { statusEl.textContent = 'Monitor de preço (sem contrato ativo)'; statusEl.style.color = '#8b949e'; }
          if (metaEl && ticks.length) {
            const last = ticks[ticks.length - 1];
            metaEl.textContent = `Último tick: ${fmt(last.price, 5)} — ${new Date(last.epoch * 1000).toLocaleTimeString('pt-BR')}`;
          }
          return;
        }
      } catch (_) {}
      updateRtContractChart([], null);
      return;
    }

    state.activeContract = c;
    const params = new URLSearchParams({ n: '600' });
    if (c.symbol) params.set('symbol', c.symbol);
    if (c.entry_epoch && c.entry_epoch > 0) params.set('from_epoch', String(c.entry_epoch));

    const rTicks = await fetch(API_BASE_URL + '/api/ticks?' + params.toString(), { cache: 'no-store' });
    const ticks = await rTicks.json();
    updateRtContractChart(Array.isArray(ticks) ? ticks : [], c);
  } catch (e) {
    console.warn('[rtContract]', e);
  }
}

// ─── Polling: Equity ─────────────────────────────────────────────────────────

/** Busca /api/equity e atualiza a curva de saldo. */
async function pollEquity() {
  try {
    const data = await _fetchJson('/api/equity');
    updateEquityChart(data);
  } catch (error) {
    _recordPollError('equity', error);
  }
}

// ─── Polling: Indicadores ────────────────────────────────────────────────────

/** Busca /api/indicators, atualiza painel de indicadores e anexa RSI/MACD às séries (máx. 60 pontos) dos gráficos. */
async function pollIndicators() {
  try {
    const d = await _fetchJson('/api/indicators');
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
    const now = Date.now() / 1000;
    state.rsiSeries.push({ x: now, y: d.rsi });
    state.macdSeries.push({ x: now, y: d.macd_hist });
    if (state.rsiSeries.length > 60) state.rsiSeries.shift();
    if (state.macdSeries.length > 60) state.macdSeries.shift();

    // Atualiza gráficos de RSI e MACD
    state.charts.rsi.data.datasets[0].data = [...state.rsiSeries];
    state.charts.rsi.update('none');
    state.charts.macd.data.datasets[0].data = [...state.macdSeries];
    state.charts.macd.update('none');

  } catch (error) {
    _recordPollError('indicators', error, 'marketCondBadge');
  }
}

// ─── Polling: Model info ──────────────────────────────────────────────────────

/** Consulta /api/model e atualiza informações de ticks, dataset e arquivos do modelo. */
async function pollModel() {
  try {
    const r = await fetch(API_BASE_URL + '/api/model');
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

/**
 * Carrega os trades de /api/trades (`force` ignora cache do servidor) em state.allTrades e redesenha a tabela.
 */
async function loadHistory(force = false) {
  const n = el('histCount')?.value || 50;
  try {
    const payload = await _fetchJson(`/api/trades?n=${n}${force ? '&force=1' : ''}`, { cache: 'no-store' });
    if (Array.isArray(payload)) {
      state.allTrades = payload;
      state.historySource = 'deriv_profit_table';
      state.historyError = '';
    } else {
      state.allTrades = Array.isArray(payload.trades) ? payload.trades : [];
      state.historySource = payload.source || 'deriv_profit_table';
      state.historyError = payload.error || '';
    }
    renderHistory();
  } catch (e) {
    state.historyError = e?.message || 'Falha ao consultar contratos na Deriv.';
    state.allTrades = [];
    _recordPollError('history', e, 'histSummary');
    renderHistory();
  }
}

/** Renderiza a tabela de histórico aplicando o filtro atual e o resumo (contratos, win rate, lucro). */
function renderHistory() {
  const filter = state.histFilter;
  let trades = state.allTrades;

  if (filter === 'BUY')  trades = trades.filter(t => t.direction === 'BUY');
  if (filter === 'SELL') trades = trades.filter(t => t.direction === 'SELL');
  if (filter === 'WIN')  trades = trades.filter(t => t.result === 'WIN');
  if (filter === 'LOSS') trades = trades.filter(t => t.result === 'LOSS');

  const tbody = el('histBody');
  if (!trades.length) {
    const msg = state.historyError
      ? `Não foi possível consultar o relatório de contratos da Deriv: ${escapeHtml(state.historyError)}`
      : 'Nenhum contrato encontrado na Deriv.';
    tbody.innerHTML = `<tr><td colspan="8" class="empty-msg">${msg}</td></tr>`;
    el('histSummary').textContent = state.historyError ? 'Fonte: Deriv profit_table' : '';
    return;
  }

  const totalProfit = trades.reduce((s, t) => s + (t.profit || 0), 0);
  const wins  = trades.filter(t => t.result === 'WIN').length;
  const wr    = trades.length ? (wins / trades.length * 100).toFixed(1) : 0;

  tbody.innerHTML = trades.map(t => `
    <tr>
      <td class="td-mono">${t.timestamp?.slice(0, 19) || '—'}</td>
      <td class="td-mono">${escapeHtml(t.contract_id || '—')}</td>
      <td class="td-mono">${escapeHtml(t.symbol || '—')}</td>
      <td><span class="badge badge-${t.direction?.toLowerCase()}">${t.direction || '—'}</span></td>
      <td class="td-mono">${fmtUsd(t.stake)}</td>
      <td class="td-mono">${fmtUsd(t.sell_price)}</td>
      <td class="td-mono">${t.duration ?? '—'}t</td>
      <td><span class="badge badge-${t.result?.toLowerCase()}">${t.result || '—'}</span></td>
      <td class="td-mono ${t.profit >= 0 ? 'positive' : 'negative'}">${(t.profit >= 0 ? '+' : '') + fmtUsd(t.profit)}</td>
    </tr>
  `).join('');

  el('histSummary').textContent =
    `Fonte: Deriv profit_table | ${trades.length} contratos | Win Rate: ${wr}% | Lucro total: ${(totalProfit >= 0 ? '+' : '') + fmtUsd(totalProfit)}`;

  // Recent dots no overview
  updateRecentDots('recentTradesDots');
}

/** Desenha os pontos WIN/LOSS dos 20 trades mais recentes no contêiner; em 'recentTradesDots' também atualiza as contagens. */
function updateRecentDots(containerId) {
  const container = el(containerId);
  if (!container) return;
  const recent = state.allTrades.slice(0, 20);
  container.innerHTML = recent.length
    ? recent.map(t => `<div class="tdot ${t.result === 'WIN' ? 'win' : 'loss'}" title="${t.result} — ${fmtUsd(t.profit)}"></div>`).join('')
    : '<span style="color:var(--text-dim);font-size:12px">Sem operações</span>';
  if (containerId === 'recentTradesDots') {
    const wins   = recent.filter(t => t.result === 'WIN').length;
    const losses = recent.length - wins;
    const wc = el('recentWinCount');
    const lc = el('recentLossCount');
    if (wc) wc.textContent = wins;
    if (lc) lc.textContent = losses;
  }
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

/** Exporta os trades carregados como arquivo CSV baixado pelo navegador. */
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

/** Abre o modal de configuração de início do bot. */
function botStart() {
  el('startModal').style.display = 'flex';
}

/** Fecha o modal de início do bot. */
function closeStartModal() {
  el('startModal').style.display = 'none';
}
/** Fecha o painel de inicialização e interrompe seu polling de logs. */
function closeStartupPanel() {
  el('startupPanel').style.display = 'none';
  stopStartupPolling();
}

/** Lê os checkboxes de módulos de decisão e devolve o objeto `use_*` enviado ao /api/bot/start. */
function getExecutionSignalSelection() {
  const trained = el('useTrainedModel') ? el('useTrainedModel').checked : true;
  const weighted = el('useWeightedSignal') ? el('useWeightedSignal').checked : true;
  const priceAction = el('usePriceActionFilter') ? el('usePriceActionFilter').checked : false;
  const candlePattern = el('useCandlePatternFilter') ? el('useCandlePatternFilter').checked : false;
  return {
    use_trained_model: trained,
    use_weighted_signal: weighted,
    use_price_action_filter: priceAction,
    use_candle_pattern_filter: candlePattern,
  };
}

/** Descreve os módulos ativos da seleção, separados por ' + ', ou 'Nenhum módulo ativo'. */
function executionSignalLabel(selection = getExecutionSignalSelection()) {
  const active = [];
  if (selection.use_trained_model) {
    active.push('Modelo de treino');
  }
  if (selection.use_weighted_signal) active.push('score ponderado');
  if (selection.use_price_action_filter) active.push('Price Action');
  if (selection.use_candle_pattern_filter) active.push('padrões de vela');
  return active.length ? active.join(' + ') : 'Nenhum módulo ativo';
}

/** Indica se a seleção tem ao menos um módulo de decisão ativo. */
function hasExecutionDecisionModule(selection = getExecutionSignalSelection()) {
  return Boolean(
    selection.use_trained_model ||
    selection.use_weighted_signal ||
    selection.use_price_action_filter ||
    selection.use_candle_pattern_filter
  );
}

/** Atualiza o texto de modo de execução e marca como inválido quando nenhum módulo está ativo. */
function updateExecutionModeStatus() {
  const status = el('executionModeStatus');
  if (!status) return;
  const selection = getExecutionSignalSelection();
  status.textContent = executionSignalLabel(selection);
  status.classList.toggle('invalid', !hasExecutionDecisionModule(selection));
}

['useTrainedModel', 'useWeightedSignal', 'usePriceActionFilter', 'useCandlePatternFilter'].forEach(id => {
  document.addEventListener('change', event => {
    if (event.target?.id === id) updateExecutionModeStatus();
  });
});

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

/** Retorna o id da última fase do pipeline cujas palavras-chave aparecem nas linhas de log (ou null). */
function _detectPhaseFromLines(lines) {
  let lastActive = null;
  for (const line of lines) {
    for (const [phaseId, keywords] of Object.entries(PHASE_KEYWORDS)) {
      if (keywords.some(k => line.includes(k))) lastActive = phaseId;
    }
  }
  return lastActive;
}

/** Marca as fases anteriores como concluídas, a ativa em andamento e as demais pendentes. */
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

/** Busca logs do bot, mostra as 10 últimas linhas, atualiza a fase e fecha o painel quando o bot já está rodando. */
async function pollStartupLogs() {
  try {
    const r = await fetch(API_BASE_URL + '/api/bot/logs');
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

/** Abre o painel de inicialização, reseta as fases e inicia o polling de logs a cada 1,2 s. */
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

/** Cancela o polling de logs do painel de inicialização. */
function stopStartupPolling() {
  if (_startupPollingId) { clearInterval(_startupPollingId); _startupPollingId = null; }
}

// Quick Start: inicia no modo demo com valores padr\u00e3o, sem precisar da modal
/**
 * Monta cabeçalhos JSON com o token X-Auth-Token (dos campos da página ou do localStorage, que é atualizado).
 * @param {Object} extra Cabeçalhos adicionais.
 */
function _authHeaders(extra = {}) {
  const token = (el('backtestAuthToken') ? el('backtestAuthToken').value.trim() : '') ||
                (el('authToken') ? el('authToken').value.trim() : '') ||
                localStorage.getItem('dashboardToken') || '';
  if (token) localStorage.setItem('dashboardToken', token);
  return { 'Content-Type': 'application/json', ...(token ? { 'X-Auth-Token': token } : {}), ...extra };
}

/** Inicia o bot em modo demo (saldo 1000) com os módulos selecionados, sem usar o modal. */
async function quickStart() {
  closeStartModal();
  const executionSelection = getExecutionSignalSelection();
  if (!hasExecutionDecisionModule(executionSelection)) {
    showToast('Ative pelo menos um módulo de decisão.', 'err');
    return;
  }
  try {
    const r = await fetch(API_BASE_URL + '/api/bot/start', {
      method: 'POST',
      headers: _authHeaders(),
      body: JSON.stringify({
        mode: 'demo',
        balance: 1000,
        skip_collect: false,
        ...executionSelection,
      }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(`Bot iniciado (PID ${d.pid}) \u2014 ${executionSignalLabel(executionSelection)}`, 'ok');
      startStartupPolling();
      pollState();
    } else {
      showToast('Erro: ' + d.msg, 'err');
    }
  } catch (e) {
    showToast('Falha: ' + e.message, 'err');
  }
}
/** Inicia o bot com os parâmetros do modal (confirma antes no modo real) e abre o painel de inicialização. */
async function confirmStart() {
  closeStartModal();
  const mode            = el('modeSelect').value;
  const balance         = parseFloat(el('balanceInput').value) || 1000;
  const skipCollect     = el('skipCollect').checked;
  const noScan          = el('noScan') ? el('noScan').checked : false;
  const minTicks        = el('minTicks') ? (parseInt(el('minTicks').value) || 500) : 500;
  const historyCount    = el('historyCount') ? (parseInt(el('historyCount').value) || 2520) : 2520;
  const forceRetrain    = el('forceRetrain') ? el('forceRetrain').checked : false;
  const executionSelection = getExecutionSignalSelection();

  if (mode === 'real') {
    if (!confirm('⚠ Modo REAL usa dinheiro real. Confirma?')) return;
  }
  if (!hasExecutionDecisionModule(executionSelection)) {
    showToast('Ative pelo menos um módulo de decisão.', 'err');
    return;
  }

  try {
    const r = await fetch(API_BASE_URL + '/api/bot/start', {
      method: 'POST',
      headers: _authHeaders(),
      body: JSON.stringify({ mode, balance, skip_collect: skipCollect,
        no_scan: noScan,
        min_ticks: minTicks, history_count: historyCount,
        force_retrain: forceRetrain, ...executionSelection }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(`Bot iniciado (PID ${d.pid}) — ${executionSignalLabel(executionSelection)}`, 'ok');
      startStartupPolling();
      pollState();
    } else {
      showToast('Erro: ' + d.msg, 'err');
    }
  } catch (e) {
    showToast('Falha ao iniciar: ' + e.message, 'err');
  }
}

/** Após confirmação, solicita a parada do bot em /api/bot/stop. */
async function botStop() {
  if (!confirm('Parar o bot?')) return;
  try {
    const r = await fetch(API_BASE_URL + '/api/bot/stop', { method: 'POST', headers: _authHeaders() });
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

/** Busca os logs do bot, guarda em _allLogLines e atualiza terminal e contadores. */
async function pollLogs() {
  try {
    const r = await fetch(API_BASE_URL + '/api/bot/logs');
    const d = await r.json();
    _allLogLines = d.lines || [];
    renderLogs();
    const logLastFetch = el('logLastFetch');
    if (logLastFetch) logLastFetch.textContent = 'Atualizado: ' + new Date().toLocaleTimeString('pt-BR');
    const logLineCount = el('logLineCount');
    if (logLineCount) logLineCount.textContent = `${_allLogLines.length} linhas`;
  } catch (error) {
    _recordPollError('logs', error, 'logLastFetch');
  }
}

/** Renderiza no terminal as linhas de log (filtradas pelo campo de filtro) com classe por tipo e rolagem automática opcional. */
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

/** Retorna a classe CSS de uma linha de log conforme padrões de texto ('' se nenhum). */
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

/** Limpa as linhas de log exibidas (apenas no cliente). */
function clearLogs() {
  _allLogLines = [];
  renderLogs();
  const logLineCount = el('logLineCount');
  if (logLineCount) logLineCount.textContent = '0 linhas';
}

// ─── Polling: Estatísticas avançadas ─────────────────────────────────────────

/** Carrega /api/indicator-series e preenche as séries e gráficos de RSI e MACD. */
async function _loadIndicatorSeries() {
  try {
    const r = await fetch(API_BASE_URL + '/api/indicator-series?n=60', { cache: 'no-store' });
    const series = await r.json();
    if (!Array.isArray(series) || !series.length) return;
    state.rsiSeries  = series.map(s => ({ x: Number(s.epoch), y: s.rsi }));
    state.macdSeries = series.map(s => ({ x: Number(s.epoch), y: s.macd_hist }));
    if (state.charts.rsi) {
      state.charts.rsi.data.datasets[0].data = [...state.rsiSeries];
      state.charts.rsi.update('none');
    }
    if (state.charts.macd) {
      state.charts.macd.data.datasets[0].data = [...state.macdSeries];
      state.charts.macd.update('none');
    }
  } catch (_) {}
}

let _profitDistChart = null;

/** Cria o gráfico de barras da distribuição de lucro por operação. */
function _initProfitDistChart() {
  const ctx = el('profitDistChart');
  if (!ctx) return;
  _profitDistChart = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Operações',
        data: [],
        backgroundColor: d => {
          const lbl = d.chart.data.labels[d.dataIndex] || '';
          return lbl.startsWith('-') || lbl.startsWith('<') ? 'rgba(255,71,87,0.75)' : 'rgba(0,255,136,0.75)';
        },
        borderRadius: 3,
      }],
    },
    options: {
      ...baseChartOptions,
      scales: {
        x: { ...baseChartOptions.scales.x, ticks: { maxRotation: 35, font: { size: 10 } } },
        y: { ...baseChartOptions.scales.y, ticks: { maxTicksLimit: 5 } },
      },
    },
  });
}

/** Consulta /api/stats e atualiza profit factor, expectativa, médias, distribuição de lucro e estatísticas por direção. */
async function pollStats() {
  try {
    const d = await _fetchJson('/api/stats', { cache: 'no-store' });
    if (!d || d.error) return;

    const pf = d.profit_factor;
    el('statProfitFactor').textContent = pf != null ? fmt(pf, 2) : '∞';
    el('statProfitFactor').style.color = pf == null || pf >= 1.2 ? '#00ff88' : pf >= 1.0 ? '#ffb830' : '#ff4757';

    el('statExpectancy').textContent = (d.expectancy >= 0 ? '+' : '') + fmtUsd(d.expectancy);
    el('statExpectancy').style.color = d.expectancy >= 0 ? '#00ff88' : '#ff4757';

    el('statAvgWin').textContent = '+' + fmtUsd(d.avg_win);
    el('statAvgWin').style.color = '#00ff88';
    el('statGrossProfit').textContent = 'Bruto: +' + fmtUsd(d.gross_profit);

    el('statAvgLoss').textContent = fmtUsd(d.avg_loss);
    el('statAvgLoss').style.color = '#ff4757';
    el('statGrossLoss').textContent = 'Bruto: -' + fmtUsd(d.gross_loss);

    // Distribuição de profit
    if (_profitDistChart && d.hist_labels) {
      _profitDistChart.data.labels = d.hist_labels;
      _profitDistChart.data.datasets[0].data = d.hist_counts;
      _profitDistChart.update('none');
    }

    // Por direção
    const wrap = el('dirStatsWrap');
    if (wrap && d.by_direction) {
      const entries = Object.entries(d.by_direction);
      if (!entries.length) {
        wrap.innerHTML = '<div class="candle-pattern-empty">Sem dados</div>';
      } else {
        wrap.innerHTML = entries.map(([dir, s]) => `
          <div class="dir-stat-row">
            <span class="dir-badge dir-${dir.toLowerCase()}">${dir}</span>
            <div class="dir-stat-detail">
              <span>${s.total} trades &nbsp;|&nbsp; WR ${fmtPct(s.wr)}</span>
              <span class="${s.profit >= 0 ? 'positive' : 'negative'}">${(s.profit >= 0 ? '+' : '') + fmtUsd(s.profit)}</span>
            </div>
            <div class="dir-wr-bar-bg"><div class="dir-wr-bar-fill" style="width:${s.wr}%"></div></div>
          </div>`).join('');
      }
    }
  } catch (error) {
    _recordPollError('stats', error, 'dirStatsWrap');
  }
}

// ─── Polling: Métricas do modelo ─────────────────────────────────────────────

/** Consulta /api/model-metrics e atualiza métricas do melhor modelo e o comparativo de ROC-AUC. */
async function pollModelMetrics() {
  try {
    const d = await _fetchJson('/api/model-metrics', { cache: 'no-store' });
    if (!d || !d.best_model) {
      const noData = 'Modelo ainda não treinado';
      ['mmBestModel','mmAccuracy','mmAuc','mmF1','mmNTrain','mmNTest','mmTimestamp']
        .forEach(id => { if (el(id) && el(id).textContent === '—') el(id).textContent = noData; });
      if (el('mmBestModel')) el('mmBestModel').textContent = noData;
      if (el('mmTimestamp')) el('mmTimestamp').textContent = '—';
      return;
    }
    if (el('mmBestModel'))   el('mmBestModel').textContent   = d.best_model;
    if (el('mmAccuracy'))    el('mmAccuracy').textContent    = d.accuracy != null ? fmtPct(d.accuracy * 100) : '—';
    if (el('mmAuc'))         el('mmAuc').textContent         = d.auc       != null ? fmt(d.auc, 4)          : '—';
    if (el('mmF1'))          el('mmF1').textContent          = d.f1        != null ? fmt(d.f1, 4)           : '—';
    if (el('mmNTrain'))      el('mmNTrain').textContent      = d.n_train   != null ? d.n_train.toLocaleString('pt-BR') : '—';
    if (el('mmNTest'))       el('mmNTest').textContent       = d.n_test    != null ? d.n_test.toLocaleString('pt-BR')  : '—';
    if (el('mmTimestamp'))   el('mmTimestamp').textContent   = d.timestamp ? new Date(d.timestamp).toLocaleString('pt-BR') : '—';

    // Colorir acurácia
    if (el('mmAccuracy') && d.accuracy != null) {
      el('mmAccuracy').style.color = d.accuracy >= 0.65 ? '#00ff88' : d.accuracy >= 0.55 ? '#ffb830' : '#ff4757';
    }

    // Comparativo ROC-AUC
    const wrap = el('rocComparisonWrap');
    if (wrap && Array.isArray(d.models_comparison) && d.models_comparison.length > 0) {
      const maxAuc = Math.max(...d.models_comparison.map(m => m.auc));
      wrap.innerHTML = d.models_comparison.map(m => {
        const pct = Math.round((m.auc / 0.7) * 100); // escala visual: 0.7 = 100%
        const barW = Math.min(pct, 100);
        const color = m.name === d.best_model ? '#00ff88' : '#58a6ff';
        return `<div class="roc-row">
          <span class="roc-name">${m.name}</span>
          <div class="roc-bar-bg"><div class="roc-bar-fill" style="width:${barW}%;background:${color}"></div></div>
          <span class="roc-auc" style="color:${color}">${fmt(m.auc, 4)}</span>
        </div>`;
      }).join('');
    }
  } catch (error) {
    _recordPollError('model-metrics', error, 'mmStatus');
  }
}

// ─── Polling: CPU/RAM do sistema ─────────────────────────────────────────────

/** Consulta /api/system e atualiza as barras de CPU, RAM, disco e consumo do bot. */
async function pollSystem() {
  try {
    const d = await _fetchJson('/api/system');

    /** Preenche barra e valor (cor por limiar); ignora se faltar elemento ou pct. */
    function _setBar(barId, valId, pct, suffix, isBot = false) {
      const bar = el(barId);
      const val = el(valId);
      if (!bar || !val || pct == null) return;
      bar.style.width = Math.min(pct, 100) + '%';
      bar.style.background = pct > 85 ? '#ff4757' : pct > 65 ? '#ffb830' : isBot ? '#58a6ff' : '#00ff88';
      val.textContent = fmt(pct, 1) + suffix;
    }

    _setBar('sysCpuBar', 'sysCpuVal', d.cpu_pct, '%');
    _setBar('sysRamBar', 'sysRamVal', d.ram_pct, `% (${fmt(d.ram_used_mb, 0)} MB)`);
    _setBar('sysDiskBar', 'sysDiskVal', d.disk_pct, '%');
    _setBar('botCpuBar', 'botCpuVal', d.bot_cpu_pct, '%', true);

    const botRamBar = el('botRamBar');
    const botRamVal = el('botRamVal');
    if (botRamBar && botRamVal && d.bot_ram_mb != null) {
      botRamBar.style.width = Math.min(d.bot_ram_mb / 5, 100) + '%';
      botRamBar.style.background = '#58a6ff';
      botRamVal.textContent = fmt(d.bot_ram_mb, 1) + ' MB';
    }
  } catch (error) {
    _recordPollError('system', error);
  }
}

// ─── Backtest offline ───────────────────────────────────────────────────────

/** Cria o gráfico de linha temporal do saldo simulado do backtest. */
function initBacktestChart() {
  const canvas = el('backtestEquityChart');
  if (!canvas) return;
  state.charts.backtestEquity = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      datasets: [{
        label: 'Saldo simulado',
        data: [],
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.10)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
        fill: true,
      }],
    },
    options: {
      ...baseChartOptions,
      parsing: false,
      scales: {
        x: {
          type: 'time',
          grid: { display: false },
          ticks: { maxTicksLimit: 6, maxRotation: 0 },
        },
        y: { ...baseChartOptions.scales.y },
      },
    },
  });
}

/** Atualiza badge, detalhe e estado dos botões de iniciar/cancelar do backtest conforme `status`. */
function _setBacktestStatus(status, detail = '') {
  const badge = el('backtestStatus');
  const allowed = ['idle', 'running', 'completed', 'failed', 'stopped', 'stopping'];
  const normalized = allowed.includes(status) ? status : 'idle';
  const labels = {
    idle: 'Aguardando',
    running: 'Executando',
    completed: 'Concluído',
    failed: 'Falhou',
    stopped: 'Cancelado',
    stopping: 'Cancelando',
  };
  if (badge) {
    badge.className = `backtest-status ${normalized}`;
    badge.textContent = labels[normalized];
  }
  if (el('backtestStatusDetail') && detail) {
    el('backtestStatusDetail').textContent = detail;
  }
  const running = normalized === 'running' || normalized === 'stopping';
  if (el('backtestStartBtn')) {
    el('backtestStartBtn').disabled = running || state.botRunning;
  }
  if (el('backtestStopBtn')) {
    el('backtestStopBtn').style.display = running ? '' : 'none';
    el('backtestStopBtn').disabled = normalized === 'stopping';
  }
}

/** Valida o formulário e inicia um backtest offline via /api/backtest/start (bloqueado com o bot rodando). */
async function startBacktest() {
  if (state.botRunning) {
    showToast('Pare o bot antes de executar o backtest.', 'err');
    return;
  }
  const numericIds = [
    'backtestBalance', 'backtestPayout', 'backtestDuration',
    'backtestTimeframe', 'backtestMaxTicks', 'backtestCostRate',
    'backtestFixedCost',
  ];
  for (const id of numericIds) {
    if (!el(id)?.checkValidity()) {
      el(id).reportValidity();
      return;
    }
  }
  const withAi = !!el('backtestWithAi')?.checked;
  const dynamicDuration = !!el('backtestDynamicDuration')?.checked;
  if ((withAi || dynamicDuration) && !confirm(
    'Os modelos só podem ser avaliados em dados fora da amostra. Confirma que estes ticks não participaram do treinamento?'
  )) return;

  const payload = {
    symbol: el('backtestSymbol')?.value.trim() || '',
    balance: Number(el('backtestBalance')?.value),
    payout: Number(el('backtestPayout')?.value),
    duration: Number(el('backtestDuration')?.value),
    duration_unit: el('backtestDurationUnit')?.value || 'm',
    timeframe: Number(el('backtestTimeframe')?.value),
    max_ticks: Number(el('backtestMaxTicks')?.value),
    cost_rate: Number(el('backtestCostRate')?.value),
    fixed_cost: Number(el('backtestFixedCost')?.value),
    with_ai: withAi,
    dynamic_duration: dynamicDuration,
    candle_patterns: !!el('backtestCandlePatterns')?.checked,
  };

  try {
    _setBacktestStatus('running', 'Preparando execução offline...');
    const response = await fetch(API_BASE_URL + '/api/backtest/start', {
      method: 'POST',
      headers: _authHeaders(),
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      _setBacktestStatus('failed', data.msg || 'Não foi possível iniciar.');
      showToast(data.msg || 'Falha ao iniciar backtest.', 'err');
      return;
    }
    state.backtestRunId = data.run_id;
    state.backtestResultRunId = null;
    _setBacktestStatus('running', `Execução ${data.run_id} iniciada.`);
    showToast('Backtest iniciado.', 'ok');
    pollBacktestStatus();
  } catch (error) {
    _setBacktestStatus('failed', error.message);
    showToast('Falha ao iniciar backtest: ' + error.message, 'err');
  }
}

/** Após confirmação, solicita o cancelamento do backtest em /api/backtest/stop. */
async function stopBacktest() {
  if (!confirm('Cancelar o backtest em execução?')) return;
  try {
    _setBacktestStatus('stopping', 'Solicitando cancelamento...');
    const response = await fetch(API_BASE_URL + '/api/backtest/stop', {
      method: 'POST',
      headers: _authHeaders(),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      showToast(data.msg || 'Não foi possível cancelar.', 'err');
      return;
    }
    showToast('Cancelamento solicitado.', 'ok');
  } catch (error) {
    showToast('Falha ao cancelar: ' + error.message, 'err');
  }
}

/** Consulta /api/backtest/status, atualiza status e logs e carrega o resultado quando concluído e ainda não exibido. */
async function pollBacktestStatus() {
  try {
    const data = await _fetchJson('/api/backtest/status', { cache: 'no-store' });
    state.backtestRunId = data.run_id || state.backtestRunId;
    let detail = 'Nenhuma execução iniciada.';
    if (data.run_id) detail = `Execução ${data.run_id}`;
    if (data.error) detail += ` — ${data.error}`;
    _setBacktestStatus(data.status || 'idle', detail);

    const logs = el('backtestLogs');
    if (logs) {
      logs.textContent = Array.isArray(data.logs) ? data.logs.join('\n') : '';
      logs.style.display = logs.textContent ? '' : 'none';
    }
    if (
      data.report_available
      && data.status === 'completed'
      && state.backtestResultRunId !== data.run_id
    ) {
      await loadBacktestResult(data.run_id);
    }
  } catch (error) {
    _recordPollError('backtest-status', error, 'backtestStatusDetail');
  }
}

/** Busca o relatório em /api/backtest/result e o renderiza, registrando `runId` como já exibido. */
async function loadBacktestResult(runId = state.backtestRunId) {
  try {
    const data = await _fetchJson('/api/backtest/result?n=200', { cache: 'no-store' });
    if (!data.ok) return;
    renderBacktestResult(data.report);
    state.backtestResultRunId = runId;
  } catch (error) {
    _recordPollError('backtest-result', error, 'backtestStatusDetail');
  }
}

/** Preenche cards de métricas, curva de saldo e tabela de operações simuladas a partir do relatório do backtest. */
function renderBacktestResult(report) {
  const metrics = report?.metrics || {};
  const settings = report?.settings || {};
  const trades = Array.isArray(report?.trades) ? report.trades : [];

  el('backtestTrades').textContent = metrics.trades ?? 0;
  el('backtestWinsLosses').textContent = `${metrics.wins ?? 0} wins / ${metrics.losses ?? 0} losses`;
  el('backtestWinRate').textContent = fmtPct(metrics.win_rate_pct);
  el('backtestReturn').textContent = (metrics.return_pct > 0 ? '+' : '') + fmtPct(metrics.return_pct);
  el('backtestReturn').className = 'card-value ' + (metrics.return_pct > 0 ? 'positive' : metrics.return_pct < 0 ? 'negative' : '');
  el('backtestNetProfit').textContent = (metrics.net_profit > 0 ? '+' : '') + fmtUsd(metrics.net_profit);
  el('backtestDrawdown').textContent = fmtPct(metrics.max_drawdown_pct);
  el('backtestDrawdownUsd').textContent = fmtUsd(metrics.max_drawdown_abs);
  el('backtestFinalBalance').textContent = fmtUsd(metrics.final_balance);
  el('backtestProfitFactor').textContent = metrics.profit_factor == null ? '—' : fmt(metrics.profit_factor, 3);
  el('backtestExpectancy').textContent = fmtUsd(metrics.expectancy);
  el('backtestTotalStaked').textContent = fmtUsd(metrics.total_staked);
  el('backtestTotalCosts').textContent = fmtUsd(metrics.total_costs);
  el('backtestMaxLosses').textContent = metrics.max_consecutive_losses ?? '—';
  el('backtestResultSymbol').textContent = report.symbol || '—';
  el('backtestMode').textContent = settings.use_ai ? 'Técnico + IA' : 'Somente técnico';

  const chart = state.charts.backtestEquity;
  if (chart) {
    chart.data.datasets[0].data = (report.equity_curve || []).map(point => ({
      x: Number(point.epoch) * 1000,
      y: Number(point.balance),
    }));
    chart.update();
  }

  const tbody = el('backtestTradesBody');
  if (!tbody) return;
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-msg">Nenhuma operação simulada.</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(trade => {
    const direction = trade.direction === 'BUY' ? 'BUY' : 'SELL';
    const outcome = ['WIN', 'LOSS', 'TIE'].includes(trade.outcome) ? trade.outcome : 'LOSS';
    const expiry = new Date(Number(trade.expiry_epoch) * 1000).toLocaleString('pt-BR');
    const profit = Number(trade.net_profit || 0);
    return `<tr>
      <td class="td-mono">${escapeHtml(expiry)}</td>
      <td><span class="badge badge-${direction.toLowerCase()}">${direction}</span></td>
      <td class="td-mono">${fmt(trade.entry_price, 5)}</td>
      <td class="td-mono">${fmt(trade.exit_price, 5)}</td>
      <td class="td-mono">${Number(trade.duration)}${escapeHtml(trade.duration_unit)}</td>
      <td><span class="badge badge-${outcome.toLowerCase()}">${outcome}</span></td>
      <td class="td-mono">${fmtUsd(trade.stake)}</td>
      <td class="td-mono ${profit >= 0 ? 'positive' : 'negative'}">${profit > 0 ? '+' : ''}${fmtUsd(profit)}</td>
      <td class="td-mono">${fmtUsd(trade.balance_after)}</td>
    </tr>`;
  }).join('');
}

// ─── Inicialização ────────────────────────────────────────────────────────────

/** Inicializa o dashboard: restaura token, cria gráficos, faz a primeira carga e agenda os polls periódicos. */
function init() {
  // Restore saved auth token into field if present
  const savedToken = localStorage.getItem('dashboardToken');
  if (savedToken && el('authToken')) el('authToken').value = savedToken;
  if (savedToken && el('backtestAuthToken')) el('backtestAuthToken').value = savedToken;

  // Solicitar permissão para notificações push
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }

  initEquityChart();
  initPriceChart();
  initRtContractChart();
  initRsiChart();
  initMacdChart();
  _initProfitDistChart();
  initBacktestChart();
  _loadIndicatorSeries();
  updateExecutionModeStatus();

  _startPoller('state', pollState, POLL_SUMMARY);
  _syncTabPollers();
}


// ─── Candle Patterns ───────────────────────────────────────────

/** Busca /api/candle-patterns e lista os padrões de vela (ícone, força, contexto) nos painéis correspondentes. */
async function pollCandlePatterns() {
  try {
    const params = new URLSearchParams();
    const symbol = el('activeSymbol')?.textContent?.trim();
    if (symbol) params.set('symbol', symbol);
    const res = await fetch(API_BASE_URL + '/api/candle-patterns?' + params.toString(), { cache: 'no-store' });
    const data = await res.json();

    const wraps = [
      document.getElementById('candlePatternsWrap'),
      document.getElementById('candlePatternsWrapOverview'),
    ];

    for (const wrap of wraps) {
      if (!wrap) continue;

      if (!Array.isArray(data) || data.length === 0) {
        wrap.innerHTML = '<div class="candle-pattern-empty">Nenhum padrão detectado</div>';
        continue;
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
    }
  } catch (e) {
    _recordPollError('candle-patterns', e);
  }
}

document.addEventListener('DOMContentLoaded', init);
