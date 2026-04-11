/* ═══════════════════════════════════════════════════════════════
   Deriv Bot Dashboard — app.js
   Polling automático + lógica das abas + Chart.js
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── Backend URL ────────────────────────────────────────────────────────────
// Aponta para o servidor local Flask. Altere se o servidor rodar em outro IP.
const API_BASE_URL = 'http://localhost:5055';

// ─── Estado ────────────────────────────────────────────────────────────────

const state = {
  activeTab:    'overview',
  botRunning:   false,
  histFilter:   'all',
  allTrades:    [],
  activeContract: null,

  // Chart instances
  charts: {
    equity: null,
    price:  null,
    rsi:    null,
    macd:   null,
    rtContract: null,
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

// Candlestick desativado por padrão para evitar renderização distorcida de velas.
const ENABLE_CANDLESTICK = false;

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
      pollContractRealtime();
    }
    if (tab === 'stats') {
      setTimeout(() => { _profitDistChart?.resize(); }, 50);
      pollStats();
    }
    if (tab === 'pipeline') {
      pollSystem();
    }
    if (tab === 'ai') {
      pollModelMetrics();
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
  chart.data.datasets[0].data = data.map(d => ({ x: d.timestamp?.slice(0, 16) || '', y: d.balance_after, _result: d.result }));
  chart.update('none');
}

// ─── Price Chart ────────────────────────────────────────────────────────────

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
async function pollCandles() {
  try {
    const r = await fetch(API_BASE_URL + '/api/candles?n=100');
    const candles = await r.json();
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
  } catch (_) {}
}

function updatePriceChart(ticks) {
  const chart = state.charts.price;
  if (!chart || !ticks.length) return;
  if (state._priceChartIsCandlestick) return; // updated by pollCandles

  chart.data.datasets[0].data = ticks.map(t => ({ x: t.epoch, y: t.price }));
  const prices = ticks.map(t => t.price);
  chart.data.datasets[1].data = calcEma(prices, 9).map((v, i) => ({ x: ticks[i]?.epoch, y: v }));
  chart.data.datasets[2].data = calcEma(prices, 21).map((v, i) => ({ x: ticks[i]?.epoch, y: v }));
  chart.update('none');
}

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
  const side    = isBuy ? '▲ CALL / BUY' : '▼ PUT / SELL';
  const dirColor = isBuy ? '#00ff88' : '#ff4757';
  const dirBg    = isBuy ? 'rgba(0,255,136,0.12)' : 'rgba(255,71,87,0.12)';

  if (statusEl) { statusEl.textContent = `${contract.symbol || '—'} • ${isBuy ? 'CALL/BUY' : 'PUT/SELL'}`; statusEl.style.color = dirColor; }
  if (dirEl)    { dirEl.textContent = side; dirEl.style.background = dirBg; dirEl.style.color = dirColor; }

  const entryTxt   = entryTick ? fmt(entryTick.price, 5) : fmt(contract.entry_price, 5);
  const durationSec = (Number(contract.duration) || 0) * 60;
  const startEpoch  = Number(contract.buy_timestamp) || Number(contract.entry_epoch) || 0;

  function _tick() {
    const now       = Date.now() / 1000;
    const elapsed   = startEpoch > 0 ? Math.max(0, now - startEpoch) : 0;
    const remaining = durationSec > 0 ? Math.max(0, durationSec - elapsed) : null;
    const pct       = durationSec > 0 ? Math.min(100, (elapsed / durationSec) * 100) : 0;
    const currentTxt = lastTick ? fmt(lastTick.price, 5) : '—';

    let timerTxt = '';
    if (remaining !== null) {
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

function updateRtContractChart(ticks, contract) {
  const chart = state.charts.rtContract;
  if (!chart) return;

  if (!contract && (!Array.isArray(ticks) || !ticks.length)) {
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    chart.update('none');
    _setRtContractStatus(null);
    return;
  }

  const series = ticks.map(t => ({ x: Number(t.epoch), y: Number(t.price) }));
  chart.data.datasets[0].data = series;

  let entryTick = null;
  if (contract && contract.entry_epoch) {
    entryTick = ticks.find(t => Number(t.epoch) >= Number(contract.entry_epoch));
    if (!entryTick && ticks.length) entryTick = ticks[0];
  }
  chart.data.datasets[1].data = entryTick ? [{ x: Number(entryTick.epoch), y: Number(entryTick.price) }] : [];

  chart.update('active');
  if (contract) _setRtContractStatus(contract, entryTick, ticks[ticks.length - 1]);
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

async function pollState() {
  try {
    const r = await fetch(API_BASE_URL + '/api/state', { cache: 'no-store' });
    const d = await r.json();
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
      // — Estado operacional detalhado —
      const rs = d.risk_state ?? {};
      const pnlPct = rs.daily_pnl_pct ?? 0;
      const isPaused = (d.summary?.is_paused) || (rs.pause_remaining_sec > 0);
      const isStopLoss   = pnlPct <= -25;
      const isTakeProfit = pnlPct >= 50;
      const hasModel = !!d.model?.model_exists;
      const ticksCount = Number(d.model?.ticks_count ?? 0);
      const datasetRows = Number(d.model?.dataset_rows ?? 0);
      const minTicksTarget = Math.max(1, Number(d.model?.min_ticks_target ?? 500));
      const hasActiveContract = !!d.active_contract?.has_active;
      if (isStopLoss) {
        el('cardStatus').textContent = 'STOP LOSS';
        el('cardStatus').className   = 'card-value negative';
        el('cardStatusSub').textContent = `PnL: ${pnlPct.toFixed(1)}% — Dia encerrado`;
      } else if (isTakeProfit) {
        el('cardStatus').textContent = 'TAKE PROFIT';
        el('cardStatus').className   = 'card-value positive';
        el('cardStatusSub').textContent = `PnL: +${pnlPct.toFixed(1)}% — Meta atingida`;
      } else if (isPaused) {
        el('cardStatus').textContent = 'PAUSADO';
        el('cardStatus').className   = 'card-value warning';
        const rem = rs.pause_remaining_sec ?? 0;
        el('cardStatusSub').textContent = rem > 0
          ? `Retoma em ${Math.floor(rem/60)}m ${rem%60}s`
          : 'Losses consecutivos';
      } else if (!hasModel && ticksCount < minTicksTarget) {
        el('cardStatus').textContent = 'COLETANDO';
        el('cardStatus').className   = 'card-value';
        el('cardStatusSub').textContent = `${ticksCount} / ${minTicksTarget} ticks coletados`;
      } else if (!hasModel && ticksCount >= minTicksTarget) {
        el('cardStatus').textContent = 'CAPTANDO';
        el('cardStatus').className   = 'card-value';
        el('cardStatusSub').textContent = `IA aprendendo (${datasetRows} linhas)`;
      } else {
        el('cardStatus').textContent = 'OPERANDO';
        el('cardStatus').className   = 'card-value positive';
        el('cardStatusSub').textContent = hasActiveContract
          ? 'Contrato ativo em andamento'
          : 'Procurando entrada para o contrato';
      }
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
    if (d.active_symbol) {
      const symEl = el('activeSymbol');
      if (symEl) symEl.textContent = d.active_symbol;
    }

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
    if (el('infoModel')) { el('infoModel').textContent = m.model_exists ? '✓ OK' : '✗ Ausente'; el('infoModel').style.color = m.model_exists ? '#00ff88' : '#ff4757'; }
    if (el('infoTft')) { el('infoTft').textContent = m.tft_exists ? '✓ OK' : '✗ Ausente'; el('infoTft').style.color = m.tft_exists ? '#00ff88' : '#ff4757'; }
    if (el('infoModelMtime') && m.model_mtime) el('infoModelMtime').textContent = new Date(m.model_mtime * 1000).toLocaleString('pt-BR');

    // — Métricas do último treino (via /api/state para atualização a cada 5s) —
    const mm = m.last_metrics ?? {};
    if (el('mmBestModel'))  el('mmBestModel').textContent  = mm.best_model || '—';
    if (el('mmAccuracy'))   el('mmAccuracy').textContent   = mm.accuracy != null ? fmtPct(mm.accuracy * 100) : '—';
    if (el('mmAuc'))        el('mmAuc').textContent        = mm.auc       != null ? fmt(mm.auc, 4)           : '—';
    if (el('mmF1'))         el('mmF1').textContent         = mm.f1        != null ? fmt(mm.f1, 4)            : '—';
    if (el('mmNTrain'))     el('mmNTrain').textContent     = mm.n_train   != null ? mm.n_train.toLocaleString('pt-BR') : '—';
    if (el('mmNTest'))      el('mmNTest').textContent      = mm.n_test    != null ? mm.n_test.toLocaleString('pt-BR')  : '—';
    if (el('mmTimestamp'))  el('mmTimestamp').textContent  = mm.timestamp ? new Date(mm.timestamp).toLocaleString('pt-BR') : '—';
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
  } catch (_) {
    _pollFailures++;
    if (_pollFailures >= 3) _updateConnectionBanner(true);
  }
}

// ─── Notificações de eventos críticos ────────────────────────────────────────

const _notifState = { paused: false, stopLoss: false, drift: false };

function _checkCriticalEvents(summary, riskState) {
  if (!('Notification' in window)) return;
  const hasPerm = Notification.permission === 'granted';

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

function _closeMobileSidebar() {
  const sidebar = el('sidebar');
  const overlay = el('sidebarOverlay');
  const btn = el('hamburgerBtn');
  if (sidebar) sidebar.classList.remove('open');
  if (overlay) overlay.classList.remove('visible');
  if (btn) btn.classList.remove('open');
}

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

// ─── Função compartilhada para aplicar indicadores aos elementos ─────────────

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
      el('cardStatus').textContent = 'RODANDO';
      el('cardStatus').className   = 'card-value positive';
      el('cardStatusSub').textContent = fmtUptime(d.uptime_sec);
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

async function pollTicks() {
  try {
    const r = await fetch(API_BASE_URL + '/api/ticks?n=300');
    const ticks = await r.json();

    if (ticks.length) updatePriceChart(ticks);
  } catch (_) {}
}

async function pollContractRealtime() {
  try {
    const resp = await fetch(API_BASE_URL + '/api/contract-active', { cache: 'no-store' });
    const c = await resp.json();

    if (!c || !c.has_active) {
      // Sem contrato ativo: mostra últimos 150 ticks do símbolo atual (modo monitor)
      state.activeContract = null;
      const activeSymbol = el('activeSymbol')?.textContent?.trim();
      const params = new URLSearchParams({ n: '150' });
      if (activeSymbol) params.set('symbol', activeSymbol);
      try {
        const rTicks = await fetch(API_BASE_URL + '/api/ticks?' + params.toString(), { cache: 'no-store' });
        const ticks = await rTicks.json();
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

async function pollEquity() {
  try {
    const r = await fetch(API_BASE_URL + '/api/equity');
    const data = await r.json();
    updateEquityChart(data);
  } catch (_) {}
}

// ─── Polling: Indicadores ────────────────────────────────────────────────────

async function pollIndicators() {
  try {
    const r = await fetch(API_BASE_URL + '/api/indicators');
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

async function loadHistory() {
  const n = el('histCount')?.value || 50;
  try {
    const r = await fetch(`${API_BASE_URL}/api/trades?n=${n}`);
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
function _authHeaders(extra = {}) {
  const token = (el('authToken') ? el('authToken').value.trim() : '') ||
                localStorage.getItem('dashboardToken') || '';
  if (token) localStorage.setItem('dashboardToken', token);
  return { 'Content-Type': 'application/json', ...(token ? { 'X-Auth-Token': token } : {}), ...extra };
}

async function quickStart() {
  closeStartModal();
  try {
    const r = await fetch(API_BASE_URL + '/api/bot/start', {
      method: 'POST',
      headers: _authHeaders(),
      body: JSON.stringify({ mode: 'demo', balance: 1000, skip_collect: false }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(`Bot iniciado (PID ${d.pid}) \u2014 Demo $1000`, 'ok');
      startStartupPolling();
      pollState();
    } else {
      showToast('Erro: ' + d.msg, 'err');
    }
  } catch (e) {
    showToast('Falha: ' + e.message, 'err');
  }
}
async function confirmStart() {
  closeStartModal();
  const mode            = el('modeSelect').value;
  const balance         = parseFloat(el('balanceInput').value) || 1000;
  const skipCollect     = el('skipCollect').checked;
  const noScan          = el('noScan') ? el('noScan').checked : false;
  const retrainInterval = el('retrainInterval') ? (parseInt(el('retrainInterval').value) || 10) : 10;
  const minTicks        = el('minTicks') ? (parseInt(el('minTicks').value) || 500) : 500;
  const historyCount    = el('historyCount') ? (parseInt(el('historyCount').value) || 500) : 500;
  const forceRetrain    = el('forceRetrain') ? el('forceRetrain').checked : false;

  if (mode === 'real') {
    if (!confirm('⚠ Modo REAL usa dinheiro real. Confirma?')) return;
  }

  try {
    const r = await fetch(API_BASE_URL + '/api/bot/start', {
      method: 'POST',
      headers: _authHeaders(),
      body: JSON.stringify({ mode, balance, skip_collect: skipCollect,
        no_scan: noScan, retrain_interval: retrainInterval,
        min_ticks: minTicks, history_count: historyCount,
        force_retrain: forceRetrain }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(`Bot iniciado (PID ${d.pid}) — modo ${mode}`, 'ok');
      startStartupPolling();
      pollState();
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

// ─── Polling: Estatísticas avançadas ─────────────────────────────────────────

async function _loadIndicatorSeries() {
  try {
    const r = await fetch(API_BASE_URL + '/api/indicator-series?n=60', { cache: 'no-store' });
    const series = await r.json();
    if (!Array.isArray(series) || !series.length) return;
    state.rsiSeries  = series.map(s => ({ x: s.epoch * 1000, y: s.rsi }));
    state.macdSeries = series.map(s => ({ x: s.epoch * 1000, y: s.macd_hist }));
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

async function pollStats() {
  try {
    const r = await fetch(API_BASE_URL + '/api/stats', { cache: 'no-store' });
    const d = await r.json();
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
  } catch (_) {}
}

// ─── Polling: Métricas do modelo ─────────────────────────────────────────────

async function pollModelMetrics() {
  try {
    const r = await fetch(API_BASE_URL + '/api/model-metrics', { cache: 'no-store' });
    const d = await r.json();
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
  } catch (_) {}
}

// ─── Polling: CPU/RAM do sistema ─────────────────────────────────────────────

async function pollSystem() {
  try {
    const r = await fetch(API_BASE_URL + '/api/system');
    const d = await r.json();

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
  } catch (_) {}
}

// ─── Inicialização ────────────────────────────────────────────────────────────

function init() {
  // Restore saved auth token into field if present
  const savedToken = localStorage.getItem('dashboardToken');
  if (savedToken && el('authToken')) el('authToken').value = savedToken;

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
  _loadIndicatorSeries();

  // Primeira carga (estado consolidado + outros)
  pollState();
  pollTicks();
  pollContractRealtime();
  pollEquity();
  pollCandles();
  loadHistory();
  pollModelMetrics();
  pollStats();
  pollSystem();

  // Polls contínuos — pollState() substitui pollBotStatus + pollSummary + pollModel
  setInterval(pollState,    POLL_SUMMARY);
  pollIndicators();
  setInterval(pollIndicators, POLL_IND);
  setInterval(pollTicks,    POLL_TICKS);
  setInterval(pollContractRealtime, POLL_CONTRACT);
  setInterval(pollEquity,   POLL_SUMMARY);
  setInterval(pollCandles,  POLL_TICKS);
  setInterval(loadHistory,  POLL_TRADES);
  setInterval(pollCandlePatterns, POLL_IND);
  setInterval(pollModelMetrics, POLL_MODEL);
  setInterval(pollStats,    POLL_TRADES);
  setInterval(pollSystem,   5_000);

  pollLogs();
  setInterval(pollLogs, POLL_LOGS);
}


// ─── Candle Patterns ───────────────────────────────────────────

async function pollCandlePatterns() {
  try {
    const res = await fetch(API_BASE_URL + '/api/candle-patterns');
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
    // silenciar erros de polling
  }
}

document.addEventListener('DOMContentLoaded', init);
