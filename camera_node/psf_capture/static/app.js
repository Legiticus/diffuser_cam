// PSF Capture frontend logic (Alpine.js component `app`).
//
//   status    : `s` mirrors the backend snapshot, kept current over server-sent events
//               (/api/events); EventSource reconnects by itself after a reload or outage
//   live      : stage + exposure controls, histogram polling while the Live tab is open
//   capture   : session form, saved z plans, plan preview, progress, pause/abort/continue
//   model     : session selection, processing steps, build job, points, generate, plots

function app() {
  return {
    tab: localStorage.getItem('tab') || 'live',
    connected: false,
    toast: '',
    s: {state: 'idle', warnings: [], live: {}, job: {}, progress: {}},
    d: {z_range: [10, 340], exposure_limits: [9, 1e7], gain_limits: [1, 16], options: {}},

    // live
    zTarget: 10, liveExposure: 10000, liveGain: 1, stats: {}, histLog: true,

    // capture
    form: {x_mm: '', y_mm: '', plan_name: ''},
    plans: {}, planName: '', planPreview: {},

    // model
    sessions: [], selected: [], steps: {}, built: null,
    points: [{x: 0, y: 0, z: 100, intensity: 1}], result: {}, gamma: 2,

    get p() { return this.s.progress || {}; },

    async init() {
      this.d = await this.get('/api/defaults');
      const o = this.d.options;
      this.form = {...this.form, z_min: o.z_min, z_max: o.z_max, n_positions: o.n_positions,
        spacing: o.spacing, frames_per_position: o.frames_per_position, passes: o.passes,
        background: o.background, background_frames: '', exposure_mode: o.exposure_mode,
        target_fraction: o.target_fraction, max_exposure_us: o.max_exposure_us,
        min_signal_fraction: o.min_signal_fraction, settle_s: o.settle_s,
        allowed_saturated_px: o.allowed_saturated_px};
      Object.assign(this.form, JSON.parse(localStorage.getItem('form') || '{}'));   // survive reloads
      this.$watch('form', (f) => localStorage.setItem('form', JSON.stringify(f)));
      this.steps = {...this.d.processing};
      this.liveExposure = o.exposure_us;
      this.liveGain = o.gain;
      this.s = await this.get('/api/status');
      if (this.s.job && this.s.job.result) this.built = this.s.job.result;
      this.plans = await this.get('/api/plans');
      this.connect();
      this.previewPlan();
      this.loadSessions();
      setInterval(() => this.pollStats(), 500);
    },

    connect() {
      const es = new EventSource('/api/events');
      es.onopen = () => { this.connected = true; };
      es.onerror = () => { this.connected = false; };
      es.onmessage = (ev) => {
        const prev = this.s;
        this.s = JSON.parse(ev.data);
        if (this.s.job && this.s.job.result && prev.job && prev.job.running) this.built = this.s.job.result;
        if (prev.state !== 'idle' && this.s.state === 'idle') this.loadSessions();
      };
    },

    // --- helpers -------------------------------------------------------------------
    async get(url) {
      const r = await fetch(url);
      if (!r.ok) throw new Error((await r.json()).detail);
      return r.json();
    },
    async post(url, body = {}, method = 'POST') {
      const r = await fetch(url, {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        this.toast = data.detail || r.statusText;
        throw new Error(this.toast);
      }
      return data;
    },
    setTab(t) { this.tab = t; localStorage.setItem('tab', t); if (t === 'model') this.loadSessions(); },
    stateLabel() { return this.s.state.replaceAll('_', ' ') + (this.s.paused ? ' (paused)' : ''); },
    running() { return ['capturing', 'waiting_for_user'].includes(this.s.state); },
    manualLocked() { return this.s.state !== 'idle' || !!this.s.manual_busy || !!(this.s.job && this.s.job.running); },

    // --- live ----------------------------------------------------------------------
    home() { this.post('/api/stage/home').catch(() => {}); },
    moveTo(z) {
      z = Math.round(z * 10) / 10;
      this.zTarget = z;
      this.post('/api/stage/move', {z}).catch(() => {});
    },
    jog(step) {
      const z = Math.min(this.d.z_range[1], Math.max(this.d.z_range[0], (this.s.z_mm ?? this.d.z_range[0]) + step));
      this.moveTo(z);
    },
    async applyControls() {
      const r = await this.post('/api/camera/controls', {exposure_us: this.liveExposure, gain: this.liveGain}).catch(() => null);
      if (r) { this.liveExposure = r.exposure_us; this.liveGain = r.gain; }
    },
    expToSlider(e) {
      const [lo, hi] = this.d.exposure_limits;
      return 1000 * Math.log(e / lo) / Math.log(hi / lo);
    },
    sliderToExp(v) {
      const [lo, hi] = this.d.exposure_limits;
      return Math.round(lo * Math.pow(hi / lo, v / 1000));
    },
    async pollStats() {
      if (this.tab !== 'live' || document.hidden) return;
      try { this.stats = await this.get('/api/live/stats'); } catch (e) { return; }
      this.drawHistogram();
    },
    drawHistogram() {
      const c = this.$refs.hist;
      if (!c || !this.stats.histogram) return;
      const g = c.getContext('2d'), h = this.stats.histogram, W = c.width, H = c.height;
      const val = (v) => this.histLog ? Math.log10(1 + v) : v;
      const max = Math.max(...h.map(val)) || 1;
      g.clearRect(0, 0, W, H);
      const bw = W / h.length;
      h.forEach((v, i) => {
        g.fillStyle = i === h.length - 1 && v > 0 ? '#ff4d4f' : '#9aa3b2';
        const bh = H * val(v) / max;
        g.fillRect(i * bw, H - bh, Math.max(1, bw - 0.5), bh);
      });
      g.fillStyle = '#e5383b';                       // black level marker
      g.fillRect(W * this.stats.black_level / 1024, 0, 1, H);
    },

    // --- capture -------------------------------------------------------------------
    planFields() {
      const f = this.form;
      return {z_min: f.z_min, z_max: f.z_max, n_positions: f.n_positions, spacing: f.spacing};
    },
    loadPlan() {
      const plan = this.plans[this.form.plan_name];
      if (plan) Object.assign(this.form, {z_min: plan.z_min, z_max: plan.z_max, n_positions: plan.n_positions, spacing: plan.spacing});
      this.planName = this.form.plan_name;
      this.previewPlan();
    },
    planEdited() {
      const plan = this.plans[this.form.plan_name];
      if (plan && Object.entries(this.planFields()).some(([k, v]) => plan[k] !== v)) this.form.plan_name = '';
      clearTimeout(this._pt);
      this._pt = setTimeout(() => this.previewPlan(), 250);
    },
    async previewPlan() {
      const r = await fetch('/api/plans/preview', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(this.planFields())});
      const data = await r.json();
      this.planPreview = r.ok ? data : {error: data.detail};
    },
    spacingText() {
      const z = this.planPreview.actual_z || [];
      if (z.length < 2) return '—';
      const gaps = z.slice(1).map((v, i) => v - z[i]);
      return Math.min(...gaps).toFixed(3) + ' – ' + Math.max(...gaps).toFixed(3) + ' mm';
    },
    async savePlan() {
      const plan = await this.post('/api/plans', {...this.planFields(), name: this.planName}).catch(() => null);
      if (!plan) return;
      this.plans = await this.get('/api/plans');
      this.form.plan_name = plan.name;
    },
    async deletePlan() {
      if (!confirm('Delete plan "' + this.form.plan_name + '"?')) return;
      this.plans = await this.post('/api/plans/' + encodeURIComponent(this.form.plan_name), {}, 'DELETE');
      this.form.plan_name = '';
    },
    start() {
      if (this.form.x_mm === '' || this.form.y_mm === '') { this.toast = 'Enter the x and y offsets of the source.'; return; }
      this.post('/api/session/start', {...this.form, exposure_us: this.s.live.exposure_us, gain: this.s.live.gain}).catch(() => {});
    },
    abort() { if (confirm('Abort the running session? Frames taken so far are kept.')) this.post('/api/session/abort').catch(() => {}); },
    progressPct() { return this.p.positions_total ? 100 * this.p.positions_done / this.p.positions_total : 0; },
    eta() {
      const t = this.p.eta_s;
      if (t == null) return '—';
      const m = Math.floor(t / 60), sec = Math.round(t % 60);
      return (m ? m + ' min ' : '') + sec + ' s';
    },

    // --- model ---------------------------------------------------------------------
    async loadSessions() {
      this.sessions = await this.get('/api/sessions');
      this.selected = this.selected.filter(id => this.sessions.some(s => s.id === id));
    },
    selectedPlansDiffer() {
      const key = (s) => [s.z_min, s.z_max, s.n_positions, s.spacing].join();
      return new Set(this.sessions.filter(s => this.selected.includes(s.id)).map(key)).size > 1;
    },
    build() { this.built = null; this.post('/api/model/build', {sessions: this.selected, steps: this.steps}).catch(() => {}); },
    async generate() {
      const pts = this.points.map(p => ({x: +p.x, y: +p.y, z: +p.z, intensity: p.intensity === '' ? 1 : +p.intensity}));
      this.result = await this.post('/api/model/generate', {points: pts}).catch(() => ({}));
    },
    uploadPoints(ev) {
      const file = ev.target.files[0];
      if (!file) return;
      file.text().then(text => {
        const data = JSON.parse(text);
        if (!Array.isArray(data.points)) throw new Error('JSON must look like {"points": [{"x":..,"y":..,"z":..}]}');
        this.points = data.points.map(p => ({x: p.x, y: p.y, z: p.z, intensity: p.intensity ?? 1}));
      }).catch(e => { this.toast = 'Could not read points: ' + e.message; });
      ev.target.value = '';
    },

    // --- plots (small inline SVGs) --------------------------------------------------
    plot(series, xlabel, ylabel) {
      const W = 460, H = 170, L = 46, R = 10, T = 10, B = 30;
      const pts = series.flatMap(s => s.points);
      if (!pts.length) return '';
      let [x0, x1] = [Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0]))];
      let [y0, y1] = [Math.min(0, ...pts.map(p => p[1])), Math.max(0, ...pts.map(p => p[1]))];
      if (x1 === x0) { x0 -= 1; x1 += 1; }
      if (y1 === y0) { y0 -= 1; y1 += 1; }
      const pad = (y1 - y0) * 0.1; y0 -= pad; y1 += pad;
      const X = v => L + (W - L - R) * (v - x0) / (x1 - x0);
      const Y = v => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
      let svg = `<svg viewBox="0 0 ${W} ${H}" class="plot">`;
      svg += `<line x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}" class="zero"/>`;
      svg += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" class="frame"/>`;
      svg += `<text x="${L}" y="${H - 8}">${x0.toFixed(0)}</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${x1.toFixed(0)}</text>`;
      svg += `<text x="${(L + W) / 2}" y="${H - 8}" text-anchor="middle">${xlabel}</text>`;
      svg += `<text x="${L - 4}" y="${T + 10}" text-anchor="end">${y1.toPrecision(2)}</text><text x="${L - 4}" y="${H - B}" text-anchor="end">${y0.toPrecision(2)}</text>`;
      svg += `<text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">${ylabel}</text>`;
      series.forEach((s, k) => {
        if (s.line) svg += `<polyline fill="none" stroke="${s.color}" stroke-width="1.5" points="${s.points.map(p => X(p[0]) + ',' + Y(p[1])).join(' ')}"/>`;
        else s.points.forEach(p => { svg += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.5" fill="${s.color}"/>`; });
        svg += `<text x="${W - R - 4}" y="${T + 12 + 12 * k}" text-anchor="end" fill="${s.color}">${s.name}</text>`;
      });
      return svg + '</svg>';
    },
    jitterPlot(ses) {
      const dy = [], dx = [], ex = [];
      ses.pass_shifts.forEach(z => z.passes.forEach(q => {
        (q.excluded ? ex : dx).push([z.z_mm, q.dx]);
        if (!q.excluded) dy.push([z.z_mm, q.dy]);
      }));
      return this.plot([{name: 'dx', color: '#e5383b', points: dx}, {name: 'dy', color: '#9aa3b2', points: dy},
        {name: 'excluded', color: '#f5a524', points: ex}], 'z (mm)', 'px');
    },
    driftPlot(ses) {
      return this.plot([
        {name: 'source x', color: '#e5383b', line: true, points: ses.drift.map(d => [d.z_mm, d.source_dx_mm])},
        {name: 'source y', color: '#9aa3b2', line: true, points: ses.drift.map(d => [d.z_mm, d.source_dy_mm])}],
        'z (mm)', 'mm');
    },
  };
}
