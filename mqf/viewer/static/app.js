/* Més que un Frame · Viewer
   One playhead on the match timeline drives everything: the video (tactical = the match timeline,
   broadcast = linked by the sync maps), the 2D pitch, the tracking overlay, the lanes and the rail. */
'use strict';

const CODE = { H: 0, A: 1, D: 2, X: 3 };
const NAMES = ['H', 'A', 'D', 'X'];
const COL = { H: '#3d7eff', A: '#f04848', D: '#5c5c5c', none: '#151515', hatch: '#2c2c2c', diff: '#e5a93b' };
const LABEL_COL = [COL.H, COL.A, COL.D];
const STATE_TEXT = ['Not linked to the match here', 'Live', 'Close-up or no pitch view', 'Replay or another camera'];
const MISSING = -32768;

// timeline geometry (CSS px), shared by the label column and the canvases
const G = { ovHead: 18, ovRow: 10, ovGap: 3, ovDiff: 6, between: 16, lensRow: 30, lensGap: 8, diffRow: 14, ruler: 22 };

const $ = (id) => document.getElementById(id);
const S = {
  meta: null, fps: 25, n: 0, L: 105, W: 68,
  players: null, ball: null, slots: 0, team: null,
  sets: [], set: 'original', ballSrc: null, origBall: null, flags: null,   // tracking versions (Original / Fixed)
  lanes: [], truth: -1, compare: -1,
  videos: {}, view: 'tactical',
  frame: 0, playing: false, rate: 1, lensSec: 30,
  overlay: true, trails: true,
  diff: [], diffMin: 0, diffShown: [],
  hover: null, ready: false,
};
const store = {
  get(k, d) { try { const v = localStorage.getItem('mqf.viewer.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('mqf.viewer.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ---------------------------------------------------------------------------- formatting
const fmtInt = (v) => v.toLocaleString('en-US');
function fmtDur(frames) {
  const s = frames / S.fps;
  if (s < 10) return s.toFixed(1) + ' s';
  if (s < 60) return Math.round(s) + ' s';
  const m = Math.floor(s / 60), r = Math.round(s - 60 * m);
  return `${m}:${String(r).padStart(2, '0')}`;
}
function mmss(frames, withFrames) {
  const neg = frames < 0; frames = Math.abs(frames);
  const s = Math.floor(frames / S.fps), m = Math.floor(s / 60);
  let out = `${neg ? '−' : ''}${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  if (withFrames) out += '.' + String(frames % S.fps).padStart(2, '0');
  return out;
}
function halfOf(f) {
  const hs = S.meta.halves;
  for (const [h, a, b] of hs) if (f >= a && f <= b) return { half: h, start: a, end: b };
  return null;
}
function clockText(f, short) {
  const h = halfOf(f);
  const hs = S.meta.halves;
  if (h) return `${h.half === 1 ? '1st' : h.half === 2 ? '2nd' : h.half + 'th'}${short ? '' : ' half'} ${mmss(f - h.start, !short)}`;
  if (!hs.length) return mmss(f, !short);
  if (f < hs[0][1]) return short ? 'Pre-match' : 'Before kick-off';
  if (f > hs[hs.length - 1][2]) return short ? 'Post-match' : 'After the final whistle';
  return 'Halftime';
}
const labelName = (c) => c === 0 ? S.meta.home : c === 1 ? S.meta.away : c === 2 ? 'Dead ball' : 'No label';
function sw(c) { const s = document.createElement('span'); s.className = 'sw ' + (c === 'diff' ? 'diff' : NAMES[c]); return s; }

// ---------------------------------------------------------------------------- lanes
function rle(codes) {
  const starts = [], labs = [];
  let prev = -1;
  for (let i = 0; i < codes.length; i++) if (codes[i] !== prev) { starts.push(i); labs.push(codes[i]); prev = codes[i]; }
  return { starts: Int32Array.from(starts), labs: Uint8Array.from(labs) };
}
function makeLane(name, kind, source, codes, removable) {
  return { name, kind, source, codes, runs: rle(codes), removable: !!removable };
}
function runIndex(lane, f) {
  const st = lane.runs.starts;
  let lo = 0, hi = st.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (st[mid] <= f) lo = mid; else hi = mid - 1; }
  return lo;
}
function runBounds(lane, i) {
  const st = lane.runs.starts;
  return [st[i], i + 1 < st.length ? st[i + 1] : S.n, lane.runs.labs[i]];
}
function computeDiff() {
  S.diff = [];
  if (S.truth < 0 || S.compare < 0) return;
  const g = S.lanes[S.truth].codes, m = S.lanes[S.compare].codes;
  let cur = -1, s = 0;
  for (let f = 0; f <= S.n; f++) {
    const key = f < S.n && g[f] !== 3 && m[f] !== g[f] ? g[f] * 4 + m[f] : -1;
    if (key !== cur) {
      if (cur >= 0) S.diff.push({ s, e: f, g: cur >> 2, m: cur & 3 });
      cur = key; s = f;
    }
  }
}
function truthLabelled() {
  if (S.truth < 0) return null;
  const c = S.lanes[S.truth].codes;
  let a = -1, b = -1;
  for (let f = 0; f < S.n; f++) if (c[f] !== 3) { if (a < 0) a = f; b = f; }
  return a < 0 ? null : [a, b];
}

// ---------------------------------------------------------------------------- loading
async function fetchBin(name, onBytes) {
  const r = await fetch('/api/bin/' + name);
  if (!r.ok) throw new Error(`could not load ${name}`);
  const total = +r.headers.get('Content-Length') || 0;
  const reader = r.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got); got += value.length; onBytes(value.length);
  }
  return buf.buffer;
}

async function load() {
  const meta = await (await fetch('/api/meta')).json();
  S.meta = meta; S.fps = meta.fps; S.n = meta.n_frames; [S.L, S.W] = meta.pitch;
  document.title = `${meta.title} · Més que un Frame`;
  $('title').textContent = meta.title;
  $('gameKey').textContent = meta.key;

  // every tracking version (Original = as delivered; Fixed = cleaned tracklets + filled ball)
  const sets = meta.tracking || [{ key: 'original', label: 'Original', players: 'players', ball: 'ball',
    ball_source: null, n_slots: meta.n_slots, team_of_slot: meta.team_of_slot, note: 'As delivered' }];
  const wanted = [...meta.lanes.map((l) => l.blob)];
  const sizes = {};
  for (const t of sets) {
    wanted.push(t.players, t.ball); sizes[t.players] = S.n * t.n_slots * 4; sizes[t.ball] = S.n * 4;
    if (t.ball_source) wanted.push(t.ball_source);
    if (t.flags) { wanted.push(t.flags); sizes[t.flags] = S.n * t.n_slots; }
  }
  for (const v of meta.videos) for (const k of ['homography', 'to_match', 'from_match', 'state']) if (v[k]) wanted.push(v[k]);
  const total = wanted.reduce((a, k) => a + (sizes[k] || S.n), 0);
  let got = 0;
  const progress = (b) => {
    got += b;
    $('loadingBar').style.width = Math.min(100, (100 * got) / total).toFixed(1) + '%';
    $('loadingText').textContent = `Loading tracking and labels · ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`;
  };
  const bufs = {};
  await Promise.all(wanted.map(async (k) => { bufs[k] = await fetchBin(k, progress); }));

  S.sets = sets.map((t) => ({
    key: t.key, label: t.label, note: t.note || '', slots: t.n_slots, team: Uint8Array.from(t.team_of_slot),
    players: new Int16Array(bufs[t.players]), ball: new Int16Array(bufs[t.ball]),
    src: t.ball_source ? new Uint8Array(bufs[t.ball_source]) : null,
    flags: t.flags ? new Uint8Array(bufs[t.flags]) : null,
  }));
  S.origBall = S.sets[0].ball;
  useSet(S.sets[0].key, false);
  meta.lanes.forEach((l) => S.lanes.push(makeLane(l.name, l.kind, l.source, new Uint8Array(bufs[l.blob]))));
  S.truth = S.lanes.findIndex((l) => l.kind === 'truth');
  S.compare = S.lanes.findIndex((l) => l.kind === 'model');

  for (const v of meta.videos) {
    const el = $('vid-' + v.key);
    const entry = {
      meta: v, el, fps: v.fps || S.fps,
      hom: v.homography ? new Float32Array(bufs[v.homography]) : null,
      to: v.to_match ? new Int32Array(bufs[v.to_match]) : null,
      from: v.from_match ? new Int32Array(bufs[v.from_match]) : null,
      state: v.state ? new Uint8Array(bufs[v.state]) : null,
      nudge: v.key === 'broadcast' ? store.get(`nudge.${meta.key}`, 0) : 0,
      shown: -1, pending: null, guard: false, error: null, ready: false,
    };
    S.videos[v.key] = entry;
    if (el && v.available) {
      el.src = '/video/' + v.key;
      el.addEventListener('loadedmetadata', () => {
        entry.ready = true;
        // a frame-rate or file mix-up shows up as a length that does not match the tracking
        const frames = Math.round(el.duration * entry.fps), expected = v.n_frames;
        if (expected && Math.abs(frames - expected) > Math.max(25, 0.005 * expected)) {
          entry.lengthNote = `Video has ${fmtInt(frames)} frames at ${entry.fps} fps, its tracking ${fmtInt(expected)}: frames may not line up.`;
        }
        layoutStage(); seekVideo(entry);
      });
      el.addEventListener('error', () => {
        entry.error = 'This browser cannot play the video file. Chrome or Edge play H.264 MP4; other codecs may need re-encoding with ffmpeg.';
        render();
      });
      el.addEventListener('seeked', () => { if (entry.pending !== null) applySeek(entry); else entry.guard = false; });
      if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
        const cb = (now, md) => { onVideoFrame(entry, md.mediaTime); el.requestVideoFrameCallback(cb); };
        el.requestVideoFrameCallback(cb);
      } else {
        el.addEventListener('timeupdate', () => onVideoFrame(entry, el.currentTime));
      }
      el.addEventListener('pause', () => { if (S.playing && activeVideo() === entry) setPlaying(false); });
    }
  }
  $('loading').hidden = true;
  S.ready = true;
}

// ---------------------------------------------------------------------------- video <-> match
function activeVideo() { return S.view === 'pitch' ? null : S.videos[S.view] || null; }
function videoFrameFor(v, f) {
  if (!v.from) return f;
  const b = v.from[f];
  return b < 0 ? -1 : b + v.nudge;
}
function matchFrameFor(v, vf) {
  if (!v.to) return vf;
  const b = vf - v.nudge;
  return b >= 0 && b < v.to.length ? v.to[b] : -1;
}
function seekVideo(v) {
  if (!v || !v.ready) return;
  const vf = videoFrameFor(v, S.frame);
  if (vf < 0) return;
  v.pending = Math.min(vf, (v.meta.n_frames || Infinity) - 1);
  if (S.playing) v.guard = true;          // ignore frames presented before this seek lands
  if (!v.el.seeking) applySeek(v);
}
function applySeek(v) {
  const vf = v.pending; v.pending = null;
  v.el.currentTime = (vf + 0.5) / v.fps;
}
function onVideoFrame(v, mediaTime) {
  const vf = Math.round(mediaTime * v.fps);
  v.shown = vf;
  if (S.playing && activeVideo() === v && !v.guard) {
    const f = matchFrameFor(v, vf);
    if (f >= 0 && f < S.n) S.frame = f;
  }
  requestRender();
}

// ---------------------------------------------------------------------------- playback
let clockTs = null, clockAcc = 0;
function setPlaying(on) {
  S.playing = on;
  $('playBtn').classList.toggle('playing', on);
  $('playLabel').textContent = on ? 'Pause' : 'Play';
  $('playBtn').setAttribute('aria-label', on ? 'Pause (Space)' : 'Play (Space)');
  const v = activeVideo();
  if (videoUsable(v)) {
    if (on) { v.el.playbackRate = S.rate; v.el.play().catch(() => setPlaying(false)); } else { v.el.pause(); seekVideo(v); }
  } else if (on) {
    clockTs = null; clockAcc = 0; requestAnimationFrame(tick);
  }
  if (!on) writeHash();
  requestRender();
}
function videoUsable(v) { return v && v.meta.available && !v.error && v.ready && videoFrameFor(v, S.frame) >= 0; }
function tick(ts) {
  if (!S.playing) return;
  const v = activeVideo();
  if (videoUsable(v)) { seekVideo(v); v.el.playbackRate = S.rate; v.el.play().catch(() => setPlaying(false)); return; }
  if (clockTs !== null) {
    clockAcc += ((ts - clockTs) / 1000) * S.fps * S.rate;
    const step = Math.floor(clockAcc);
    if (step) { clockAcc -= step; S.frame = Math.min(S.n - 1, S.frame + step); if (S.frame >= S.n - 1) setPlaying(false); }
  }
  clockTs = ts;
  render();
  requestAnimationFrame(tick);
}
function seek(f, opts = {}) {
  f = Math.max(0, Math.min(S.n - 1, Math.round(f)));
  if (S.playing && opts.pause) setPlaying(false);
  S.frame = f;
  seekVideo(activeVideo());
  if (!opts.noHash) writeHashSoon();
  requestRender();
}
function setView(view) {
  const was = S.playing;
  if (was) setPlaying(false);
  const old = activeVideo();
  if (old) old.el.pause();
  S.view = view;
  document.querySelectorAll('#viewSeg button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === view)));
  layoutStage();
  seekVideo(activeVideo());
  writeHash();
  requestRender();
  if (was) setTimeout(() => setPlaying(true), 60);
}
function setRate(r) {
  S.rate = r; store.set('rate', r);
  document.querySelectorAll('#speedSeg button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.rate === r)));
  const v = activeVideo();
  if (v) v.el.playbackRate = r;
}
function setLens(sec) {
  S.lensSec = sec; store.set('lens', sec);
  document.querySelectorAll('#lensSeg button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.lens === sec)));
  requestRender();
}

// ---------------------------------------------------------------------------- hash (come back to a moment)
let hashTimer = null;
function writeHash() { try { history.replaceState(null, '', `#f=${S.frame}&v=${S.view}&t=${S.set}`); } catch { /* file: */ } }
function writeHashSoon() { clearTimeout(hashTimer); hashTimer = setTimeout(writeHash, 300); }
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const out = {};
  if (p.has('f')) out.f = +p.get('f');
  if (p.has('v')) out.v = p.get('v');
  if (p.has('t')) out.t = p.get('t');
  return out;
}

// ---------------------------------------------------------------------------- stage layout
function layoutStage() {
  const stage = $('stage');
  const W = stage.clientWidth, H = stage.clientHeight;
  for (const v of Object.values(S.videos)) {
    if (!v.el) continue;
    const on = S.view === v.meta.key && v.meta.available && !v.error;
    v.el.classList.toggle('on', on);
    if (!on) continue;
    const vw = v.el.videoWidth || 16, vh = v.el.videoHeight || 9;
    const s = Math.min(W / vw, H / vh);
    const w = Math.round(vw * s), h = Math.round(vh * s);
    const x = Math.round((W - w) / 2), y = Math.round((H - h) / 2);
    Object.assign(v.el.style, { width: w + 'px', height: h + 'px', left: x + 'px', top: y + 'px' });
    v.box = { x, y, w, h, scale: s };
  }
  $('pitch').classList.toggle('on', S.view === 'pitch');
  sizeCanvas($('pitch'), W, H);
  pitchCache = null;
  const av = activeVideo();
  const ov = $('overlay');
  if (av && av.box) {
    Object.assign(ov.style, { left: av.box.x + 'px', top: av.box.y + 'px' });
    sizeCanvas(ov, av.box.w, av.box.h);
  }
  requestRender();
}
function sizeCanvas(c, w, h) {
  const dpr = window.devicePixelRatio || 1;
  c.style.width = w + 'px'; c.style.height = h + 'px';
  const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// ---------------------------------------------------------------------------- tracking access
const px = (f, k) => S.players[(f * S.slots + k) * 2];
const py = (f, k) => S.players[(f * S.slots + k) * 2 + 1];
function ballAt(f) {
  const x = S.ball[f * 2], y = S.ball[f * 2 + 1];
  return x === MISSING ? null : [x / 100, y / 100];
}
function lastBall(f, maxBack) {
  for (let g = f; g >= Math.max(0, f - maxBack); g--) { const b = ballAt(g); if (b) return { b, ago: f - g }; }
  return null;
}
const BALL_SRC = ['Tracked', 'Interpolated', 'Model', 'Not filled'];
// what the cleaner did to a point of the Fixed set (mqf.tracks.FLAG_*): 1 team repaired,
// 2 keeper takeover (Sigmar's rule), 3 keeper drawn at his saved place (not tracked)
const FLAG = { team: 1, takeover: 2, fill: 3 };
const flagAt = (f, k) => (S.flags ? S.flags[f * S.slots + k] : 0);
function drawPlayer(ctx, u, v, r, k, fl) {
  ctx.beginPath(); ctx.arc(u, v, r, 0, 7);
  const col = S.team[k] === 0 ? COL.H : COL.A;
  if (fl === FLAG.fill) {                            // not tracked: hollow, dashed, in the team colour
    ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fill();
    ctx.setLineDash([3, 2]); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]);
    ctx.lineWidth = 1.5;
    return;
  }
  ctx.fillStyle = col; ctx.fill();
  ctx.strokeStyle = '#000'; ctx.stroke();
  if (fl === FLAG.team || fl === FLAG.takeover) {   // team repaired: a ring, amber for the keeper takeover
    ctx.beginPath(); ctx.arc(u, v, r + 3, 0, 7);
    ctx.strokeStyle = fl === FLAG.takeover ? COL.diff : '#fff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineWidth = 1.5;
  }
}
function ballSource(f) { return S.ballSrc ? S.ballSrc[f] : 0; }
function gapAround(f) {                    // length of the gap in the DELIVERED ball that frame f sits in
  const B = S.origBall, lim = 20 * 60 * S.fps;
  if (B[f * 2] !== MISSING) return 0;
  let a = f, b = f;
  while (a > 0 && f - a < lim && B[(a - 1) * 2] === MISSING) a--;
  while (b < S.n - 1 && b - f < lim && B[(b + 1) * 2] === MISSING) b++;
  return b - a + 1;
}
function useSet(key, persist = true) {
  const t = S.sets.find((x) => x.key === key) || S.sets[0];
  S.set = t.key; S.players = t.players; S.ball = t.ball; S.slots = t.slots; S.team = t.team; S.ballSrc = t.src;
  S.flags = t.flags;
  document.querySelectorAll('#trkSeg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.set === t.key)));
  if (persist) { store.set(`set.${S.meta.key}`, t.key); writeHashSoon(); }
  requestRender();
}
function cycleSet() {
  if (S.sets.length < 2) return;
  const i = S.sets.findIndex((t) => t.key === S.set);
  useSet(S.sets[(i + 1) % S.sets.length].key);
}
function buildTrackingSeg() {
  const seg = $('trkSeg');
  seg.textContent = '';
  seg.hidden = S.sets.length < 2;
  for (const t of S.sets) {
    const b = document.createElement('button');
    b.dataset.set = t.key; b.textContent = t.label;
    b.title = `${t.note || t.label} (T)`;
    b.setAttribute('aria-pressed', String(t.key === S.set));
    b.addEventListener('click', () => useSet(t.key));
    seg.appendChild(b);
  }
}

// ---------------------------------------------------------------------------- the stage: overlay + pitch
function project(H, o, xm, ym) {
  const x = xm / S.L, y = ym / S.W;
  const w = H[o + 6] * x + H[o + 7] * y + H[o + 8];
  return [(H[o] * x + H[o + 1] * y + H[o + 2]) / w, (H[o + 3] * x + H[o + 4] * y + H[o + 5]) / w];
}
function drawOverlay() {
  const c = $('overlay');
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const v = activeVideo();
  if (!S.overlay || !v || !v.box || !v.hom || v.error) return;
  const vf = v.shown >= 0 ? v.shown : videoFrameFor(v, S.frame);      // the picture on screen
  if (vf < 0 || vf * 9 + 8 >= v.hom.length || !Number.isFinite(v.hom[vf * 9])) return;
  const f = matchFrameFor(v, vf);
  if (f < 0 || f >= S.n) return;
  if (v.state && v.state[vf - v.nudge] !== 1) return;     // a replay or close-up: live tracking would mislead
  const o = vf * 9, s = v.box.scale;
  const r = Math.max(4, Math.min(7, v.box.w / 180));
  ctx.lineWidth = 1.5;
  for (let k = 0; k < S.slots; k++) {
    const x = px(f, k);
    if (x === MISSING) continue;
    const [u, w] = project(v.hom, o, x / 100, py(f, k) / 100);
    if (!Number.isFinite(u)) continue;
    drawPlayer(ctx, u * s, w * s, r, k, flagAt(f, k));
  }
  const b = ballAt(f);
  if (b) {
    const src = ballSource(f);                       // a filled ball: dashed, amber when the model drew it
    const [u, w] = project(v.hom, o, b[0], b[1]);
    ctx.beginPath(); ctx.arc(u * s, w * s, r + 2, 0, 7);
    ctx.strokeStyle = src === 2 ? COL.diff : '#fff'; ctx.lineWidth = 2;
    if (src) ctx.setLineDash([4, 3]);
    ctx.stroke(); ctx.setLineDash([]);
  }
}

let pitchCache = null;
function pitchGeom() {
  const c = $('pitch');
  const W = c.clientWidth, H = c.clientHeight, m = 28;
  const s = Math.min((W - 2 * m) / S.L, (H - 2 * m) / S.W);
  return { W, H, s, ox: (W - S.L * s) / 2, oy: (H - S.W * s) / 2 };
}
function toScreen(g, x, y) {
  const o = S.meta.orientation || {};
  return [g.ox + (o.flip_x ? S.L - x : x) * g.s, g.oy + (o.flip_y ? S.W - y : y) * g.s];
}
function drawPitchLines(g) {
  const off = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  off.width = Math.round(g.W * dpr); off.height = Math.round(g.H * dpr);
  const ctx = off.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b0b0b';
  ctx.fillRect(g.ox, g.oy, S.L * g.s, S.W * g.s);
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5;
  const L = S.L, W = S.W;
  const poly = (pts, close) => { ctx.beginPath(); pts.forEach(([x, y], i) => { const [u, v] = toScreen(g, x, y); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); if (close) ctx.closePath(); ctx.stroke(); };
  poly([[0, 0], [L, 0], [L, W], [0, W]], true);
  poly([[L / 2, 0], [L / 2, W]]);
  const circ = []; for (let a = 0; a <= 64; a++) circ.push([L / 2 + 9.15 * Math.cos(a / 64 * 2 * Math.PI), W / 2 + 9.15 * Math.sin(a / 64 * 2 * Math.PI)]);
  poly(circ);
  for (const [x0, sg] of [[0, 1], [L, -1]]) {
    poly([[x0, W / 2 - 20.16], [x0 + sg * 16.5, W / 2 - 20.16], [x0 + sg * 16.5, W / 2 + 20.16], [x0, W / 2 + 20.16]]);
    poly([[x0, W / 2 - 9.16], [x0 + sg * 5.5, W / 2 - 9.16], [x0 + sg * 5.5, W / 2 + 9.16], [x0, W / 2 + 9.16]]);
    const arc = []; const cx = x0 + sg * 11;
    for (let a = -53; a <= 53; a += 4) { const t = a * Math.PI / 180; arc.push([cx + sg * 9.15 * Math.cos(t), W / 2 + 9.15 * Math.sin(t)]); }
    poly(arc.filter(([x]) => sg > 0 ? x > x0 + 16.5 : x < x0 - 16.5));
  }
  return off;
}
function drawPitch() {
  if (S.view !== 'pitch') return;
  const c = $('pitch');
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const g = pitchGeom();
  if (!pitchCache) pitchCache = drawPitchLines(g);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(pitchCache, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const f = S.frame;
  const r = Math.max(4, Math.min(8, g.s * 0.8));
  if (S.trails) {
    ctx.lineWidth = 1.5; ctx.globalAlpha = 0.4;
    for (let k = 0; k < S.slots; k++) {
      ctx.beginPath(); let pen = false;
      for (let t = Math.max(0, f - S.fps); t <= f; t++) {
        const x = px(t, k);
        if (x === MISSING) { pen = false; continue; }
        const [u, v] = toScreen(g, x / 100, py(t, k) / 100);
        pen ? ctx.lineTo(u, v) : ctx.moveTo(u, v); pen = true;
      }
      ctx.strokeStyle = S.team[k] === 0 ? COL.H : COL.A; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  ctx.lineWidth = 1.5;
  for (let k = 0; k < S.slots; k++) {
    const x = px(f, k);
    if (x === MISSING) continue;
    const [u, v] = toScreen(g, x / 100, py(f, k) / 100);
    drawPlayer(ctx, u, v, r, k, flagAt(f, k));
  }
  const b = ballAt(f);
  const src = ballSource(f);
  if (b && src) {                                     // filled, not tracked: a ring, amber for the model
    const [u, v] = toScreen(g, b[0], b[1]);
    const col = src === 2 ? COL.diff : '#d9d9d9';
    ctx.beginPath(); ctx.arc(u, v, r * 0.75, 0, 7); ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fill();
    ctx.setLineDash([3, 2]); ctx.lineWidth = 2; ctx.strokeStyle = col; ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.font = '12px Inter, "Segoe UI", sans-serif';
    ctx.fillText(src === 2 ? 'model' : 'interpolated', u + r + 4, v + 4);
  } else if (b) {
    const [u, v] = toScreen(g, b[0], b[1]);
    ctx.beginPath(); ctx.arc(u, v, r * 0.62, 0, 7); ctx.fillStyle = '#fff'; ctx.fill();
    ctx.strokeStyle = '#000'; ctx.stroke();
  } else {
    const lb = lastBall(f, 10 * S.fps);
    if (lb) {
      const [u, v] = toScreen(g, lb.b[0], lb.b[1]);
      ctx.setLineDash([3, 3]); ctx.strokeStyle = '#949494';
      ctx.beginPath(); ctx.arc(u, v, r * 0.9, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#949494'; ctx.font = '12px Inter, "Segoe UI", sans-serif';
      ctx.fillText(`ball lost ${fmtDur(lb.ago)}`, u + r + 4, v + 4);
    }
  }
}

// ---------------------------------------------------------------------------- stage messages
function stageStatus() {
  const chip = $('stageChip'), msg = $('stageMsg'), btn = $('stageMsgBtn');
  let chipText = '', chipWarn = false, msgText = '', showBtn = false;
  const v = activeVideo();
  if (v) {
    if (!v.meta.available) { msgText = v.meta.note || `No ${v.meta.label.toLowerCase()} video for this game.`; showBtn = true; }
    else if (v.error) { msgText = v.error; showBtn = true; }
    else if (v.lengthNote && !v.from) { chipText = v.lengthNote; chipWarn = true; }
    else if (v.from) {
      const vf = S.playing ? v.shown : videoFrameFor(v, S.frame);
      if (!S.playing && vf < 0) { msgText = 'The broadcast has no picture for this moment of the match (outside the recording, halftime or a recording cut).'; showBtn = true; }
      else if (vf >= 0 && v.state) {
        const st = v.state[Math.min(vf - v.nudge, v.state.length - 1)] ?? 0;
        if (st !== 1) { chipText = STATE_TEXT[st]; chipWarn = true; }
        else if (v.lengthNote) { chipText = v.lengthNote; chipWarn = true; }
      }
    }
  }
  chip.hidden = !chipText; chip.textContent = chipText; chip.classList.toggle('warn', chipWarn);
  msg.hidden = !msgText; $('stageMsgText').textContent = msgText; btn.hidden = !showBtn;
}

// ---------------------------------------------------------------------------- timeline
let hatch = null;
function hatchPattern(ctx) {
  if (hatch) return hatch;
  const t = document.createElement('canvas'); t.width = t.height = 6;
  const c = t.getContext('2d');
  c.fillStyle = COL.none; c.fillRect(0, 0, 6, 6);
  c.strokeStyle = COL.hatch; c.lineWidth = 1;
  c.beginPath(); c.moveTo(-1, 7); c.lineTo(7, -1); c.stroke();
  hatch = ctx.createPattern(t, 'repeat');
  return hatch;
}
const nLanes = () => S.lanes.length;
const ovHeight = () => G.ovHead + nLanes() * (G.ovRow + G.ovGap) + G.ovDiff;
const lensHeight = () => nLanes() * (G.lensRow + G.lensGap) + G.diffRow + G.lensGap + G.ruler;

function buildTimelineLabels() {
  const col = $('tlLabels');
  col.innerHTML = '';
  const ov = document.createElement('div');
  ov.className = 'tl-label small-text';
  ov.style.height = ovHeight() + 'px';
  ov.style.marginBottom = G.between + 'px';
  ov.innerHTML = '<span>Whole match</span>';
  col.appendChild(ov);
  S.lanes.forEach((lane, i) => {
    const el = document.createElement('div');
    el.className = 'tl-label lane' + (i === S.truth ? ' truth' : '') + (i === S.compare ? ' compared' : '');
    el.style.height = G.lensRow + 'px'; el.style.marginBottom = G.lensGap + 'px';
    el.title = lane.kind === 'model' ? 'Compare this lane with the ground truth' : 'Ground truth';
    const name = document.createElement('span'); name.className = 'name';
    name.append(document.createTextNode(lane.name));
    if (lane.removable) {
      const x = document.createElement('button'); x.className = 'remove'; x.textContent = '×'; x.setAttribute('aria-label', `Remove ${lane.name}`);
      x.addEventListener('click', (e) => { e.stopPropagation(); removeLane(i); });
      name.append(x);
    }
    const sub = document.createElement('span'); sub.className = 'sub';
    sub.textContent = lane.kind === 'truth' ? lane.source : (i === S.compare ? `Compared · ${lane.source}` : lane.source);
    el.append(name, sub);
    if (lane.kind === 'model') el.addEventListener('click', () => { S.compare = i; refreshLanes(); });
    col.appendChild(el);
  });
  const d = document.createElement('div');
  d.className = 'tl-label small-text';
  d.style.height = G.diffRow + 'px'; d.style.marginBottom = (G.lensGap + G.ruler) + 'px';
  d.innerHTML = '<span>Disagreement</span>';
  col.appendChild(d);
  $('tlEmpty').hidden = nLanes() > 0;
  $('tlEmpty').textContent = 'No lanes yet. Drop a frame,label CSV here, or run python -m mqf.make_submission.';
}

let ovCache = null;
function layoutTimeline() {
  const w = $('overview').parentElement.clientWidth;
  sizeCanvas($('overview'), w, ovHeight());
  sizeCanvas($('lens'), w, lensHeight());
  $('lens').style.marginTop = G.between + 'px';
  ovCache = null;
  requestRender();
}
function buildOverviewCache() {
  const c = $('overview');
  const W = c.clientWidth, Hh = ovHeight();
  const dpr = window.devicePixelRatio || 1;
  const off = document.createElement('canvas');
  off.width = Math.round(W * dpr); off.height = Math.round(Hh * dpr);
  const ctx = off.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cols = Math.max(1, Math.floor(W));
  const fpp = S.n / cols;
  // half blocks: labels above, a faint step for what is not graded
  ctx.font = '12px Inter, "Segoe UI", sans-serif'; ctx.fillStyle = '#949494'; ctx.textBaseline = 'top';
  for (const [h, a, b] of S.meta.halves) {
    const x0 = (a / S.n) * W, x1 = ((b + 1) / S.n) * W;
    ctx.fillText(h === 1 ? '1st half' : h === 2 ? '2nd half' : `Half ${h}`, x0 + 2, 0);
    ctx.fillStyle = '#2a2a2a'; ctx.fillRect(x0, 14, Math.max(1, x1 - x0), 2); ctx.fillStyle = '#949494';
  }
  S.lanes.forEach((lane, i) => {
    const y = G.ovHead + i * (G.ovRow + G.ovGap);
    ctx.fillStyle = hatchPattern(ctx); ctx.fillRect(0, y, W, G.ovRow);
    const counts = new Float32Array(cols * 4);
    const st = lane.runs.starts, lb = lane.runs.labs;
    for (let r = 0; r < st.length; r++) {
      const a = st[r], b = r + 1 < st.length ? st[r + 1] : S.n, k = lb[r];
      let ca = Math.floor(a / fpp), cb = Math.min(cols - 1, Math.floor((b - 1) / fpp));
      for (let cc = ca; cc <= cb; cc++) {
        const lo = Math.max(a, cc * fpp), hi = Math.min(b, (cc + 1) * fpp);
        if (hi > lo) counts[cc * 4 + k] += hi - lo;
      }
    }
    for (let cc = 0; cc < cols; cc++) {
      const tot = fpp;
      let yy = y;
      for (const k of [0, 2, 1]) {
        const h = (counts[cc * 4 + k] / tot) * G.ovRow;
        if (h <= 0) continue;
        ctx.fillStyle = LABEL_COL[k]; ctx.fillRect(cc, yy, 1, h); yy += h;
      }
    }
  });
  // disagreement density
  const yd = G.ovHead + nLanes() * (G.ovRow + G.ovGap);
  const dens = new Float32Array(cols);
  for (const d of S.diff) {
    let ca = Math.floor(d.s / fpp), cb = Math.min(cols - 1, Math.floor((d.e - 1) / fpp));
    for (let cc = ca; cc <= cb; cc++) dens[cc] += Math.min(d.e, (cc + 1) * fpp) - Math.max(d.s, cc * fpp);
  }
  ctx.fillStyle = COL.diff;
  for (let cc = 0; cc < cols; cc++) if (dens[cc] > 0) {
    const h = Math.max(1.5, Math.sqrt(dens[cc] / fpp) * G.ovDiff);
    ctx.fillRect(cc, yd + G.ovDiff - h, 1, h);
  }
  ovCache = off;
}
function drawOverview() {
  const c = $('overview');
  const ctx = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  if (!ovCache) buildOverviewCache();
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height); ctx.drawImage(ovCache, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = c.clientWidth, Hh = ovHeight();
  const span = S.lensSec * S.fps;
  const x0 = ((S.frame - span / 2) / S.n) * W, x1 = ((S.frame + span / 2) / S.n) * W;
  ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(x0) + 0.5, G.ovHead - 1.5, Math.max(3, Math.round(x1 - x0)), Hh - G.ovHead + 2);
  const xp = Math.round((S.frame / S.n) * W) + 0.5;
  ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.moveTo(xp, G.ovHead - 2); ctx.lineTo(xp, Hh); ctx.stroke();
}
function lensWindow() {
  const W = $('lens').clientWidth;
  const span = S.lensSec * S.fps;
  return { W, span, f0: S.frame - span / 2, ppf: W / span };
}
function drawLens() {
  const c = $('lens');
  const ctx = c.getContext('2d');
  const { W, span, f0, ppf } = lensWindow();
  const Hh = lensHeight();
  ctx.clearRect(0, 0, W, Hh);
  const fx = (f) => (f - f0) * ppf;
  S.lanes.forEach((lane, i) => {
    const y = i * (G.lensRow + G.lensGap);
    ctx.fillStyle = hatchPattern(ctx); ctx.fillRect(0, y, W, G.lensRow);
    let r = runIndex(lane, Math.max(0, Math.floor(f0)));
    const st = lane.runs.starts;
    for (; r < st.length && st[r] < f0 + span; r++) {
      const [a, b, k] = runBounds(lane, r);
      if (k === 3) continue;
      const xa = Math.max(0, fx(a)), xb = Math.min(W, fx(b));
      ctx.fillStyle = LABEL_COL[k]; ctx.fillRect(xa, y, Math.max(1, xb - xa), G.lensRow);
      if (a > f0 && ppf * (b - a) > 3) { ctx.fillStyle = '#111'; ctx.fillRect(xa, y, 1, G.lensRow); }
    }
    if (f0 < 0) ctx.clearRect(0, y, fx(0), G.lensRow);                 // before the recording
    if (f0 + span > S.n) ctx.clearRect(fx(S.n), y, W, G.lensRow);    // after it
  });
  // disagreements
  const yd = nLanes() * (G.lensRow + G.lensGap);
  ctx.fillStyle = '#151515'; ctx.fillRect(0, yd + G.diffRow / 2 - 0.5, W, 1);
  ctx.fillStyle = COL.diff;
  for (const d of diffsIn(f0, f0 + span)) {
    const xa = Math.max(0, fx(d.s)), xb = Math.min(W, fx(d.e));
    ctx.fillRect(xa, yd + 2, Math.max(1.5, xb - xa), G.diffRow - 4);
  }
  // ruler: ticks on the half clock
  const yr = yd + G.diffRow + G.lensGap;
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const stepS = steps.find((s) => s * S.fps * ppf >= 84) || 600;
  const h = halfOf(Math.round(S.frame));
  const base = h ? h.start : 0;
  const stepF = stepS * S.fps;
  ctx.font = '12px Inter, "Segoe UI", sans-serif'; ctx.textBaseline = 'top'; ctx.fillStyle = '#949494';
  for (let t = Math.ceil((f0 - base) / stepF) * stepF + base; t < f0 + span; t += stepF) {
    const x = Math.round(fx(t)) + 0.5;
    ctx.fillRect(x - 0.5, yr, 1, 5);
    const label = mmss(t - base, false);
    const tw = ctx.measureText(label).width;
    if (x - tw / 2 > 2 && x + tw / 2 < W - 2) ctx.fillText(label, x - tw / 2, yr + 7);
  }
  // hover line
  if (S.hover && S.hover.canvas === 'lens') {
    const x = Math.round(fx(S.hover.f)) + 0.5;
    ctx.fillStyle = '#949494'; ctx.fillRect(x - 0.5, 0, 1, yd + G.diffRow);
  }
  // playhead
  const xp = Math.round(W / 2) + 0.5;
  ctx.fillStyle = '#fff'; ctx.fillRect(xp - 0.5, 0, 1, yd + G.diffRow + 4);
}
function diffsIn(a, b) {
  const out = [];
  let lo = 0, hi = S.diff.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (S.diff[mid].e <= a) lo = mid + 1; else hi = mid; }
  for (let i = lo; i < S.diff.length && S.diff[i].s < b; i++) out.push(S.diff[i]);
  return out;
}

// ---------------------------------------------------------------------------- rail
function renderRail() {
  const f = S.frame;
  $('frameReadout').textContent = 'Frame ' + fmtInt(f);
  $('clockReadout').textContent = clockText(f, false);

  const v = activeVideo();
  const rv = $('railVideo');
  if (v && v.from && v.meta.available) {
    const vf = S.playing ? v.shown : videoFrameFor(v, f);
    const st = vf >= 0 && v.state ? v.state[Math.min(vf - v.nudge, v.state.length - 1)] : 0;
    rv.hidden = false;
    rv.textContent = vf >= 0 ? `${v.meta.label} frame ${fmtInt(vf)} · ${(STATE_TEXT[st] ?? '').toLowerCase()}` : `No ${v.meta.label.toLowerCase()} picture for this match frame`;
  } else rv.hidden = true;

  // labels here
  const list = $('hereList');
  list.textContent = '';
  const g = S.truth >= 0 ? S.lanes[S.truth] : null;
  const gi = g ? runIndex(g, f) : -1;
  const [gs, ge, gk] = g ? runBounds(g, gi) : [0, 0, 3];
  S.lanes.forEach((lane, i) => {
    const li = document.createElement('li');
    const name = document.createElement('span'); name.className = 'name'; name.textContent = lane.name;
    const ri = runIndex(lane, f);
    const [a, b, k] = runBounds(lane, ri);
    const val = document.createElement('span'); val.className = 'val';
    val.append(sw(k), document.createTextNode(labelName(k)));
    const sub = document.createElement('span'); sub.className = 'sub';
    if (k === 3) sub.textContent = lane.kind === 'truth' ? 'Outside the labelled window' : 'Abstains here';
    else {
      let t = `${k === 2 ? 'Stoppage' : 'Spell'} of ${fmtDur(b - a)}, ${fmtDur(f - a)} in`;
      if (lane.kind === 'model' && g && gk !== 3) {
        if (k !== gk) { t = `Disagrees: truth says ${labelName(gk)}`; sub.classList.add('warn'); }
        else {
          const ds = a - gs, de = b - ge;
          const part = (d, w) => d === 0 ? `${w} on the frame` : `${w} ${Math.abs(d)} f ${d > 0 ? 'late' : 'early'}`;
          t = `${part(ds, 'Starts')}, ${part(de, 'ends').toLowerCase()}`;
        }
      }
      sub.textContent = t;
    }
    li.append(name, val, sub);
    list.appendChild(li);
  });
  if (!S.lanes.length) { const li = document.createElement('li'); li.className = 'muted'; li.textContent = 'No lanes loaded.'; list.appendChild(li); }

  // tracking here
  const cur = S.sets.find((t) => t.key === S.set);
  $('trkSet').textContent = cur ? cur.label : 'Original';
  $('trkSet').title = cur ? cur.note : '';
  const b = ballAt(f), src = ballSource(f), bEl = $('trkBall');
  bEl.classList.toggle('imputed', !!(b && src));
  if (b && src) bEl.textContent = `${BALL_SRC[src]} · gap ${fmtDur(gapAround(f))}`;
  else if (b) bEl.textContent = 'Tracked';
  else { const lb = lastBall(f, 60 * S.fps); bEl.textContent = lb ? `Lost for ${fmtDur(lb.ago)}` : 'Not tracked'; }
  let h = 0, a = 0, rep = 0, take = 0, fill = 0;
  for (let k = 0; k < S.slots; k++) {
    if (px(f, k) === MISSING) continue;
    S.team[k] === 0 ? h++ : a++;
    const fl = flagAt(f, k);
    if (fl === FLAG.team) rep++; else if (fl === FLAG.takeover) take++; else if (fl === FLAG.fill) fill++;
  }
  $('trkHome').textContent = h;
  $('trkAway').textContent = a;
  $('trkFixRow').hidden = !S.flags;
  $('trkFix').textContent = !S.flags ? '' : (rep + take + fill === 0 ? 'none'
    : [rep && `${rep} team`, take && `${take} keeper takeover`, fill && `${fill} keeper at saved place`].filter(Boolean).join(' · '));

  // sync block
  const bv = S.videos.broadcast;
  $('syncBlock').hidden = !(S.view === 'broadcast' && bv && bv.from && bv.meta.available && !bv.error);
  if (!$('syncBlock').hidden) {
    const s = bv.meta.summary || {};
    $('syncBlock').title = s.linked_share !== undefined
      ? `Linked from the two tracking files; the broadcast shows live play on ${(100 * s.match_live_share).toFixed(0)}% of match frames. Nudge if the tracking overlay drifts from the players.`
      : 'Linked from the two tracking files.';
    $('nudgeVal').textContent = (bv.nudge > 0 ? '+' : bv.nudge < 0 ? '−' : '±') + Math.abs(bv.nudge) + ' f';
  }
  markCurrentDiff();
}

function renderDiffList() {
  const ol = $('diffList');
  ol.textContent = '';
  const sum = $('diffSummary');
  if (S.truth < 0) { sum.textContent = 'Load a ground-truth lane to list where the model disagrees.'; S.diffShown = []; return; }
  if (S.compare < 0) { sum.textContent = 'Load a model lane to compare.'; S.diffShown = []; return; }
  const shown = S.diff.filter((d) => d.e - d.s >= S.diffMin);
  S.diffShown = shown;
  const total = shown.reduce((a, d) => a + d.e - d.s, 0);
  const truth = S.lanes[S.truth];
  const scope = /public/i.test(truth.source) ? ' Public window: check correctness here, do not tune on it.' : '';
  sum.textContent = shown.length
    ? `${fmtInt(shown.length)} stretches, ${fmtDur(total)} in all, ${S.lanes[S.compare].name} against the truth.${scope}`
    : `No disagreements${S.diffMin ? ' of 1 s or more' : ''} where the truth has labels.${scope}`;
  const frag = document.createDocumentFragment();
  const cap = 1500;
  shown.slice(0, cap).forEach((d, i) => {
    const li = document.createElement('li');
    li.dataset.i = i;
    const b = document.createElement('button');
    const when = document.createElement('span'); when.className = 'when num'; when.textContent = clockText(d.s, true);
    const what = document.createElement('span'); what.className = 'what';
    const arrow = document.createElement('span'); arrow.className = 'arrow'; arrow.textContent = '→';
    const tg = document.createElement('span'); tg.className = 't'; tg.textContent = short(d.g);
    const tm = document.createElement('span'); tm.className = 't'; tm.textContent = short(d.m);
    what.append(sw(d.g), tg, arrow, sw(d.m), tm);
    const dur = document.createElement('span'); dur.className = 'dur num'; dur.textContent = fmtDur(d.e - d.s);
    b.append(when, what, dur);
    b.title = `${labelName(d.g)} in the truth, ${labelName(d.m)} in the model · frames ${fmtInt(d.s)}–${fmtInt(d.e - 1)}`;
    b.addEventListener('click', () => seek(d.s - Math.round(S.fps / 2)));
    li.appendChild(b);
    frag.appendChild(li);
  });
  if (shown.length > cap) { const li = document.createElement('li'); li.className = 'more'; li.textContent = `${fmtInt(shown.length - cap)} more not listed. Filter to 1 s or more.`; frag.appendChild(li); }
  ol.appendChild(frag);
  lastCurrent = -2;
}
const STOP = new Set(['fc', 'cf', 'sc', 'ac', 'club', 'cd', 'ud', 'rc', 'rcd', 'sd', 'ca', 'afc', 'sv', 'vfl', 'vfb', 'tsg', 'bv']);
function shortTeam(name) { const w = name.split(/\s+/).filter((t) => !STOP.has(t.toLowerCase())); return w.length ? w.join(' ') : name; }
const short = (c) => c === 0 ? shortTeam(S.meta.home) : c === 1 ? shortTeam(S.meta.away) : c === 2 ? 'Dead' : 'None';
let lastCurrent = -2;
function markCurrentDiff() {
  const f = S.frame;
  let cur = -1;
  const arr = S.diffShown;
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].e <= f) lo = mid + 1; else hi = mid; }
  if (lo < arr.length && arr[lo].s <= f) cur = lo;
  if (cur === lastCurrent) return;
  const ol = $('diffList');
  ol.querySelector('li.current')?.classList.remove('current');
  if (cur >= 0) {
    const li = ol.querySelector(`li[data-i="${cur}"]`);
    if (li) { li.classList.add('current'); li.scrollIntoView({ block: 'nearest' }); }
  }
  lastCurrent = cur;
}

// ---------------------------------------------------------------------------- render loop
let renderQueued = false;
function requestRender() { if (!renderQueued) { renderQueued = true; requestAnimationFrame(() => { renderQueued = false; render(); }); } }
function render() {
  if (!S.ready) return;
  drawOverview();
  drawLens();
  drawPitch();
  drawOverlay();
  stageStatus();
  renderRail();
  drawTooltip();
}

function refreshLanes() {
  computeDiff();
  buildTimelineLabels();
  layoutTimeline();
  renderDiffList();
  requestRender();
}
function removeLane(i) {
  S.lanes.splice(i, 1);
  S.truth = S.lanes.findIndex((l) => l.kind === 'truth');
  if (S.compare === i || S.compare >= S.lanes.length) S.compare = S.lanes.findIndex((l) => l.kind === 'model');
  else if (S.compare > i) S.compare--;
  refreshLanes();
}

// ---------------------------------------------------------------------------- navigation
function jumpTruthChange(dir) {
  if (S.truth < 0) return;
  const lane = S.lanes[S.truth];
  const st = lane.runs.starts;
  const i = runIndex(lane, S.frame);
  if (dir > 0) { if (i + 1 < st.length) seek(st[i + 1]); }
  else seek(st[i] < S.frame ? st[i] : st[Math.max(0, i - 1)]);
}
function jumpDiff(dir) {
  const arr = S.diffShown.length ? S.diffShown : S.diff;
  if (!arr.length) return;
  const pre = Math.round(S.fps / 2);
  const target = dir > 0 ? arr.find((d) => d.s - pre > S.frame) : [...arr].reverse().find((d) => d.s - pre < S.frame - 1);
  if (target) seek(target.s - pre);
}

// ---------------------------------------------------------------------------- pointer on the timeline
function bindTimeline() {
  const ov = $('overview'), lens = $('lens');
  const ovFrame = (e) => ((e.clientX - ov.getBoundingClientRect().left) / ov.clientWidth) * S.n;
  ov.addEventListener('pointerdown', (e) => { ov.setPointerCapture(e.pointerId); seek(ovFrame(e)); });
  ov.addEventListener('pointermove', (e) => {
    if (ov.hasPointerCapture(e.pointerId)) seek(ovFrame(e), { noHash: true });
    S.hover = { canvas: 'overview', f: Math.round(ovFrame(e)), x: e.clientX, y: e.clientY }; requestRender();
  });
  ov.addEventListener('pointerup', () => writeHash());
  ov.addEventListener('pointerleave', () => { S.hover = null; requestRender(); });

  let drag = null;
  const lensFrame = (e) => { const { f0, ppf } = lensWindow(); return f0 + (e.clientX - lens.getBoundingClientRect().left) / ppf; };
  lens.addEventListener('pointerdown', (e) => {
    lens.setPointerCapture(e.pointerId);
    if (S.playing) setPlaying(false);
    drag = { x: e.clientX, f: S.frame, moved: false, at: lensFrame(e) };
  });
  lens.addEventListener('pointermove', (e) => {
    if (drag) {
      const dx = e.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      if (drag.moved) seek(drag.f - dx / lensWindow().ppf, { noHash: true });
    }
    S.hover = { canvas: 'lens', f: Math.round(lensFrame(e)), x: e.clientX, y: e.clientY }; requestRender();
  });
  lens.addEventListener('pointerup', (e) => { if (drag && !drag.moved) seek(drag.at); drag = null; writeHash(); });
  lens.addEventListener('pointerleave', () => { S.hover = null; requestRender(); });
  lens.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) { zoomLens(e.deltaY > 0 ? 1 : -1); return; }
    const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
    seek(S.frame + d * (S.lensSec * S.fps) / 900, { noHash: false });
  }, { passive: false });
}
const LENSES = [10, 30, 120, 600];
function zoomLens(dir) {
  const i = LENSES.indexOf(S.lensSec);
  setLens(LENSES[Math.max(0, Math.min(LENSES.length - 1, (i < 0 ? 1 : i) + dir))]);
}
function drawTooltip() {
  const tip = $('tooltip');
  if (!S.hover || S.hover.f < 0 || S.hover.f >= S.n) { tip.hidden = true; return; }
  const f = S.hover.f;
  tip.textContent = '';
  const head = document.createElement('div'); head.className = 'row num';
  head.textContent = `Frame ${fmtInt(f)} · ${clockText(f, false)}`;
  tip.appendChild(head);
  S.lanes.forEach((lane) => {
    const k = lane.codes[f];
    const r = document.createElement('div'); r.className = 'row';
    r.append(sw(k), document.createTextNode(`${lane.name}: ${labelName(k)}`));
    tip.appendChild(r);
  });
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, S.hover.x + 12) + 'px';
  tip.style.top = Math.max(8, S.hover.y - h - 12) + 'px';
}

// ---------------------------------------------------------------------------- adding lanes
async function addLaneFromFile(file) {
  const text = await file.text();
  const lines = text.split(/\r?\n/);
  let i0 = 0;
  if (!/^\d/.test(lines[0])) i0 = 1;
  const codes = new Uint8Array(S.n);
  let n = 0;
  for (let i = i0; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const comma = line.indexOf(',');
    const fr = +line.slice(0, comma);
    const lab = line.slice(comma + 1).trim().toUpperCase();
    if (fr !== n || !(lab in CODE)) { alertLane(`${file.name}: row ${i + 1} is not "${n},<H|A|D|X>"`); return; }
    if (n >= S.n) { alertLane(`${file.name}: more rows than the match has frames (${fmtInt(S.n)})`); return; }
    codes[n++] = CODE[lab];
  }
  if (n !== S.n) { alertLane(`${file.name}: ${fmtInt(n)} rows, the match has ${fmtInt(S.n)}`); return; }
  S.lanes.push(makeLane(file.name.replace(/\.csv$/i, ''), 'model', 'dropped file', codes, true));
  if (S.compare < 0) S.compare = S.lanes.length - 1;
  refreshLanes();
}
function alertLane(msg) {
  const t = $('tlEmpty');
  t.hidden = false; t.textContent = msg;
  setTimeout(() => { t.hidden = S.lanes.length > 0; }, 6000);
}

// ---------------------------------------------------------------------------- controls
function bindControls() {
  $('playBtn').addEventListener('click', () => setPlaying(!S.playing));
  document.querySelectorAll('[data-step]').forEach((b) => b.addEventListener('click', () => seek(S.frame + +b.dataset.step, { pause: true })));
  document.querySelectorAll('#viewSeg button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
  document.querySelectorAll('#speedSeg button').forEach((b) => b.addEventListener('click', () => setRate(+b.dataset.rate)));
  document.querySelectorAll('#lensSeg button').forEach((b) => b.addEventListener('click', () => setLens(+b.dataset.lens)));
  document.querySelectorAll('#diffSeg button').forEach((b) => b.addEventListener('click', () => {
    S.diffMin = +b.dataset.min;
    document.querySelectorAll('#diffSeg button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    renderDiffList(); requestRender();
  }));
  document.querySelectorAll('[data-nudge]').forEach((b) => b.addEventListener('click', () => {
    const v = S.videos.broadcast; if (!v) return;
    v.nudge += +b.dataset.nudge; store.set(`nudge.${S.meta.key}`, v.nudge);
    seekVideo(v); requestRender();
  }));
  $('nudgeVal').addEventListener('dblclick', () => { const v = S.videos.broadcast; if (v) { v.nudge = 0; store.set(`nudge.${S.meta.key}`, 0); seekVideo(v); requestRender(); } });
  $('optOverlay').addEventListener('change', (e) => { S.overlay = e.target.checked; store.set('overlay', S.overlay); requestRender(); });
  $('optTrails').addEventListener('change', (e) => { S.trails = e.target.checked; store.set('trails', S.trails); requestRender(); });
  $('stageMsgBtn').addEventListener('click', () => setView('pitch'));
  $('addLane').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) addLaneFromFile(f); e.target.value = ''; });

  const help = $('help'), helpBtn = $('helpBtn');
  const toggleHelp = (on) => { help.hidden = !on; helpBtn.setAttribute('aria-expanded', String(on)); };
  helpBtn.addEventListener('click', (e) => { e.stopPropagation(); toggleHelp(help.hidden); });
  document.addEventListener('click', (e) => { if (!help.hidden && !help.contains(e.target)) toggleHelp(false); });

  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => { if ([...(e.dataTransfer?.types || [])].includes('Files')) { dragDepth++; $('drop').hidden = false; } });
  window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('drop').hidden = true; });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault(); dragDepth = 0; $('drop').hidden = true;
    [...(e.dataTransfer?.files || [])].filter((f) => /\.csv$/i.test(f.name)).forEach(addLaneFromFile);
  });

  window.addEventListener('keydown', (e) => {
    if (e.target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    let handled = true;
    if (k === ' ' || k === 'k' || k === 'K') setPlaying(!S.playing);
    else if (k === 'ArrowRight') seek(S.frame + (e.shiftKey ? S.fps : 1), { pause: true });
    else if (k === 'ArrowLeft') seek(S.frame - (e.shiftKey ? S.fps : 1), { pause: true });
    else if (k === 'l' || k === 'L') seek(S.frame + 5 * S.fps);
    else if (k === 'j' || k === 'J') seek(S.frame - 5 * S.fps);
    else if (k === ']') jumpTruthChange(1);
    else if (k === '[') jumpTruthChange(-1);
    else if (k === '.') jumpDiff(1);
    else if (k === ',') jumpDiff(-1);
    else if (k === '1') setView('tactical');
    else if (k === '2') setView('broadcast');
    else if (k === '3') setView('pitch');
    else if (k === 'o' || k === 'O') { $('optOverlay').click(); }
    else if (k === 't' || k === 'T') cycleSet();
    else if (k === '+' || k === '=') zoomLens(-1);
    else if (k === '-' || k === '_') zoomLens(1);
    else if (k === '?') $('helpBtn').click();
    else if (k === 'Escape') toggleHelp(false);
    else handled = false;
    if (handled) e.preventDefault();
  });
  new ResizeObserver(() => { layoutStage(); layoutTimeline(); }).observe(document.body);
}

function buildLegend() {
  const ul = $('legend');
  ul.textContent = '';
  for (const [c, name] of [[0, S.meta.home], [1, S.meta.away], [2, 'Dead ball'], [3, 'No label'], ['diff', 'Model disagrees with the truth']]) {
    const li = document.createElement('li'); li.append(sw(c), document.createTextNode(name)); ul.appendChild(li);
  }
  if (S.sets.some((t) => t.flags)) {                // Fixed only: what the cleaner repaired
    for (const [cls, name] of [['ring-team', 'Fixed: team repaired'], ['ring-take', 'Fixed: keeper takeover'],
                               ['ring-fill', 'Fixed: keeper at saved place']]) {
      const li = document.createElement('li'); const r = document.createElement('span'); r.className = 'ring ' + cls;
      li.append(r, document.createTextNode(name)); ul.appendChild(li);
    }
  }
}

// ---------------------------------------------------------------------------- start
(async function main() {
  bindControls();
  try { await load(); } catch (err) {
    $('loadingText').textContent = `Could not load the match: ${err.message}. Is python -m mqf.viewer still running?`;
    return;
  }
  S.overlay = store.get('overlay', true); $('optOverlay').checked = S.overlay;
  S.trails = store.get('trails', true); $('optTrails').checked = S.trails;
  setRate(store.get('rate', 1));
  setLens(store.get('lens', 30));
  // views without a video are still selectable: the stage explains what is missing
  buildLegend();
  $('trkHomeName').textContent = `${shortTeam(S.meta.home)} players`;
  $('trkAwayName').textContent = `${shortTeam(S.meta.away)} players`;
  bindTimeline();
  const h = readHash();
  useSet(h.t && S.sets.some((t) => t.key === h.t) ? h.t : store.get(`set.${S.meta.key}`, 'original'), false);
  buildTrackingSeg();
  const labelled = truthLabelled();
  S.frame = Number.isFinite(h.f) ? Math.max(0, Math.min(S.n - 1, h.f)) : labelled ? labelled[0] : (S.meta.halves[0]?.[1] ?? 0);
  const firstView = h.v && ['tactical', 'broadcast', 'pitch'].includes(h.v) ? h.v
    : (S.videos.tactical?.meta.available ? 'tactical' : 'pitch');
  refreshLanes();
  setView(firstView);
  document.fonts?.ready.then(() => { ovCache = null; requestRender(); });
})();
