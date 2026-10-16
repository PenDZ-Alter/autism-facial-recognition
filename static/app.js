const $ = id => document.getElementById(id);
const MAX_IMAGES = 50;
const state = { images: [], models: [], selected: new Set(), nextId: 1, last: null };

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

/* ------------------------------------------------------------ settings */
function loadSelected() {
  try { return new Set(JSON.parse(localStorage.getItem('selectedModels') || '[]')); } catch { return new Set(); }
}
function saveSelected() {
  try { localStorage.setItem('selectedModels', JSON.stringify([...state.selected])); } catch {}
}
function updateRun() {
  $('run').disabled = !(state.images.length && state.selected.size);
}

/* ------------------------------------------------------------ images */
const pngOn = () => $('allowPng').checked;
function validImage(f) {
  if (/\.jpe?g$/i.test(f.name)) return true;
  return pngOn() && /\.png$/i.test(f.name);
}

function addImages(fileList) {
  $('error').textContent = '';
  let rejected = 0;
  for (const f of fileList) {
    if (state.images.length >= MAX_IMAGES) { $('error').textContent = `Maximum ${MAX_IMAGES} images per run.`; break; }
    if (!validImage(f)) { rejected++; continue; }
    state.images.push({ id: state.nextId++, file: f, url: URL.createObjectURL(f) });
  }
  if (rejected) {
    $('error').textContent = `${rejected} file(s) skipped. ` + (pngOn() ? 'Use JPG, JPEG or PNG.' : 'Use JPG or JPEG, or turn on PNG support.');
  }
  renderThumbs();
}

function renderThumbs() {
  const ul = $('thumbs');
  ul.replaceChildren();
  for (const im of state.images) {
    const li = el('li');
    const img = el('img'); img.src = im.url; img.alt = im.file.name;
    const x = el('button', '', '×'); x.type = 'button'; x.setAttribute('aria-label', 'Remove ' + im.file.name);
    x.addEventListener('click', () => { state.images = state.images.filter(i => i.id !== im.id); renderThumbs(); });
    li.append(img, x);
    ul.append(li);
  }
  updateRun();
}

const drop = $('drop'), fileInput = $('file');
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener('change', () => { addImages(fileInput.files); fileInput.value = ''; });
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => addImages(e.dataTransfer.files));

$('allowPng').addEventListener('change', () => {
  fileInput.accept = pngOn() ? '.jpg,.jpeg,.png,image/jpeg,image/png' : '.jpg,.jpeg,image/jpeg';
  $('fmtHint').textContent = (pngOn() ? 'JPG, JPEG or PNG' : 'JPG or JPEG') + ', up to 10 MB each';
  if (!pngOn()) {
    const before = state.images.length;
    state.images = state.images.filter(validImage);
    if (before !== state.images.length) $('error').textContent = 'PNG images were removed because PNG is turned off.';
    renderThumbs();
  }
});

/* ------------------------------------------------------------ models */
async function loadModels() {
  const box = $('models');
  try {
    const res = await fetch('/api/models');
    const d = await res.json();
    state.models = d.models;
    const names = new Set(d.models.map(m => m.name));
    state.selected = new Set([...state.selected].filter(n => names.has(n)));
    box.replaceChildren();
    if (!d.models.length) {
      box.append(el('li', 'muted', 'No .keras files found. Put them in ' + d.folder + ' or use "Add model file".'));
    }
    for (const m of d.models) {
      const li = el('li');
      const label = el('label');
      const cb = el('input'); cb.type = 'checkbox'; cb.checked = state.selected.has(m.name);
      cb.addEventListener('change', () => {
        cb.checked ? state.selected.add(m.name) : state.selected.delete(m.name);
        saveSelected(); updateRun();
      });
      const info = el('div');
      info.append(el('div', 'm-name', m.name), el('div', 'm-meta', `${m.size_mb} MB · preprocess: ${m.preprocess} · threshold ${m.threshold}`));
      label.append(cb, el('span', 'check'), info);
      li.append(label);
      box.append(li);
    }
  } catch (err) {
    box.replaceChildren(el('li', 'muted', 'Could not load models: ' + err.message));
  }
  updateRun();
}
function setAll(on) {
  state.selected = on ? new Set(state.models.map(m => m.name)) : new Set();
  saveSelected();
  document.querySelectorAll('#models input').forEach(cb => { cb.checked = on; });
  updateRun();
}
$('selAll').addEventListener('click', () => setAll(true));
$('selNone').addEventListener('click', () => setAll(false));
$('refresh').addEventListener('click', loadModels);

$('uploadModel').addEventListener('click', () => $('modelFile').click());
$('modelFile').addEventListener('change', async e => {
  const f = e.target.files[0]; e.target.value = '';
  if (!f) return;
  const msg = $('modelMsg'); msg.className = 'msg'; msg.textContent = 'Uploading ' + f.name + '…';
  const fd = new FormData(); fd.append('model', f);
  try {
    const res = await fetch('/api/models/upload', { method: 'POST', body: fd });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Upload failed.');
    state.selected.add(d.name); saveSelected();
    msg.textContent = 'Added ' + d.name + '.';
    await loadModels();
  } catch (err) { msg.className = 'msg bad'; msg.textContent = err.message; }
});

/* ------------------------------------------------------------ run */
$('run').addEventListener('click', async () => {
  const btn = $('run');
  btn.disabled = true; btn.textContent = 'Running…'; $('error').textContent = '';
  const fd = new FormData();
  state.images.forEach(im => fd.append('images', im.file, im.file.name));
  [...state.selected].forEach(m => fd.append('models', m));
  fd.append('allow_png', pngOn());
  fd.append('preprocess', $('prep').value);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: fd });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'Request failed.');
    state.last = { data: d, images: state.images.slice() };
    renderResults(d, state.images.slice());
  } catch (err) {
    $('error').textContent = err.message;
  } finally {
    btn.textContent = 'Run prediction'; updateRun();
  }
});

$('clear').addEventListener('click', () => {
  state.images = []; state.last = null; renderThumbs();
  $('out').replaceChildren(); $('empty').style.display = 'block'; $('csv').hidden = true; $('error').textContent = '';
});

/* ------------------------------------------------------------ results */
function predRow(name, p, ms) {
  const row = el('div', 'pred');
  row.append(el('div', 'p-model', name));
  if (p.error) { row.append(el('div', 'p-err', p.error)); return row; }
  row.append(el('div', 'p-class c' + p.label, `${p.class_name} (${p.label})`));
  row.append(el('div', 'p-conf', p.confidence.toFixed(2) + '%'));

  const wrap = el('div', 'bar-wrap');
  wrap.title = `Raw output (P of class 1): ${p.raw_output.toFixed(5)} · threshold ${p.threshold} · preprocess ${p.preprocess}`;
  const meter = el('div', 'meter');
  const a = el('div', 'a'); a.style.width = p.prob_asd + '%';
  meter.append(a);
  const tick = el('div', 'tick'); tick.style.left = ((1 - p.threshold) * 100) + '%';
  const nums = el('div', 'bar-nums');
  nums.append(el('span', '', 'ASD ' + p.prob_asd.toFixed(1) + '%'), el('span', '', 'Non-ASD ' + p.prob_non_asd.toFixed(1) + '%'));
  wrap.append(meter, tick, nums);
  row.append(wrap);
  return row;
}

function renderResults(d, images) {
  $('empty').style.display = 'none';
  const out = $('out'); out.replaceChildren();
  d.images.forEach((entry, i) => {
    const card = el('article', 'result-card');
    const head = el('div', 'rc-head');
    const img = el('img'); img.alt = ''; if (images[i]) img.src = images[i].url;
    const t = el('div');
    t.append(el('div', 'rc-title', entry.filename));

    if (entry.error) {
      t.append(el('div', 'rc-sub', entry.error));
      t.lastChild.style.color = 'var(--warn)';
      head.append(img, t); card.append(head); out.append(card);
      return;
    }
    const preds = d.models.map(m => entry.predictions[m]).filter(p => p && !p.error);
    if (preds.length > 1) {
      const non = preds.filter(p => p.label === 1).length;
      const txt = non === preds.length ? `All ${preds.length} models predict Non-ASD`
        : non === 0 ? `All ${preds.length} models predict ASD`
        : `Models disagree: ${non} Non-ASD, ${preds.length - non} ASD`;
      t.append(el('div', 'rc-sub', txt));
    }
    head.append(img, t); card.append(head);
    d.models.forEach(m => { if (entry.predictions[m]) card.append(predRow(m, entry.predictions[m])); });
    out.append(card);
  });
  const times = Object.entries(d.timings_ms || {}).map(([m, ms]) => `${m}: ${Math.round(ms)} ms`).join(' · ');
  if (times) out.append(el('p', 'muted', 'Inference time (all images per model): ' + times));
  $('csv').hidden = false;
}

/* ------------------------------------------------------------ CSV export */
$('csv').addEventListener('click', () => {
  if (!state.last) return;
  const q = v => '"' + String(v).replace(/"/g, '""') + '"';
  const rows = [['image', 'model', 'label', 'class', 'confidence_pct', 'prob_asd_pct', 'prob_non_asd_pct', 'raw_output', 'error']];
  for (const e of state.last.data.images) {
    if (e.error) { rows.push([e.filename, '', '', '', '', '', '', '', e.error]); continue; }
    for (const m of state.last.data.models) {
      const p = e.predictions[m]; if (!p) continue;
      rows.push(p.error ? [e.filename, m, '', '', '', '', '', '', p.error]
        : [e.filename, m, p.label, p.class_name, p.confidence.toFixed(2), p.prob_asd.toFixed(2), p.prob_non_asd.toFixed(2), p.raw_output.toFixed(6), '']);
    }
  }
  const blob = new Blob([rows.map(r => r.map(q).join(',')).join('\n')], { type: 'text/csv' });
  const a = el('a'); a.href = URL.createObjectURL(blob); a.download = 'predictions.csv'; a.click();
  URL.revokeObjectURL(a.href);
});

/* ------------------------------------------------------------ init */
state.selected = loadSelected();
loadModels();
