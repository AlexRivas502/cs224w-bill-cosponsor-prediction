// Run every chart <script> of split_report.html against the minimal DOM from _build/js_check.js.
// Usage: node report/page_check.js report/split_report.html
const fs = require('fs'), path = require('path');
const domSrc = fs.readFileSync(path.join(__dirname, '..', '_build', 'js_check.js'), 'utf8').split('const [chartsFile')[0];
eval(domSrc);  // defines the minimal DOM + global d3 (our own harness code, not external input)
global.d3 = global.window.d3;
const doc = global.document;
const page = fs.readFileSync(process.argv[2], 'utf8');
const divs = [...page.matchAll(/<div class="chart" id="([^"]+)"/g)].map(m => m[1]);
divs.forEach(id => { const el = doc.body.appendChild(doc.createElement('div')); el.setAttribute('id', id); });
const scripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).filter(s => s.includes('const payload'));
let bad = 0;
scripts.forEach((s, i) => {
  try { eval(s); } catch (e) { bad++; console.log('script', i, 'threw', String(e).slice(0, 200)); }
});
divs.forEach(id => {
  const el = doc.getElementById(id);
  const svgs = el ? el._desc().filter(n => n.localName === 'svg').length : 0;
  const err = el && el.innerHTML.includes('chart error');
  if (!svgs || err) { bad++; console.log('no chart rendered in', id, err ? el.innerHTML.slice(0, 200) : ''); }
});
console.log(`page charts: ${divs.length}, scripts: ${scripts.length}, failing: ${bad}`);
process.exit(bad ? 1 : 0);
