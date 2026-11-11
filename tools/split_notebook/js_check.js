// Runtime check for D3 charts without a browser: a minimal DOM good enough for d3-selection,
// scales, axes, shapes and event binding. Catches thrown errors and NaN/undefined in SVG attributes.
// Usage: node js_check.js charts.json report.json
const fs = require('fs');
const path = require('path');

class Style {
  setProperty(k, v) { this[k] = String(v); }
  removeProperty(k) { delete this[k]; }
  getPropertyValue(k) { return this[k] || ''; }
}

class ClassList {
  constructor(node) { this.node = node; }
  _get() { return (this.node.getAttribute('class') || '').split(/\s+/).filter(Boolean); }
  add(...c) { const s = new Set(this._get()); c.forEach(x => s.add(x)); this.node.setAttribute('class', [...s].join(' ')); }
  remove(...c) { this.node.setAttribute('class', this._get().filter(x => !c.includes(x)).join(' ')); }
  contains(c) { return this._get().includes(c); }
}

class Node {
  constructor(tag, ns, doc) {
    this.localName = tag; this.tagName = tag.toUpperCase(); this.nodeName = this.tagName;
    this.namespaceURI = ns; this.ownerDocument = doc; this.childNodes = []; this.parentNode = null;
    this.attributes = {}; this.style = new Style(); this.nodeType = 1; this._text = '';
    this.classList = new ClassList(this); this.__data__ = undefined;
  }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.childNodes.push(c); return c; }
  insertBefore(c, ref) {
    if (!ref) return this.appendChild(c);
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this; this.childNodes.splice(this.childNodes.indexOf(ref), 0, c); return c;
  }
  removeChild(c) { const i = this.childNodes.indexOf(c); if (i >= 0) this.childNodes.splice(i, 1); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  get children() { return this.childNodes.filter(n => n.nodeType === 1); }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get nextSibling() { const p = this.parentNode; if (!p) return null; return p.childNodes[p.childNodes.indexOf(this) + 1] || null; }
  get previousSibling() { const p = this.parentNode; if (!p) return null; return p.childNodes[p.childNodes.indexOf(this) - 1] || null; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  setAttributeNS(ns, k, v) { this.setAttribute(k, v); }
  getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; }
  getAttributeNS(ns, k) { return this.getAttribute(k); }
  hasAttribute(k) { return k in this.attributes; }
  removeAttribute(k) { delete this.attributes[k]; }
  removeAttributeNS(ns, k) { this.removeAttribute(k); }
  get id() { return this.getAttribute('id') || ''; }
  get textContent() { return this._text + this.childNodes.map(c => c.textContent).join(''); }
  set textContent(v) { this.childNodes = []; this._text = v == null ? '' : String(v); }
  get innerHTML() { return this._html || ''; }
  set innerHTML(v) { this.childNodes = []; this._text = ''; this._html = String(v); }
  addEventListener() {} removeEventListener() {} dispatchEvent() { return true; }
  getBBox() { return { x: 0, y: 0, width: 10, height: 10 }; }
  getComputedTextLength() { return 10; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 20, right: 100, bottom: 20 }; }
  get clientLeft() { return 0; } get clientTop() { return 0; }
  get ownerSVGElement() { let n = this.parentNode; while (n && n.localName !== 'svg') n = n.parentNode; return n; }
  _matches1(s) {
    if (s === '*') return true;
    const m = s.match(/^([a-zA-Z0-9_-]*)((?:[.#][a-zA-Z0-9_-]+)*)$/);
    if (!m) return false;
    if (m[1] && m[1].toLowerCase() !== this.localName.toLowerCase()) return false;
    for (const part of (m[2].match(/[.#][a-zA-Z0-9_-]+/g) || [])) {
      if (part[0] === '#' && this.id !== part.slice(1)) return false;
      if (part[0] === '.' && !this.classList.contains(part.slice(1))) return false;
    }
    return true;
  }
  matches(sel) {
    return sel.split(',').map(s => s.trim()).some(s => {
      const parts = s.split(/\s+/);
      if (!this._matches1(parts[parts.length - 1])) return false;
      let n = this.parentNode;
      for (let i = parts.length - 2; i >= 0; i--) {
        while (n && !(n._matches1 && n._matches1(parts[i]))) n = n.parentNode;
        if (!n) return false;
        n = n.parentNode;
      }
      return true;
    });
  }
  _desc(out = []) { for (const c of this.children) { out.push(c); c._desc(out); } return out; }
  querySelectorAll(sel) { return this._desc().filter(n => n.matches(sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  get firstElementChild() { return this.children[0] || null; }
  // Used by selection.order(): 4 = `other` follows this node, 2 = precedes (document order).
  compareDocumentPosition(other) {
    let root = this; while (root.parentNode) root = root.parentNode;
    const order = [root, ...root._desc()];
    const a = order.indexOf(this), b = order.indexOf(other);
    return b > a ? 4 : b < a ? 2 : 0;
  }
}

const XHTML = 'http://www.w3.org/1999/xhtml';
const doc = {
  createElement(tag) { return new Node(tag, XHTML, doc); },
  createElementNS(ns, tag) { return new Node(tag, ns, doc); },
  createTextNode(t) { const n = new Node('#text', XHTML, doc); n.nodeType = 3; n._text = String(t); return n; },
  getElementById(id) { return doc.documentElement.querySelector('#' + id); },
  querySelector(s) { return doc.documentElement.querySelector(s); },
  querySelectorAll(s) { return doc.documentElement.querySelectorAll(s); },
};
doc.documentElement = new Node('html', XHTML, doc);
doc.head = doc.documentElement.appendChild(new Node('head', XHTML, doc));
doc.body = doc.documentElement.appendChild(new Node('body', XHTML, doc));
global.document = doc;
global.window = { document: doc, devicePixelRatio: 1,
  getComputedStyle: n => ({ getPropertyValue: k => n.style.getPropertyValue(k) }) };

const d3 = require(path.join(__dirname, '..', '..', 'notebooks', 'vendor', 'd3.v7.min.js'));
global.window.d3 = d3;

const [chartsFile, reportFile] = process.argv.slice(2);
const charts = JSON.parse(fs.readFileSync(chartsFile, 'utf8'));
const results = [];
for (const c of charts) {
  const el = doc.body.appendChild(doc.createElement('div'));
  el.setAttribute('id', c.id);
  const r = { name: c.name, ok: true, error: null, elements: {}, bad_attrs: [] };
  try {
    // eval is the point of this harness: it runs the notebook's own chart functions (generated by
    // show_d3 in 00_setup.py), never external input.
    const draw = eval('(' + c.func + ')');
    draw(el, c.payload.data, d3, c.payload.opts);
  } catch (e) { r.ok = false; r.error = String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e); }
  for (const n of el._desc()) {
    r.elements[n.localName] = (r.elements[n.localName] || 0) + 1;
    for (const [k, v] of Object.entries(n.attributes)) {
      if (/NaN|undefined|Infinity/.test(v) && r.bad_attrs.length < 5) r.bad_attrs.push(`${n.localName}[${k}]=${v.slice(0, 60)}`);
    }
  }
  if (!r.elements.svg) { r.ok = false; r.error = r.error || 'no <svg> created'; }
  if (r.bad_attrs.length) r.ok = false;
  results.push(r);
}
fs.writeFileSync(reportFile, JSON.stringify(results, null, 1));
const bad = results.filter(r => !r.ok);
console.log(`charts: ${results.length}, failing: ${bad.length}`);
bad.forEach(r => console.log(`  FAIL ${r.name}: ${r.error || ''} ${r.bad_attrs.join('; ')}`));
