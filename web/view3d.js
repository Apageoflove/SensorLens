"use strict";
// 简易 WebGL 3D 视图：点云 + 3D 标注线框 + 轨迹。z 轴朝上，左键旋转，右键/Shift+左键平移，滚轮缩放。
(function () {
  const VS_PTS = `
attribute vec3 pos; attribute float val;
uniform mat4 mvp; uniform float size; uniform vec2 range; uniform int mode;
varying vec3 col;
vec3 jet(float t){ t=clamp(t,0.0,1.0); return clamp(vec3(1.5-abs(4.0*t-3.0),1.5-abs(4.0*t-2.0),1.5-abs(4.0*t-1.0)),0.0,1.0); }
void main(){
  gl_Position = mvp * vec4(pos,1.0);
  float v = mode==0 ? pos.z : val;
  col = jet((v-range.x)/max(range.y-range.x,1e-6));
  gl_PointSize = size;
}`;
  const FS_PTS = `precision mediump float; varying vec3 col; void main(){ gl_FragColor = vec4(col,1.0); }`;
  const VS_LN = `attribute vec3 pos; attribute vec3 color; uniform mat4 mvp; varying vec3 col;
void main(){ gl_Position = mvp*vec4(pos,1.0); col=color; }`;
  const FS_LN = `precision mediump float; varying vec3 col; void main(){ gl_FragColor=vec4(col,1.0); }`;

  function compile(gl, vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function lookAt(eye, c, up) {
    const z = norm(sub(eye, c)), x = norm(cross(up, z)), y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
  }
  function mul(a, b) {  // 列主序 a*b
    const o = new Array(16).fill(0);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    return o;
  }
  function hex(c) {
    const m = /^#?([0-9a-f]{6})$/i.exec(c || "");
    if (!m) return [1, 1, 1];
    const n = parseInt(m[1], 16);
    return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  class View3D {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl", { antialias: true });
      if (!gl) throw new Error("浏览器不支持 WebGL");
      this.gl = gl;
      this.pp = compile(gl, VS_PTS, FS_PTS);
      this.lp = compile(gl, VS_LN, FS_LN);
      this.ptBuf = gl.createBuffer(); this.lnBuf = gl.createBuffer();
      this.nPts = 0; this.nLn = 0;
      this.cam = { target: [0, 0, 0], dist: 60, yaw: Math.PI, pitch: 0.6 };
      this.opts = { size: 2, mode: 0, range: [-2, 3] };
      this._bind();
    }
    setPoints(f32, stride = 4) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.ptBuf);
      gl.bufferData(gl.ARRAY_BUFFER, f32, gl.STATIC_DRAW);
      this.nPts = f32.length / stride;
      const z = [], it = [];
      for (let i = 0; i < f32.length; i += stride * Math.max(1, Math.floor(this.nPts / 20000))) { z.push(f32[i + 2]); it.push(f32[i + 3]); }
      const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) * p)] : 0; };
      this.zRange = [pct(z, 0.02), pct(z, 0.98)];
      this.iRange = [pct(it, 0.02), pct(it, 0.98)];
      this.setMode(this.opts.mode);
    }
    setMode(mode) {
      this.opts.mode = mode;
      this.opts.range = (mode === 0 ? this.zRange : this.iRange) || [0, 1];
      this.draw();
    }
    // segs: [{a:[x,y,z], b:[x,y,z], color:"#rrggbb"}]
    setLines(segs) {
      const gl = this.gl, d = new Float32Array(segs.length * 12);
      segs.forEach((s, i) => {
        const c = hex(s.color);
        d.set([...s.a, ...c, ...s.b, ...c], i * 12);
      });
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lnBuf);
      gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW);
      this.nLn = segs.length * 2;
      this.draw();
    }
    frame(center, radius) {
      this.cam.target = center.slice();
      this.cam.dist = Math.max(10, radius * 1.6);
      this.draw();
    }
    matrix() {
      const c = this.cam, w = this.canvas.width, h = this.canvas.height;
      const eye = [c.target[0] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw),
                   c.target[1] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw),
                   c.target[2] + c.dist * Math.sin(c.pitch)];
      return mul(perspective(Math.PI / 4, w / Math.max(1, h), 0.1, 5000), lookAt(eye, c.target, [0, 0, 1]));
    }
    resize() {
      const r = this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      this.canvas.width = Math.max(1, Math.round(r.width * dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * dpr));
      this.draw();
    }
    draw() {
      const gl = this.gl;
      if (this._raf) return;
      this._raf = requestAnimationFrame(() => {
        this._raf = 0;
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0.055, 0.055, 0.055, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        const m = this.matrix();
        if (this.nPts) {
          gl.useProgram(this.pp);
          gl.bindBuffer(gl.ARRAY_BUFFER, this.ptBuf);
          const lp = gl.getAttribLocation(this.pp, "pos"), lv = gl.getAttribLocation(this.pp, "val");
          gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 16, 0);
          gl.enableVertexAttribArray(lv); gl.vertexAttribPointer(lv, 1, gl.FLOAT, false, 16, 12);
          gl.uniformMatrix4fv(gl.getUniformLocation(this.pp, "mvp"), false, m);
          gl.uniform1f(gl.getUniformLocation(this.pp, "size"), this.opts.size * (window.devicePixelRatio || 1));
          gl.uniform2f(gl.getUniformLocation(this.pp, "range"), this.opts.range[0], this.opts.range[1]);
          gl.uniform1i(gl.getUniformLocation(this.pp, "mode"), this.opts.mode);
          gl.drawArrays(gl.POINTS, 0, this.nPts);
          gl.disableVertexAttribArray(lv);
        }
        if (this.nLn) {
          gl.useProgram(this.lp);
          gl.bindBuffer(gl.ARRAY_BUFFER, this.lnBuf);
          const lp = gl.getAttribLocation(this.lp, "pos"), lc = gl.getAttribLocation(this.lp, "color");
          gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 24, 0);
          gl.enableVertexAttribArray(lc); gl.vertexAttribPointer(lc, 3, gl.FLOAT, false, 24, 12);
          gl.uniformMatrix4fv(gl.getUniformLocation(this.lp, "mvp"), false, m);
          gl.drawArrays(gl.LINES, 0, this.nLn);
          gl.disableVertexAttribArray(lc);
        }
      });
    }
    project(p) {  // 世界点 → 画布 CSS 像素，用于标签
      const m = this.matrix();
      const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
      const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
      const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
      if (w <= 0.1) return null;
      const r = this.canvas.getBoundingClientRect();
      return [(x / w + 1) / 2 * r.width, (1 - y / w) / 2 * r.height];
    }
    _bind() {
      const cv = this.canvas;
      let drag = null;
      cv.addEventListener("contextmenu", (e) => e.preventDefault());
      cv.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; });
      window.addEventListener("mouseup", () => { drag = null; });
      window.addEventListener("mousemove", (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        const c = this.cam;
        if (drag.pan) {
          const k = c.dist * 0.0015;
          const fx = -Math.cos(c.yaw), fy = -Math.sin(c.yaw);
          c.target[0] += (-fy * dx + fx * dy) * k;
          c.target[1] += (fx * dx + fy * dy) * k;
        } else {
          c.yaw -= dx * 0.006;
          c.pitch = Math.max(0.03, Math.min(1.55, c.pitch + dy * 0.006));
        }
        this.draw(); this.onchange && this.onchange();
      });
      cv.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.cam.dist = Math.max(2, Math.min(3000, this.cam.dist * Math.exp(e.deltaY * 0.0012)));
        this.draw(); this.onchange && this.onchange();
      }, { passive: false });
    }
  }
  window.View3D = View3D;
})();
