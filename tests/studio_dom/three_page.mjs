/* «three» для стенда страницы студии: весь three настоящий, кроме
   WebGLRenderer — в node нет WebGL, а логике редактора холст не нужен.
   Стенд проверяет загрузку модели, правку мышью, сохранение версий и так далее. */
export * from '../../suspension/vendor/three.module.js';

export class WebGLRenderer {
  constructor() {
    this.domElement = {
      style: {}, width: 0, height: 0, _ev: {},
      addEventListener(t, f) { (this._ev[t] ||= []).push(f); },
      removeEventListener() {},
      setPointerCapture() {}, releasePointerCapture() {},
      hasPointerCapture() { return true; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 1200, height: 700 }; },
    };
    this.shadowMap = { enabled: false, type: 0 };
    this.toneMapping = 0;
    this.toneMappingExposure = 1;
    this.localClippingEnabled = false;
    this.frames = 0;
  }
  setPixelRatio() {}
  setSize(w, h) { this.domElement.width = w; this.domElement.height = h; }
  render() { this.frames++; }
  setClearColor() {}
  dispose() {}
  getContext() { return null; }
  getSize(v) { return v.set(this.domElement.width, this.domElement.height); }
}
