/* OrbitControls для стенда: вращение и зум не проверяем, но target, update()
   и обновление матрицы камеры обязаны работать — от них зависят лучи мыши. */
import { Vector3 } from '../../suspension/vendor/three.module.js';

export class OrbitControls {
  constructor(camera, dom) {
    this.object = camera;
    this.domElement = dom;
    this.target = new Vector3();
    this.enableDamping = false;
    this.dampingFactor = 0.05;
    this.updates = 0;
  }
  update() { this.updates++; this.object.updateMatrixWorld(true); return true; }
  dispose() {}
  saveState() {}
  reset() {}
}
