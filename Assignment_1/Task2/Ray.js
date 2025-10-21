class Ray {
  constructor(localOffset, relativeAngle, maxLength = 0.15 * width) {
    this.localOffset = localOffset.copy();
    this.relativeAngle = relativeAngle;
    this.maxLength = maxLength;
    this.pos = createVector(0, 0);
    this.dir = createVector(0, 0);
    this.hitPoint = null;
    this.currentLength = this.maxLength;
  }

  update(robotPos, robotHeading, walls) {
    const offsetRotated = this.localOffset.copy().rotate(robotHeading);
    this.pos = p5.Vector.add(robotPos, offsetRotated);
    this.dir = p5.Vector.fromAngle(robotHeading + this.relativeAngle).setMag(this.maxLength);

    let closest = null;
    let record = Infinity;

    for (let wall of walls) {
      const hit = this.intersectSegment(wall.starting_pt, wall.ending_pt);
      if (hit && hit.dist < record) {
        record = hit.dist;
        closest = hit.point;
      }
    }

    if (closest) {
      this.hitPoint = closest.copy();
      this.currentLength = record;
    } else {
      this.hitPoint = null;
      this.currentLength = this.maxLength;
    }
  }

  show() {
    push();
    stroke(255, 200);
    strokeWeight(2);
    const end = this.hitPoint
      ? this.hitPoint
      : createVector(this.pos.x + this.dir.x, this.pos.y + this.dir.y);
    line(this.pos.x, this.pos.y, end.x, end.y);

    // origin marker
    noStroke();
    fill(255, 0, 0);
    circle(this.pos.x, this.pos.y, 4);

    // hit marker
    if (this.hitPoint) {
      fill(255, 50, 50);
      circle(this.hitPoint.x, this.hitPoint.y, 6);
    }

    pop();
  }

  intersectSegment(segA, segB) {
    const p = this.pos.copy();
    const r = this.dir.copy();
    const q = segA.copy();
    const s = p5.Vector.sub(segB, segA);
    const rxs = r.x * s.y - r.y * s.x;
    if (abs(rxs) < 1e-8) return null;

    const q_p = p5.Vector.sub(q, p);
    const t = (q_p.x * s.y - q_p.y * s.x) / rxs;
    const u = (q_p.x * r.y - q_p.y * r.x) / rxs;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      const inter = p5.Vector.add(p, r.copy().mult(t));
      const dist = p.dist(inter);
      return { point: inter, dist };
    }
    return null;
  }
}