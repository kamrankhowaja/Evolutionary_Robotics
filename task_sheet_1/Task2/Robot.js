class Robot {
  constructor(pos, walls) {
    this.pos = pos.copy();
    this.walls = walls;
    this.heading = 0;

    this.speed = 2;
    this.rotationSpeed = 0.10; // radians per frame

    this.state = "Forward";
    this.rotationDir = 0;

    const halfLength = 20; // robot half-length along heading
    const halfWidth = 10;  // half-width perpendicular to heading

    // Front rays (left, middle, right)
    this.rays = [
      new Ray(createVector(halfLength, -halfWidth), radians(-25)), // left-front
      new Ray(createVector(halfLength, 0), 0),                     // middle-front
      new Ray(createVector(halfLength, halfWidth), radians(25)),   // right-front
    ];

    this.safeDistance = 0.4 * this.rays[1].maxLength; // stop before collision
  }

  update() {
    // Update rays
    for (let ray of this.rays) ray.update(this.pos, this.heading, this.walls);

    const leftLen = this.rays[0].currentLength;
    const midLen = this.rays[1].currentLength;
    const rightLen = this.rays[2].currentLength;
    const maxLen = this.rays[1].maxLength;

    // ===== FSM Logic =====
    switch (this.state) {
      case "Forward":
        if (midLen < this.safeDistance) {
          // Too close: start rotating randomly
          this.state = "Rotating";
          this.rotationDir = random([-1, 1]);
        } 
        else if (leftLen > rightLen + 0.01) {
          // More space on left
          this.state = "Steering Left";
        } 
        else if (rightLen > leftLen + 0.01) {
          // More space on right
          this.state = "Steering Right";
        }

        break;

      case "Rotating":
        this.heading += this.rotationSpeed * this.rotationDir;
        if (midLen > 0.5 * maxLen) this.state = "Forward";
        break;

      case "Steering Left":
        this.heading -= this.rotationSpeed;
        if (midLen > 0.5 * maxLen) this.state = "Forward";
        break;

      case "Steering Right":
        this.heading += this.rotationSpeed;
        if (midLen > 0.5 * maxLen) this.state = "Forward";
        break;
    }

    // ===== Move robot (only if middle ray allows) =====
    const halfRobotLength = 0.5 * 40; // half of robot rectangle length
    if ((this.state === "Forward" || this.state.startsWith("Steering")) &&
        midLen > halfRobotLength) {
      const vel = p5.Vector.fromAngle(this.heading).mult(this.speed);
      this.pos.add(vel);
    }
  }

  show() {
    // Draw robot rectangle
    push();
    translate(this.pos.x, this.pos.y);
    rotate(this.heading);
    fill(0, 150, 255);
    stroke(255);
    rectMode(CENTER);
    rect(0, 0, 40, 20);
    pop();

    // Draw rays
    for (let r of this.rays) r.show();

    // Show FSM state
    push();
    fill(255);
    textSize(14);
    textAlign(LEFT, TOP);
    text("State: " + this.state, 10, 10);
    pop();
  }
}
