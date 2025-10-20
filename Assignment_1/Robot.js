class Robot {
  constructor(position, mode) {
    this.position = position.copy();
    this.heading = 0;
    this.mode = mode; // "aggression" or "fear"

    // Visuals
    this.robot_color = mode === "aggression" ? "red" : "yellow";

    // Sensors positioned relative to robot center
    this.left_sensor = new Light_Sensor(createVector(-10, 10), "left");
    this.right_sensor = new Light_Sensor(createVector(10, 10), "right");
    
    this.left_wheel = new Wheel(createVector(0, 4), "left")
    this.right_wheel = new Wheel(createVector(0, 4), "right")
    
    // Kinematic parameters
    this.speed = (this.left_wheel.velocity + this.right_wheel.velocity) / 2.0;  
    this.sensitivity = 0.002;     // turning sensitivity constant (acts as 'c')
    this.maxTurn = 1.5*PI;          // max heading change per timestep (radians)
    this.maxSpeed = 2;
    
  }

  move() {
    const direction = p5.Vector.fromAngle(this.heading);
    const step = direction.mult(this.speed);
    this.position.add(step);

    // Toroidal (wrap-around) movement
    this.position.x = (this.position.x + width) % width;
    this.position.y = (this.position.y + height) % height;
  }
  
  handleCollisions(all_robots) {
  const radius = 12; // approximate robot "size"
  for (let other of all_robots) {
    if (other === this) continue;

    const d = dist(this.position.x, this.position.y, other.position.x, other.position.y);
    if (d < radius * 2) {
      // They are colliding
      const overlap = radius * 2 - d;

      // Compute push direction (away from each other)
      const pushDir = p5.Vector.sub(this.position, other.position).normalize();

      // Move both apart a little
      this.position.add(pushDir.mult(overlap / 2));
      other.position.sub(pushDir.mult(overlap / 2));

      // Optional: make them bounce (reverse slightly)
      this.heading += random(PI / 2, PI);
      other.heading += random(PI / 2, PI);

      // Optional: slow them down
      this.speed *= 0.5;
      other.speed *= 0.5;
    }
  }
}


  update(simulator, all_robots) {
    // Update sensors
    this.left_sensor.update(this, simulator);
    this.right_sensor.update(this, simulator);

    const left = this.left_sensor.intensity;
    const right = this.right_sensor.intensity;

    // Behavior logic
    let diff;
    if (this.mode === "aggression") {
      diff = right - left;
      this.left_wheel.velocity = right;
      this.right_wheel.velocity = left;
    } else if (this.mode === "fear") {
      diff = left - right;
      this.left_wheel.velocity = left;
      this.right_wheel.velocity = right;
    }

    this.speed = (this.left_wheel.velocity + this.right_wheel.velocity) / 2.0;
    this.speed = constrain(this.speed, 0, this.maxSpeed);

    const jitter = random(-1, 1);
    diff += jitter;

    let deltaHeading = diff * this.sensitivity;
    deltaHeading = constrain(deltaHeading, -this.maxTurn, this.maxTurn);
    this.heading += deltaHeading;

    // Move robot
    this.move();

    // 🟠 Check for collisions with other robots
    this.handleCollisions(all_robots);

    // Draw robot
    this.show();
  }


  show() {
    push();
    translate(this.position.x, this.position.y);
    rotate(this.heading);
    fill(this.robot_color);
    stroke("white");
    strokeWeight(3);
    triangle(-10, -10, -10, 10, 15, 0);
    pop();
  }
}
