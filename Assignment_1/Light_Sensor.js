
class Light_Sensor {
  constructor(offset, placement) {
    this.offset = offset;
    this.placement = placement;
    this.intensity = 0;
  }

  update(robot, simulator) {
    const sensor_pos = p5.Vector.add(
      robot.position,
      p5.Vector.fromAngle(robot.heading).mult(this.offset.y)
    ).add(p5.Vector.fromAngle(robot.heading + HALF_PI).mult(this.offset.x));

    this.intensity = simulator.getLightIntensityAt(sensor_pos);
  }
}