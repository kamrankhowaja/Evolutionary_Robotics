let robot;
let walls = [];

function setup() {
  createCanvas(1000, 1000); // Adjust as needed

  // Your walls
  walls = [
    new Wall(createVector(0, 0), createVector(width, 0)),
    new Wall(createVector(width, 0), createVector(width, height)),
    new Wall(createVector(width, height), createVector(0, height)),
    new Wall(createVector(0, height), createVector(0, 0)),
    new Wall(createVector(0, 0.6 * height), createVector(0.2 * width, 0.6 * height)),
    new Wall(createVector(0.56 * width, 0.5 * height), createVector(0.56 * width, height)),
    new Wall(createVector(0.78 * width, 0.4 * height), createVector(width, 0.4 * height)),
  ];

  // Create robot
  robot = new Robot(createVector(300, 800), walls);
}

function draw() {
  background(30);

  // Draw walls
  for (let w of walls) w.update();

  // Update and show robot
  robot.update();
  robot.show();
}