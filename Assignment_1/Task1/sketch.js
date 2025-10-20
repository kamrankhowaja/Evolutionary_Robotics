var simulator;
var robots = [];
var light_sources = [];



function setup() {
  
  createCanvas(1000, 1000);

  create_user_input_interface(robots)
  
  simulator = new Simulator(width, height);
  

  
  robots = [
  
    new Robot(createVector(100, 600), "aggression"), // red colored triangle
    new Robot(createVector(100, 600), "fear") // yellow colored triangle
  
  ];
  
  updateRobotCountLabel();
  
  light_sources = [
    
    new Light_Source(createVector(300, 300)),
    new Light_Source(createVector(700, 700)),
    
    
  ]
  
  simulator.light_sources = light_sources;
  
}

function draw() {
  
  background(0);
  
    
  // --- visualize light intensity field ---
  simulator.drawLightField(15);  // smaller step = smoother shading
  simulator.drawLights();

  
  for(let robot of robots){
    
    robot.update(simulator, robots)
    
  }
  

  
}