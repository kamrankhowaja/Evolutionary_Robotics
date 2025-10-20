let uiContainer;
let label_current_num_of_robots;
let num_of_aggression_robots, num_of_fear_robots;
let btn_add_num_of_robots, btn_delete_num_of_robots, btn_delete_all_robots;

function create_user_input_interface() {
  // === Create main container ===
  uiContainer = createDiv();
  uiContainer.style("position", "absolute");
  uiContainer.style("left", width + 30 + "px");
  uiContainer.style("top", "30px");
  uiContainer.style("color", "white");
  uiContainer.style("width", "320px");
  uiContainer.style("padding", "20px");
  uiContainer.style("background", "rgba(0,0,0,0.4)");
  uiContainer.style("border", "1px solid #555");
  uiContainer.style("border-radius", "12px");
  uiContainer.style("display", "flex");
  uiContainer.style("flex-direction", "column");
  uiContainer.style("gap", "12px");
  uiContainer.style("font-family", "sans-serif");

  // === Title ===
  const title = createElement("h2", "Braitenburg Vehicles Simulator");
  title.parent(uiContainer);
  title.style("margin", "0 0 10px 0");
  title.style("font-size", "18px");
  title.style("color", "#00bfff");

  // === Aggression robots ===
  const labelAgg = createElement("label", "Aggression Robots:");
  labelAgg.parent(uiContainer);

  num_of_aggression_robots = createInput("1");
  num_of_aggression_robots.parent(uiContainer);
  num_of_aggression_robots.size(60);

  // === Fear robots ===
  const labelFear = createElement("label", "Fear Robots:");
  labelFear.parent(uiContainer);

  num_of_fear_robots = createInput("1");
  num_of_fear_robots.parent(uiContainer);
  num_of_fear_robots.size(60);

  // === Buttons ===
  btn_add_num_of_robots = createButton("Add Robots");
  btn_add_num_of_robots.parent(uiContainer);
  btn_add_num_of_robots.mousePressed(handle_creating_new_robot);

  btn_delete_num_of_robots = createButton("Delete Robots");
  btn_delete_num_of_robots.parent(uiContainer);
  btn_delete_num_of_robots.mousePressed(handle_deleting_robots);

  btn_delete_all_robots = createButton("Delete All Robots");
  btn_delete_all_robots.parent(uiContainer);
  btn_delete_all_robots.mousePressed(handle_deleting_all_robots);

  // === Current Robot Count ===
  label_current_num_of_robots = createElement("div", `Current Robots: 0`);
  label_current_num_of_robots.parent(uiContainer);
  label_current_num_of_robots.style("margin-top", "10px");
  label_current_num_of_robots.style("font-weight", "bold");
  label_current_num_of_robots.style("color", "#FFD700");
}


// === Helper to update label ===
function updateRobotCountLabel() {
  if (label_current_num_of_robots) {
    label_current_num_of_robots.html(`Current Robots: ${robots.length}`);
  }
}


// === Button handlers ===
function handle_creating_new_robot() {
  const nAgg = int(num_of_aggression_robots.value());
  const nFear = int(num_of_fear_robots.value());

  for (let i = 0; i < nAgg; i++) {
    const x = random(0, width);
    const y = random(0, height);
    robots.push(new Robot(createVector(x, y), "aggression"));
  }

  for (let i = 0; i < nFear; i++) {
    const x = random(0, width);
    const y = random(0, height);
    robots.push(new Robot(createVector(x, y), "fear"));
  }

  updateRobotCountLabel();
}


function handle_deleting_robots() {
  const nAgg = int(num_of_aggression_robots.value());
  const nFear = int(num_of_fear_robots.value());

  for (let i = 0; i < nAgg; i++) {
    const idx = robots.findIndex(r => r.mode === "aggression");
    if (idx !== -1) robots.splice(idx, 1);
  }

  for (let i = 0; i < nFear; i++) {
    const idx = robots.findIndex(r => r.mode === "fear");
    if (idx !== -1) robots.splice(idx, 1);
  }

  updateRobotCountLabel();
}


function handle_deleting_all_robots() {
  robots.length = 0;
  updateRobotCountLabel();
}
