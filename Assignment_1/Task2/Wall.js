class Wall {
  constructor(starting_pt, ending_pt) {
    this.starting_pt = starting_pt;
    this.ending_pt = ending_pt;
  }

  update() {
    push();
    strokeWeight(2);
    stroke(255);
    line(this.starting_pt.x, this.starting_pt.y, this.ending_pt.x, this.ending_pt.y);
    pop();
  }
}