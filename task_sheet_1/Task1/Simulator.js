class Simulator {
  constructor(w, h) {
    this.width = w;
    this.height = h;
    this.light_sources = [];
    this.scaleFactor = 1.0;   // <--- for global intensity scaling
  }

  toroidalDistance(p1, p2) {
    let dx = abs(p1.x - p2.x);
    let dy = abs(p1.y - p2.y);
    if (dx > this.width / 2) dx = this.width - dx;
    if (dy > this.height / 2) dy = this.height - dy;
    return sqrt(dx * dx + dy * dy);
  }

  getLightIntensityAt(position) {
    let total_intensity = 0;
    for (let light_source of this.light_sources) {
      const d = this.toroidalDistance(position, light_source.pos);
      // linear falloff
      const raw = light_source.max_intensity - light_source.decreasing_factor * d;
      total_intensity += max(0, raw);
    }

    // --- Option 3: normalized light field ---
    total_intensity *= this.scaleFactor;
    return constrain(total_intensity, 0, 255);
  }

  drawLights() {
    for (let light_source of this.light_sources) {
      stroke("white");
      strokeWeight(3);
      fill("gold");
      ellipse(light_source.pos.x, light_source.pos.y, 20, 20);
    }
  }

  // --- Bonus tip: visualize the light field ---
  drawLightField(step = 20) {
    noStroke();
    for (let x = 0; x < this.width; x += step) {
      for (let y = 0; y < this.height; y += step) {
        const intensity = this.getLightIntensityAt(createVector(x, y));
        fill(intensity);   // grayscale brightness
        rect(x, y, step, step);
      }
    }
  }
}
