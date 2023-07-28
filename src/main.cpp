#include <SDL.h>
// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/visualizer/base_visualizer.hpp"
#include "empyrean/visualizer/sdl2_visualizer.hpp"

// #include <empyrean/cuda_functions.cuh>
#include <iostream>

// settings
const unsigned int WINDOW_WIDTH = 1920;
const unsigned int WINDOW_HEIGHT = 1080;

SDL_FPoint pivotPoint(SDL_FPoint inPoint) {
  return {inPoint.x + WINDOW_WIDTH / 2, inPoint.y + WINDOW_HEIGHT / 2};
}

// Host function to display SDL2 circle
int display() {
  const int numPoints = 64;
  const float radius = 250;

  // Allocate memory for circle points
  float* host_x = new float[numPoints];
  float* host_y = new float[numPoints];

  // Calculate circle points using CUDA
  // calculateCircle(host_x, host_y, radius, numPoints);

  SDL_FPoint* vertices = new SDL_FPoint[numPoints];
  for (int i = 0; i < numPoints; i++) {
    SDL_FPoint tmp = {host_x[i], host_y[i]};
    vertices[i] = pivotPoint(tmp);
  }

  // for (int i = 0; i < numPoints; i++) {
  //   std::cout << vertices[i].x << "," << vertices[i].y << std::endl;
  // }

  return 0;
}

SDL_Point* randomPoint() {
  SDL_Point* points = new SDL_Point[2];
  points[0] = {960, 540};
  points[1] = {rand() % 1920, rand() % 1080};
  return points;
}

int main() {
  NbodyEngine engine = new NbodyEngine(DIRECT_EULER);
  BaseVisualizer* visualizer
      = new Sdl2Visualizer("Empyrean N-Body Simulator", WINDOW_WIDTH, WINDOW_HEIGHT);
  visualizer->Initialize();
  visualizer->RenderLoop(engine);
  return 0;
}
