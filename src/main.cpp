#include <SDL.h>
// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include "visualizer/base_visualizer.hpp"
#include "visualizer/sdl2_visualizer.hpp"

// #include <empyrean/cuda_functions.cuh>
#include <iostream>

// settings
const unsigned int WINDOW_WIDTH = 800;
const unsigned int WINDOW_HEIGHT = 800;

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
  points[0] = {400,400};
  points[1] = {rand()%800, rand()%800};
  return points;
}

int main() {
  AbstractVisualizer* vizr = new Sdl2Visualizer("SDL2 Window Title", WINDOW_WIDTH, WINDOW_HEIGHT);
  vizr->Initialize();
  vizr->RenderLoop(randomPoint);
  return 0;
}
