#pragma once

#include <SDL.h>

#include <string>

#include "base_visualizer.hpp"

class Sdl2Visualizer : public BaseVisualizer {
public:
  SDL_Window* window;
  SDL_Renderer* renderer;

  Sdl2Visualizer(std::string windowTitle, int width, int height);
  int Initialize();
  int RenderLoop(std::vector<RealVector> (*renderFunction)()) override;
  ~Sdl2Visualizer();
};
