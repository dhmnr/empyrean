#pragma once

#include <SDL.h>

#include <string>

class BaseVisualizer {
public:
  int windowWidth;
  int windowHeight;
  std::string windowTitle;
  virtual int RenderLoop(SDL_Point* (*updateFunction)()) = 0;
  virtual int Initialize() = 0;
};