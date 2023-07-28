#pragma once

#include <SDL.h>

#include <string>
#include <vector>

#include "empyrean/engine/real_vector.hpp"

class BaseVisualizer {
public:
  int windowWidth;
  int windowHeight;
  std::string windowTitle;
  virtual int RenderLoop(std::vector<RealVector> (*renderFunction)()) = 0;
  virtual int Initialize() = 0;
};
