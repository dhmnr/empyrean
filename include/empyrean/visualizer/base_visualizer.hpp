#pragma once

#include <SDL.h>

#include <string>
#include <vector>

#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/engine/real_vector.hpp"

class BaseVisualizer {
public:
  int windowWidth;
  int windowHeight;
  std::string windowTitle;
  virtual int RenderLoop(NbodyEngine* engine) = 0;
  virtual int Initialize() = 0;
};
