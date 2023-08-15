#pragma once

#include <vector>

#include "empyrean/engine/body.hpp"

struct EngineState {
  std::vector<Body> Bodies;
  double gravityConstant;
  double timeStep;
};