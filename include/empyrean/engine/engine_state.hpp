#pragma once

#include <vector>

#include "empyrean/engine/body.hpp"

struct EngineState {
  std::vector<Body> bodies;
  double gravityConstant;
  double timeStep;
};
