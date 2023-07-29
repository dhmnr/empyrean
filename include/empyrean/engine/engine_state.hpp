#pragma once

#include <vector>

#include "empyrean/engine/cosmic_body.hpp"

struct EngineState {
  std::vector<CosmicBody> cosmicBodies;
  double gravityConstant;
  double timeStep;
};