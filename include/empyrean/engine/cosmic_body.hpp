#pragma once

#include <vector>

#include "empyrean/engine/real_vector.hpp"

class CosmicBody {
public:
  // vector for historical positional data
  std::vector<RealVector> position;
  RealVector velocity;
  RealVector acceleration;
  double mass;
  CosmicBody(RealVector position, double mass, RealVector velocity = RealVector(0, 0, 0),
             RealVector acceleration = RealVector(0, 0, 0));
};