#include <vector>

#include "empyrean/engine/real_vector.hpp"

class CosmicBody {
  // vector for historical positional data
  std::vector<RealVector> position;
  RealVector velocity;
  RealVector acceleration;
  double mass;
  CosmicBody(RealVector position, double mass, RealVector velocity, RealVector acceleration);
};