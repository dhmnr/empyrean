#include <vector>

#include "empyrean/engine/real_vector.hpp"

struct CosmicBody {
  // vector for historical positional data
  std::vector<RealVector> position;
  RealVector velocity;
  RealVector acceleration;
  double mass;
  double timestep;
};