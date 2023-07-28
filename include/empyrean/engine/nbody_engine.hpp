#include <vector>

#include "empyrean/engine/cosmic_body.hpp"
#include "empyrean/engine/real_vector.hpp"

class NbodyEngine {
public:
  std::vector<CosmicBody> cosmicBody;
  NbodyEngine();
  ~NbodyEngine();
  void AdvanceTime();
  std::vector<RealVector> GetNormalizedPositions();
};