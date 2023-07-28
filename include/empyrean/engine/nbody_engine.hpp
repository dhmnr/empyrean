#include <vector>

#include "empyrean/engine/cosmic_body.hpp"
#include "empyrean/engine/real_vector.hpp"

#define SERIAL_EULER 0
#define SERIAL_VERLET 1
#define PARALLEL_VERLET 2

class NbodyEngine {
public:
  std::vector<CosmicBody> cosmicBody;
  double G;
  int engineType;
  NbodyEngine(std::vector<CosmicBody> cosmicBody, double timeStep, double G,
              int engineType = SERIAL_EULER);
  std::vector<RealVector> GetNormalizedPositions();
  void AdvanceTime();
  void UpdatePositionsWithSerialEuler();
};