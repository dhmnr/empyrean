#include "empyrean/engine/nbody_engine.hpp"

#include <iostream>

NbodyEngine::NbodyEngine(std::vector<CosmicBody> cosmicBody, double timeStep, double G,
                         int engineType = SERIAL_EULER)
    : cosmicBody(cosmicBody), timeStep(timeStep), G(G), engineType(engineType) {}

void NbodyEngine::AdvanceTime() {
  switch (engineType) {
    case SERIAL_EULER:
      UpdatePositionsWithSerialEuler();
      break;

    case SERIAL_VERLET:
      std::cerr << "Error " << std::endl;
      break;

    default:
      std::cerr << "Error " << std::endl;
  }
}

std::vector<RealVector> NbodyEngine::GetNormalizedPositions() {}