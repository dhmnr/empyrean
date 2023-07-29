#include "empyrean/engine/nbody_engine.hpp"

#include <iostream>

#include "empyrean/engine/engine_state.hpp"

NbodyEngine::NbodyEngine(EngineState state, int engineType)
    : engineType(engineType), state(state) {}

// void NbodyEngine::Initialize(InitialState) {}

void NbodyEngine::AdvanceTime() {
  switch (engineType) {
    case SERIAL_EULER:
      UpdatePositionsWithSerialEuler();
      break;

    case SERIAL_VERLET:
      std::cerr << "Error Not Implemented" << std::endl;
      break;

    default:
      std::cerr << "Error " << std::endl;
  }
}

std::vector<RealVector> NbodyEngine::GetNormalizedPositions() {
  // TODO GetNormalizedPositions
}