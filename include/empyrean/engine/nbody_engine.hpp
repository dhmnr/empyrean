#pragma once

#include <vector>

#include "empyrean/engine/cosmic_body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/engine/real_vector.hpp"

#define SERIAL_EULER 0
#define SERIAL_VERLET 1
#define PARALLEL_VERLET 2

class NbodyEngine {
public:
  EngineState state;
  int engineType;
  NbodyEngine(EngineState state, int engineType = SERIAL_EULER);
  // void NbodyEngine::Initialize(InitialState) {}
  std::vector<RealVector> GetNormalizedPositions();
  void AdvanceTime();
  void UpdatePositionsWithSerialEuler();
};