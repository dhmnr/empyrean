#pragma once

#include <functional>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"

#define EULER 0
#define VERLET 1

class NbodyEngine {
public:
  EngineState state;
  std::function<void()> calculateForces;
  NbodyEngine(EngineState state, int integrationMethod = EULER);
  void writePositionsToVertexArray(float* vertexArray);
  void updatePositions(float* vertexArray);
  void calculateForcesWithDirectEuler();
};