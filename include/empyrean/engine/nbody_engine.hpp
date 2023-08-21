#pragma once

#include <functional>
#include <future>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/structs.hpp"

#define EULER 0
#define VERLET 1

class NbodyEngine {
public:
  int numBodies;
  EngineState state;
  std::reference_wrapper<SharedData> sharedData;
  std::function<void()> calculateForces;
  NbodyEngine(EngineState state, SharedData& sharedData, int integrationMethod = EULER);
  void writePositionsToVertexArray();
  void updatePositions();
  void calculateForcesWithDirectEuler();
  void start();
};
