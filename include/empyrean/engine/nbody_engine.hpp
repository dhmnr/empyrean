#pragma once

#include <functional>
#include <future>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/utils/structs.hpp"

#define EULER 0
#define VERLET 1

#define SERIAL 0
#define PARALLEL 1

class NbodyEngine {
public:
  int numBodies;
  InitialState state;
  std::reference_wrapper<SharedData> sharedData;
  std::function<void()> calculateForces;
  NbodyEngine(InitialState state, SharedData& sharedData, int integrationMethod = EULER,
              int computeType = SERIAL);
  void writePositionsToVertexArray();
  void updatePositions();
  void calculateForces_Euler_Serial();
  void calculateForces_Euler_Parallel();
  void start();
};
