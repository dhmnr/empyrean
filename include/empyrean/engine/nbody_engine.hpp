#pragma once

#include <functional>
#include <future>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/utils/structs.hpp"

#define EULER 0
#define VERLET 1

class NbodyEngine {
public:
  int useGpu;
  InitialState state;
  size_t dvecBytes, dBytes;
  glm::dvec3 *position_h, *acceleration_h, *velocity_h, *position_d, *acceleration_d, *velocity_d;
  double *mass_d, *mass_h;
  std::reference_wrapper<SharedData> sharedData;
  NbodyEngine(InitialState state, SharedData &sharedData, int integrationMethod = EULER,
              int useGpu = 0);
  ~NbodyEngine();
  void writeToHostArray();
  void updatePositions();
  void initDeviceArrays();
  void calculateForces_Euler_Serial();
  void calculateForces_Euler_Parallel();
  void start();
};
