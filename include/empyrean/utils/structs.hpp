#pragma once

#include "empyrean/engine/body.hpp"

struct SharedData {
  std::mutex mtx;
  std::condition_variable cv;
  size_t objCount;
  float* hostPointer;
  void* devicePointer;
  bool stopRequested = false;
};

struct InitialState {
  std::vector<Body> bodies;
  double gravityConstant;
  double timeStep;
};