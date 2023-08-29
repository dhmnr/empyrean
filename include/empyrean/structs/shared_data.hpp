#pragma once

struct SharedData {
  std::mutex mtx;
  std::condition_variable cv;
  size_t objCount;
  float* hostPointer;
  void* devicePointer;
  bool stopRequested = false;
  double scaleFactor;
};