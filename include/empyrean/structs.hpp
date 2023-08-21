#pragma once

struct SharedData {
  std::mutex mtx;
  std::condition_variable cv;
  int objectCount;
  float* vertexDataPtr;
  bool stopRequested = false;
};