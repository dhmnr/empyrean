
#pragma once

#include <string>

class FpsCounter {
public:
  int frameCount;
  std::string title;
  double previousTime;

  FpsCounter(std::string title);
  void displayFps();
};