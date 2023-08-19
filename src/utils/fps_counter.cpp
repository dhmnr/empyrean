#include "empyrean/utils/fps_counter.hpp"

#include <GLFW/glfw3.h>

#include <iostream>

FpsCounter::FpsCounter(std::string title) : title(title) {
  frameCount = 0;
  previousTime = glfwGetTime();
}

void FpsCounter::displayFps() {
  double currentTime = glfwGetTime();
  frameCount++;
  if (currentTime - previousTime >= 1.0) {
    std::cout << title << " : " << frameCount << " Hz" << std::endl;
    frameCount = 0;
    previousTime = currentTime;
  }
}
