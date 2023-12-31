#pragma once

#include <functional>
#include <future>
#include <iostream>
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <glm/glm.hpp>

#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/structs/initial_state.hpp"
#include "empyrean/structs/shared_data.hpp"

class GlRenderer {
public:
  GLFWwindow* window;
  GLuint VBO, VAO, shaderProgram;
  int width, height, numBodies, useGpu;
  std::reference_wrapper<SharedData> sharedData;
  GlRenderer(std::string title, int width, int height, int numBodies, int useGpu,
             SharedData& sharedData);
  ~GlRenderer();
  void compileShaders();
  void initVertexData();
  void processInput();
  void setGlFlags();
  void mainLoop();
  void start();
};
