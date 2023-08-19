#pragma once

#include <future>
#include <iostream>
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <glm/glm.hpp>

#include "empyrean/engine/nbody_engine.hpp"

class GlRenderer {
public:
  GLFWwindow* window;
  NbodyEngine* engine;
  GLuint VBO, VAO, shaderProgram;
  int width, height, objectcount;
  GlRenderer(const int width, const int height);
  ~GlRenderer();
  void compileShaders();
  void initVertexData();
  void processInput();
  void setGlFlags();
  void mainLoop(std::promise<float*>& promise);
  void start(std::promise<float*>& promise);
};
