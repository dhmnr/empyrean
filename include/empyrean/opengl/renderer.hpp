#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "empyrean/engine/nbody_engine.hpp"

class GlRenderer {
public:
  GLFWwindow* window;
  NbodyEngine* engine;
  GLuint VBO, VAO, shaderProgram;
  int width, height, objectcount;
  GlRenderer(NbodyEngine* engine, const int width, const int height);
  ~GlRenderer();
  void compileShaders();
  void initVertexData();
  void processInput();
  void setGlFlags();
  void mainLoop();
  void start();
};