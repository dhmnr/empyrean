// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/opengl/renderer.hpp"

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// settings
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

const double GRAVITY_CONSTANT = 6.6743e-11;

int main() {
  EngineState initialState
      = {{Body(glm::dvec3(0, 0, 0), (0.5 / GRAVITY_CONSTANT), glm::dvec3(0, 0, 0)),
          Body(glm::dvec3(-0.5, 0, 0), 0.0001, glm::dvec3(0, 1, 0)),
          Body(glm::dvec3(0.9, 0, 0), 0.0001, glm::dvec3(0, 0.6, 0))},
         GRAVITY_CONSTANT,
         0.0005};

  NbodyEngine* engine = new NbodyEngine(initialState, EULER);
  GlRenderer* renderer = new GlRenderer(engine, WINDOW_WIDTH, WINDOW_HEIGHT);
  renderer->start();
  return 0;
}
