// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include <atomic>
#include <future>
#include <glm/glm.hpp>
#include <iostream>
#include <thread>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/opengl/renderer.hpp"

// settings
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;

const double GRAVITY_CONSTANT = 6.6743e-11;

void startEngine(EngineState initialState, std::future<float*>& future) {
  NbodyEngine engine(initialState, EULER);
  float* vertexArray = future.get();
  engine.start(vertexArray);
}

int main() {
  EngineState initialState
      = {{Body(glm::dvec3(0, 0, 0), (0.5 / GRAVITY_CONSTANT), glm::dvec3(0, 0, 0)),
          Body(glm::dvec3(-0.5, 0, 0), 0.01, glm::dvec3(0, 1, 0)),
          Body(glm::dvec3(0.9, 0, 0), 0.01, glm::dvec3(0, -0.6, 0))},
         GRAVITY_CONSTANT,
         1e-6};

  std::promise<float*> promise;
  std::future<float*> future = promise.get_future();

  std::thread engineThread(startEngine, initialState, std::ref(future));

  GlRenderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT);
  renderer.objectcount = initialState.bodies.size();

  engineThread.detach();
  renderer.start(promise);

  return 0;
}
