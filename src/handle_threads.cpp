#include "empyrean/handle_threads.hpp"

#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/opengl/renderer.hpp"
#include "empyrean/structs.hpp"

const double GRAVITY_CONSTANT = 6.6743e-11;

void startEngine(EngineState initialState, std::reference_wrapper<SharedData> sharedData) {
  NbodyEngine engine(initialState, sharedData, EULER, PARALLEL);
  engine.start();
}

void startRenderer(std::string title, int width, int height, int numBodies,
                   std::reference_wrapper<SharedData> sharedData) {
  GlRenderer renderer(title, width, height, numBodies, sharedData);
  renderer.start();
}

void startAll(std::map<std::string, std::string> startOpts) {
  SharedData sharedData;

  int width = std::stoi(startOpts["wndWidth"]), height = std::stoi(startOpts["wndHeight"]);

  EngineState initialState
      = {{Body(glm::dvec3(0, 0, 0), (1000 / GRAVITY_CONSTANT), glm::dvec3(0, 0, 0)),
          Body(glm::dvec3(-200, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, 1.7, 0)),
          Body(glm::dvec3(300, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, -1.5, 0)),
          Body(glm::dvec3(-400, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, 1.4, 0)),
          Body(glm::dvec3(500, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, -1.3, 0)),
          Body(glm::dvec3(-700, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, 1.2, 0)),
          Body(glm::dvec3(800, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, -1.1, 0)),
          Body(glm::dvec3(-900, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, 1, 0)),
          Body(glm::dvec3(1100, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, -0.9, 0)),
          Body(glm::dvec3(-1300, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, 0.8, 0)),
          Body(glm::dvec3(1600, 0, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0, -0.5, 0)),
          Body(glm::dvec3(0, -200, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(1.7, 0, 0)),
          Body(glm::dvec3(0, 300, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(-1.5, 0, 0)),
          Body(glm::dvec3(0, -400, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(1.4, 0, 0)),
          Body(glm::dvec3(0, 500, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(-1.3, 0, 0)),
          Body(glm::dvec3(0, -700, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(1.2, 0, 0)),
          Body(glm::dvec3(0, 800, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(-1.1, 0, 0)),
          Body(glm::dvec3(0, -900, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(1, 0, 0)),
          Body(glm::dvec3(0, 1100, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(-0.9, 0, 0)),
          Body(glm::dvec3(0, -1300, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0.8, 0, 0)),
          Body(glm::dvec3(0, 1600, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(-0.5, 0, 0)),
          Body(glm::dvec3(0, -1900, 0), 0.1 / GRAVITY_CONSTANT, glm::dvec3(0.3, 0, 0))},
         GRAVITY_CONSTANT,
         1e-2};

  std::thread engineThread(startEngine, initialState, std::ref(sharedData));
  engineThread.detach();

  startRenderer(startOpts["wndTitle"], width, height, initialState.bodies.size(),
                std::ref(sharedData));
}