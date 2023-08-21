#include "empyrean/handle_threads.hpp"

#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/opengl/renderer.hpp"
#include "empyrean/structs.hpp"

const double GRAVITY_CONSTANT = 6.6743e-11;

void startEngine(EngineState initialState, std::reference_wrapper<SharedData> sharedData) {
  NbodyEngine engine(initialState, sharedData, EULER);
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
      = {{Body(glm::dvec3(0, 0, 0), (0.5 / GRAVITY_CONSTANT), glm::dvec3(0, 0, 0)),
          Body(glm::dvec3(-0.5, 0, 0), 0.01, glm::dvec3(0, 1, 0)),
          Body(glm::dvec3(0.9, 0, 0), 0.01, glm::dvec3(0, -0.6, 0))},
         GRAVITY_CONSTANT,
         1e-6};

  std::thread engineThread(startEngine, initialState, std::ref(sharedData));
  engineThread.detach();

  startRenderer(startOpts["wndTitle"], width, height, initialState.bodies.size(),
                std::ref(sharedData));
}