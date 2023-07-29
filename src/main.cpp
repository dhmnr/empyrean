#include <SDL.h>
// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include "empyrean/engine/cosmic_body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/visualizer/base_visualizer.hpp"
#include "empyrean/visualizer/sdl2_visualizer.hpp"

// #include <empyrean/cuda_functions.cuh>
#include <iostream>

// settings
const unsigned int WINDOW_WIDTH = 800;
const unsigned int WINDOW_HEIGHT = 800;

const double GRAVITY_CONSTANT = 6.6743e-11;

int main() {
  EngineState initialState
      = {{CosmicBody(RealVector(0, 0, 0), (200 / GRAVITY_CONSTANT), RealVector(0, 0, 0)),
          CosmicBody(RealVector(-200, 0, 0), 1, RealVector(0, 1, 0))},
         GRAVITY_CONSTANT,
         1};

  NbodyEngine* engine = new NbodyEngine(initialState, SERIAL_EULER);

  BaseVisualizer* visualizer
      = new Sdl2Visualizer("Empyrean N-Body Simulator", WINDOW_WIDTH, WINDOW_HEIGHT);
  visualizer->Initialize();
  visualizer->RenderLoop(engine);

  return 0;
}
