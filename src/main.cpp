// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <thread>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/handle_threads.hpp"
#include "empyrean/opengl/renderer.hpp"

int main() {
  std::map<std::string, std::string> startOpts;
  startOpts["wndTitle"] = "EMPYREAN";
  startOpts["wndWidth"] = "800";
  startOpts["wndHeight"] = "800";

  startAll(startOpts);
}
