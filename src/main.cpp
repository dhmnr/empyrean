// #include <cuda_runtime.h>
// #include <empyrean/version.h>
#include <cxxopts.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <thread>

#include "empyrean/engine/body.hpp"
#include "empyrean/engine/engine_state.hpp"
#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/handle_threads.hpp"
#include "empyrean/opengl/renderer.hpp"

int main(int argc, char** argv) {
  std::map<std::string, std::string> startOpts;

  cxxopts::Options options("empyrean", "Run N-Body Simulations on CPU/GPU");

  // clang-format off
  options.add_options()
    ("x,width", "Window Width", cxxopts::value<std::string>()->default_value("800"))
    ("y,height", "Window Height", cxxopts::value<std::string>()->default_value("800"))
    ("f,file", "Input file", cxxopts::value<std::string>()->default_value("input.yaml"))
    ("h,help", "Print usage");
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  startOpts["wndWidth"] = result["width"].as<std::string>();
  startOpts["wndHeight"] = result["height"].as<std::string>();

  startOpts["wndTitle"] = "EMPYREAN";
  // startOpts["wndWidth"] = "800";
  // startOpts["wndHeight"] = "800";

  startAll(startOpts);
  return 0;
}
