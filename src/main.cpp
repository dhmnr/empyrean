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
  std::map<std::string, std::string> stringOpts;

  cxxopts::Options options("empyrean", "Run N-Body Simulations on CPU/GPU");

  // clang-format off
  options.add_options()
    ("x,width", "Window Width", cxxopts::value<std::string>()->default_value("800"))
    ("y,height", "Window Height", cxxopts::value<std::string>()->default_value("800"))
    ("g,enable-gpu", "Use GPU", cxxopts::value<std::string>()->default_value("0"))
    ("f,file", "Input file", cxxopts::value<std::string>()->default_value("input.yaml"))
    ("h,help", "Print usage");
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  stringOpts["wndWidth"] = result["width"].as<std::string>();
  stringOpts["wndHeight"] = result["height"].as<std::string>();
  stringOpts["wndTitle"] = "EMPYREAN";
  stringOpts["enableGpu"] = result["enable-gpu"].as<std::string>();
  // stringOpts["wndWidth"] = "800";
  // stringOpts["wndHeight"] = "800";

  startAll(stringOpts);
  return 0;
}
