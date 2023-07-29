#pragma once

#include <SDL.h>

#include <string>

#include "base_visualizer.hpp"
#include "empyrean/engine/nbody_engine.hpp"

class Sdl2Visualizer : public BaseVisualizer {
public:
  SDL_Window* window;
  SDL_Renderer* renderer;

  Sdl2Visualizer(std::string windowTitle, int width, int height);
  int Initialize() override;
  int RenderLoop(NbodyEngine* engine) override;
  SDL_Point* RealVectorToSdlPoints(std::vector<RealVector> positions, int length);
  ~Sdl2Visualizer();
};
