#include "visualizer/sdl2_visualizer.hpp"

#include <SDL.h>

Sdl2Visualizer::Sdl2Visualizer(std::string title, int width, int height) {
  windowWidth = width;
  windowHeight = height;
  windowTitle = title;
}

int Sdl2Visualizer::Initialize() {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
    return 1;
  }

  // Initialize window
  window = SDL_CreateWindow(windowTitle.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                            windowWidth, windowHeight, SDL_WINDOW_SHOWN);

  if (window == nullptr) {
    std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
    SDL_Quit();
    return 1;
  }

  // Initialize renderer
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  if (renderer == nullptr) {
    std::cerr << "Failed to create SDL renderer: " << SDL_GetError() << std::endl;
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }
  return 0;
}

int Sdl2Visualizer::RenderLoop(SDL_Point* (*updateFunction)()) {
  // Main loop flag
  bool quit = false;

  // Event handler
  SDL_Event event;

  while (!quit) {
    // Handle events
    while (SDL_PollEvent(&event) != 0) {
      if (event.type == SDL_QUIT) {
        quit = true;
      }
    }

    // Clear the screen (black)
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Draw a red rectangle
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Point* points = updateFunction();
    SDL_RenderDrawLines(renderer, points, 2);
    // SDL_RenderDrawPointsF(renderer, vertices, numPoints);
    // SDL_RenderFillRect(renderer, &rect);

    // Update the screen
    SDL_RenderPresent(renderer);
  }
  return 0;
}

Sdl2Visualizer::~Sdl2Visualizer() {
  // Clean up and quit
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
