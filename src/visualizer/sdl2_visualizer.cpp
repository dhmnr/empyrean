#include "empyrean/visualizer/sdl2_visualizer.hpp"

#include <SDL.h>

#include <iostream>
#include <vector>

#include "empyrean/engine/nbody_engine.hpp"

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

  SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

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

int Sdl2Visualizer::RenderLoop(NbodyEngine* engine) {
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
    engine->AdvanceTime();

    std::vector<RealVector> positionsVector = engine->GetNormalizedPositions();
    int length = positionsVector.size();
    SDL_Point* positionsArray = RealVectorToSdlPoints(positionsVector, length);

    // Now you have a dynamically allocated C++ array (myArray) with the same elements as the vector

    // Don't forget to release the dynamically allocated memory when you're done with it
    SDL_RenderDrawPoints(renderer, positionsArray, length);
    // SDL_RenderDrawPointsF(renderer, vertices, numPoints);
    // SDL_RenderFillRect(renderer, &rect);
    delete[] positionsArray;
    // Update the screen
    SDL_RenderPresent(renderer);
  }
  return 0;
}

SDL_Point* Sdl2Visualizer::RealVectorToSdlPoints(std::vector<RealVector> positions, int length) {
  // Allocate memory for SDL_Point array based on the number of positions
  SDL_Point* sdlPoints = new SDL_Point[length];

  for (size_t i = 0; i < length; ++i) {
    // Convert 3D RealVector to 2D SDL_Point by dropping the z-coordinate
    int x = static_cast<int>(positions[i].x);
    int y = static_cast<int>(positions[i].y);

    sdlPoints[i] = {x, y};
  }

  return sdlPoints;
}

Sdl2Visualizer::~Sdl2Visualizer() {
  // Clean up and quit
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
