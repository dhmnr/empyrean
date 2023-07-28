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

int Sdl2Visualizer::RenderLoop(NbodyEngine engine) {
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
    engine.AdvanceTime();
    std::vector<SDL_Point> pointVector = RealVectorToSdlPoints(engine.GetNormalizedPositions());
    int pointsLength = pointVector.size();
    SDL_Point* pointArray = new SDL_Point[pointsLength];

    for (int i = 0; i < pointsLength; i++) {
      pointArray[i] = pointVector[i];
    }

    // Now you have a dynamically allocated C++ array (myArray) with the same elements as the vector

    // Don't forget to release the dynamically allocated memory when you're done with it
    SDL_RenderDrawPoints(renderer, pointArray, pointsLength);
    // SDL_RenderDrawPointsF(renderer, vertices, numPoints);
    // SDL_RenderFillRect(renderer, &rect);
    delete[] pointArray;
    // Update the screen
    SDL_RenderPresent(renderer);
  }
  return 0;
}

std::vector<SDL_Point> Sdl2Visualizer::RealVectorToSdlPoints(std::vector<RealVector> positions) {
  // TODO Implement RealVectorToSdlPoints
}

Sdl2Visualizer::~Sdl2Visualizer() {
  // Clean up and quit
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
