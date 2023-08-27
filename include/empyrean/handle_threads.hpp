#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/opengl/renderer.hpp"

void startEngine(InitialState initialState, std::reference_wrapper<SharedData> sharedData,
                 int enableGpu);

void startRenderer(std::string title, int width, int height, int numBodies,
                   std::reference_wrapper<SharedData> sharedData, int useGpu);

void startAll(std::map<std::string, std::string> stringOpts);