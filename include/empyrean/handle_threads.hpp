#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "empyrean/engine/nbody_engine.hpp"
#include "empyrean/opengl/renderer.hpp"

void startEngine(EngineState initialState, std::reference_wrapper<SharedData> sharedData);

void startRenderer(std::string title, int width, int height, int numBodies,
                   std::reference_wrapper<SharedData> sharedData);

void startAll(std::map<std::string, std::string> startOpts);