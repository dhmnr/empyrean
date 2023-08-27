#include "empyrean/engine/nbody_engine.hpp"

#include <math.h>

#include <condition_variable>
#include <iostream>

#include "empyrean/engine/engine_state.hpp"
#include "empyrean/utils/fps_counter.hpp"
#include "empyrean/utils/structs.hpp"

NbodyEngine::NbodyEngine(InitialState state, SharedData& sharedData, int integrationMethod,
                         int computeType)
    : state(state), sharedData(sharedData) {
  // std::cout << state.bodies[0].position.x << state.bodies[0].position.y
  //           << state.bodies[0].position.z << std::endl;
  numBodies = state.bodies.size();
  switch (integrationMethod) {
    case EULER:
      if (computeType) {
        this->calculateForces = std::bind(&NbodyEngine::calculateForces_Euler_Parallel, this);
      } else {
        this->calculateForces = std::bind(&NbodyEngine::calculateForces_Euler_Serial, this);
      }
      break;

    case VERLET:
      std::cerr << "Error Not Implemented" << std::endl;
      break;

    default:
      std::cerr << "Error " << std::endl;
  }
}

void NbodyEngine::start() {
  FpsCounter fpsCounter("N-Body Engine");
  {
    std::unique_lock<std::mutex> lock(sharedData.get().mtx);
    sharedData.get().cv.wait(lock);
  }
  while (!sharedData.get().stopRequested) {
    fpsCounter.displayFps();
    updatePositions();
  }
}

void NbodyEngine::updatePositions() {
  calculateForces();
  writePositionsToVertexArray();
}

void NbodyEngine::writePositionsToVertexArray() {
  for (size_t i = 0; i < numBodies; ++i) {
    glm::dvec3 tmpVector = state.bodies[i].position;
    // std::cout << tmpVector.x << tmpVector.y << tmpVector.z << std::endl;
    sharedData.get().hostPointer[i * 3] = tmpVector.x;
    sharedData.get().hostPointer[(i * 3) + 1] = tmpVector.y;
    sharedData.get().hostPointer[(i * 3) + 2] = tmpVector.z;
  }

  // for (size_t i = 0; i < numBodies * 3; ++i) {
  //   std::cout << vertexArray[i] << " , ";
  // }
  // std::cout << std::endl;
}

void NbodyEngine::calculateForces_Euler_Serial() {
  for (int i = 0; i < numBodies; i++) {
    glm::dvec3 tmpAcc = {0, 0, 0};
    for (int j = 0; j < numBodies; j++) {
      if (i != j) {
        glm::dvec3 distanceVector = state.bodies[j].position - state.bodies[i].position;
        tmpAcc += distanceVector
                  * ((state.gravityConstant * state.bodies[j].mass)
                     / pow(glm::length(distanceVector), 3));
        // std::cout << distanceVector.GetMagnitude() << std::endl;
      }
    }
    state.bodies[i].acceleration = tmpAcc;

    state.bodies[i].velocity += state.bodies[i].acceleration * state.timeStep;
    state.bodies[i].position += state.bodies[i].velocity * state.timeStep;

    // std::cout << "Body " << i << " | vel " << glm::length(state.bodies[i].velocity) << "| acc "
    //           << glm::length(state.bodies[i].acceleration) << "| pos " <<
    //           state.bodies[i].position.x
    //           << ", " << state.bodies[i].position.y << ", " << state.bodies[i].position.z
    //           << std::endl;
  }
}
