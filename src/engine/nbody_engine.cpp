#include "empyrean/engine/nbody_engine.hpp"

#include <math.h>

#include <iostream>

#include "empyrean/engine/engine_state.hpp"
#include "empyrean/utils/fps_counter.hpp"

NbodyEngine::NbodyEngine(EngineState state, int integrationMethod) : state(state) {
  // std::cout << state.bodies[0].position.x << state.bodies[0].position.y
  //           << state.bodies[0].position.z << std::endl;
  numBodies = state.bodies.size();
  switch (integrationMethod) {
    case EULER:
      this->calculateForces = std::bind(&NbodyEngine::calculateForcesWithDirectEuler, this);
      break;

    case VERLET:
      std::cerr << "Error Not Implemented" << std::endl;
      break;

    default:
      std::cerr << "Error " << std::endl;
  }
}

void NbodyEngine::start(float* vertexArray) {
  FpsCounter fpsCounter("N-Body Engine");

  while (true) {
    fpsCounter.displayFps();
    updatePositions(vertexArray);
  }
}

void NbodyEngine::updatePositions(float* vertexArray) {
  calculateForces();
  writePositionsToVertexArray(vertexArray);
}

void NbodyEngine::writePositionsToVertexArray(float* vertexArray) {
  if (vertexArray) {
    for (size_t i = 0; i < numBodies; ++i) {
      glm::dvec3 tmpVector = state.bodies[i].position;
      // std::cout << tmpVector.x << tmpVector.y << tmpVector.z << std::endl;
      vertexArray[i * 3] = tmpVector.x;
      vertexArray[(i * 3) + 1] = tmpVector.y;
      vertexArray[(i * 3) + 2] = tmpVector.z;
    }
  }
  // for (size_t i = 0; i < numBodies * 3; ++i) {
  //   std::cout << vertexArray[i] << " , ";
  // }
  // std::cout << std::endl;
}

void NbodyEngine::calculateForcesWithDirectEuler() {
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
