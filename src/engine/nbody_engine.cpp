#include "empyrean/engine/nbody_engine.hpp"

#include <iostream>
#include <math.h>

#include "empyrean/engine/engine_state.hpp"

NbodyEngine::NbodyEngine(EngineState state, int integrationMethod) {
  this->state = state;
  // std::cout << state.Bodies[0].position.x << state.Bodies[0].position.y
  //           << state.Bodies[0].position.z << std::endl;
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

void NbodyEngine::updatePositions(float* vertexArray) {
  calculateForces();
  writePositionsToVertexArray(vertexArray);
}

void NbodyEngine::writePositionsToVertexArray(float* vertexArray) {
  for (size_t i = 0; i < state.Bodies.size(); ++i) {
    glm::dvec3 tmpVector = state.Bodies[i].position;
    // std::cout << tmpVector.x << tmpVector.y << tmpVector.z << std::endl;
    vertexArray[i * 3] = tmpVector.x;
    vertexArray[(i * 3) + 1] = tmpVector.y;
    vertexArray[(i * 3) + 2] = tmpVector.z;
  }
  // for (size_t i = 0; i < state.Bodies.size() * 3; ++i) {
  //   std::cout << vertexArray[i] << " , ";
  // }
  // std::cout << std::endl;
}

void NbodyEngine::calculateForcesWithDirectEuler() {
  int numBodies = state.Bodies.size();

  for (int i = 0; i < numBodies; i++) {
    glm::dvec3 tmpAcc = {0, 0, 0};
    for (int j = 0; j < numBodies; j++) {
      if (i != j) {
        glm::dvec3 distanceVector = state.Bodies[j].position - state.Bodies[i].position;
        tmpAcc += distanceVector
                  * ((state.gravityConstant * state.Bodies[j].mass)
                     / pow(glm::length(distanceVector), 3));
        // std::cout << distanceVector.GetMagnitude() << std::endl;
      }
    }
    state.Bodies[i].acceleration = tmpAcc;
    state.Bodies[i].velocity += state.Bodies[i].acceleration * state.timeStep;
    state.Bodies[i].position += state.Bodies[i].velocity * state.timeStep;
    // std::cout << "Body " << i << " | vel " << glm::length(state.Bodies[i].velocity) << "| acc "
    //           << glm::length(state.Bodies[i].acceleration) << "| pos " <<
    //           state.Bodies[i].position.x
    //           << ", " << state.Bodies[i].position.y << ", " << state.Bodies[i].position.z
    //           << std::endl;
  }
}