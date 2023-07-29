#include <math.h>

#include <iostream>

#include "empyrean/engine/nbody_engine.hpp"

void NbodyEngine::UpdatePositionsWithSerialEuler() {
  int numBodies = state.cosmicBodies.size();

  for (int i = 0; i < numBodies; i++) {
    for (int j = 0; j < numBodies; j++) {
      if (i != j) {
        RealVector distanceVector
            = GetDistance(state.cosmicBodies[j].position[0], state.cosmicBodies[i].position[0]);
        state.cosmicBodies[i].acceleration = distanceVector
                                             * ((state.gravityConstant * state.cosmicBodies[j].mass)
                                                / pow(distanceVector.GetMagnitude(), 3));
        // std::cout << distanceVector.GetMagnitude() << std::endl;
      }
    }
    state.cosmicBodies[i].velocity += state.cosmicBodies[i].acceleration * state.timeStep;
    state.cosmicBodies[i].position[0] += state.cosmicBodies[i].velocity * state.timeStep;
    // if (i == 1) {
    //   std::cout << "Body " << i << " velocity : " <<
    //   state.cosmicBodies[i].velocity.GetMagnitude()
    //             << ", Acc : " << state.cosmicBodies[i].acceleration.GetMagnitude() << std::endl;
    // }
  }
}