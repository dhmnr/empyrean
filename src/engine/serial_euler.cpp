#include <math.h>

#include "empyrean/engine/nbody_engine.hpp"

void NbodyEngine::UpdatePositionsWithSerialEuler() {
  int numBodies = state.cosmicBodies.size();

  for (int i = 0; i < numBodies; i++) {
    for (int j = 0; j < numBodies; j++) {
      if (i != j) {
        RealVector distanceVector
            = GetDistance(state.cosmicBodies[i].position[0], state.cosmicBodies[j].position[0]);
        state.cosmicBodies[i].acceleration
            += distanceVector
               * ((state.gravityConstant * state.cosmicBodies[i].mass)
                  / pow(distanceVector.magnitude, 3));
        state.cosmicBodies[i].velocity += state.cosmicBodies[i].acceleration * state.timeStep;
        state.cosmicBodies[i].position[0] += state.cosmicBodies[i].velocity * state.timeStep;
      }
    }
  }
}