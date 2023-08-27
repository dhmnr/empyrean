#include "empyrean/engine/nbody_engine.hpp"

#include <cuda_runtime.h>
#include <math.h>

#include <condition_variable>
#include <iostream>

#include "empyrean/engine/engine_state.hpp"
#include "empyrean/utils/fps_counter.hpp"
#include "empyrean/utils/structs.hpp"

NbodyEngine::NbodyEngine(InitialState state, SharedData &sharedData, int integrationMethod,
                         int useGpu)
    : state(state), sharedData(sharedData), useGpu(useGpu) {
  // std::cout << state.bodies[0].position.x << state.bodies[0].position.y
  //           << state.bodies[0].position.z << std::endl;
  // glm::dvec3 *position_h;
  // glm::dvec3 *acceleration_h;
  // glm::dvec3 *velocity_h;
  // double *mass_h;

  dvecBytes = state.objCount * sizeof(glm::dvec3);
  dBytes = state.objCount * sizeof(double);

  position_h = (glm::dvec3 *)malloc(dvecBytes);
  acceleration_h = (glm::dvec3 *)malloc(dvecBytes);
  velocity_h = (glm::dvec3 *)malloc(dvecBytes);
  mass_h = (double *)malloc(dBytes);

  for (int i = 0; i < state.objCount; i++) {
    position_h[i] = state.bodies[i].position;
    acceleration_h[i] = state.bodies[i].acceleration;
    velocity_h[i] = state.bodies[i].velocity;
    mass_h[i] = state.bodies[i].mass;
  }

  if (useGpu) {
    // glm::dvec3 *position_d;
    // glm::dvec3 *acceleration_d;
    // glm::dvec3 *velocity_d;
    // double *mass_d;
    initDeviceArrays();

  } else {
  }

  // switch (integrationMethod) {
  //   case EULER:

  //     break;

  //   case VERLET:
  //     std::cerr << "Error Not Implemented" << std::endl;
  //     break;

  //   default:
  //     std::cerr << "Error " << std::endl;
  // }
}

NbodyEngine::~NbodyEngine() {
  if (useGpu) {
    // Release device memory
    cudaFree(position_d);
    cudaFree(acceleration_d);
    cudaFree(velocity_d);
    cudaFree(mass_d);
  }
  // Release host memory
  free(position_h);
  free(acceleration_h);
  free(velocity_h);
  free(mass_h);
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
  if (useGpu) {
    calculateForces_Euler_Parallel();
  } else {
    calculateForces_Euler_Serial();
    writeToHostArray();
  }
}

void NbodyEngine::writeToHostArray() {
  for (size_t i = 0; i < state.objCount; ++i) {
    sharedData.get().hostPointer[i * 3] = position_h[i].x;
    sharedData.get().hostPointer[(i * 3) + 1] = position_h[i].y;
    sharedData.get().hostPointer[(i * 3) + 2] = position_h[i].z;
  }

  // for (size_t i = 0; i < state.objCount * 3; ++i) {
  //   std::cout << vertexArray[i] << " , ";
  // }
  // std::cout << std::endl;
}

void NbodyEngine::calculateForces_Euler_Serial() {
  for (int i = 0; i < state.objCount; i++) {
    glm::dvec3 tmpAcc = {0, 0, 0};
    for (int j = 0; j < state.objCount; j++) {
      if (i != j) {
        glm::dvec3 distanceVector = position_h[j] - position_h[i];
        tmpAcc += distanceVector
                  * ((state.gravityConstant * mass_h[j]) / pow(glm::length(distanceVector), 3));
        // std::cout << distanceVector.GetMagnitude() << std::endl;
      }
    }
    acceleration_h[i] = tmpAcc;

    velocity_h[i] += acceleration_h[i] * state.timeStep;
    position_h[i] += velocity_h[i] * state.timeStep;

    // std::cout << "Body " << i << " | vel " << glm::length(state.bodies[i].velocity) << "| acc "
    //           << glm::length(state.bodies[i].acceleration) << "| pos " <<
    //           state.bodies[i].position.x
    //           << ", " << state.bodies[i].position.y << ", " << state.bodies[i].position.z
    //           << std::endl;
  }
}
