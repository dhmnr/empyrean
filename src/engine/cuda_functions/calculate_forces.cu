#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <iostream>

#include "empyrean/engine/nbody_engine.hpp"

__global__ void calculatePositions(glm::dvec3 *position, glm::dvec3 *velocity, float *devicePointer,
                                   double timeStep, int numBodies) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < numBodies) {
    position[idx] += velocity[idx] * timeStep;
    // printf("%f\n", (float)glm::length(position[idx]));
    devicePointer[idx * 3] = position[idx].x;
    devicePointer[(idx * 3) + 1] = position[idx].y;
    devicePointer[(idx * 3) + 2] = position[idx].z;
  }
}

__global__ void calculateVelocity(glm::dvec3 *acceleration, glm::dvec3 *velocity, int numBodies,
                                  double timeStep) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < numBodies) {
    glm::dvec3 tmp(0, 0, 0);
    for (int i = 0; i < numBodies; i++) {
      // printf("%f\n", (float)glm::length(acceleration[(idx * numBodies) + i]));

      tmp += acceleration[(idx * numBodies) + i] * timeStep;
    }
    // printf("%f %f %f\n", tmp.x, tmp.y, tmp.z);

    velocity[idx] += tmp;
    // printf("%f\n", (float)glm::length(tmp));
  }
}

__global__ void calculateForces(glm::dvec3 *position, glm::dvec3 *acceleration, double *mass,
                                double gravConst, int numBodies) {  //
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < numBodies && idy < numBodies) {
    if (idx != idy) {
      glm::dvec3 dist = position[idy] - position[idx];
      double dist_mag = glm::length(dist);
      acceleration[idx * numBodies + idy]
          = dist * ((gravConst * mass[idy]) / (dist_mag * dist_mag * dist_mag));
    }
    // printf("%d\n", idx * numBodies + idy);
  }
}

void NbodyEngine::initDeviceArrays() {
  cudaMalloc(&position_d, dvecBytes);
  cudaMalloc(&acceleration_d, state.objCount * dvecBytes);
  cudaMalloc(&velocity_d, dvecBytes);
  cudaMalloc(&mass_d, dBytes);

  cudaMemcpy(position_d, position_h, dvecBytes, cudaMemcpyHostToDevice);
  // cudaMemcpy(acceleration_d, acceleration_h, state.objCount * dvecBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_d, velocity_h, dvecBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(mass_d, mass_h, dBytes, cudaMemcpyHostToDevice);
}

void NbodyEngine::calculateForces_Euler_Parallel() {
  int threadlen = 8;
  dim3 blockSize(threadlen, threadlen);

  int blocklen = (int)ceil((float)state.objCount / threadlen);
  dim3 gridSize(blocklen, blocklen);
  // std::cout << threadlen << blocklen << std::endl;
  int numBodies = state.objCount;
  // float *devicePtr = ;

  // Execute the kernel
  calculateForces<<<gridSize, blockSize>>>(position_d, acceleration_d, mass_d,
                                           state.gravityConstant, numBodies);
  calculateVelocity<<<blocklen, threadlen>>>(acceleration_d, velocity_d, numBodies, state.timeStep);
  calculatePositions<<<blocklen, threadlen>>>(
      position_d, velocity_d, (float *)sharedData.get().devicePointer, state.timeStep, numBodies);
}
