#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "empyrean/engine/nbody_engine.hpp"

__global__ void calculateForcesEuler(glm::dvec3 *position, glm::dvec3 *acceleration,
                                     glm::dvec3 *velocity, double *mass, double gravityConstant,
                                     double timeStep, int numBodies, float *devicePointer) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < numBodies) {
    glm::dvec3 tmpAcc = {0, 0, 0};
    for (int i = 0; i < numBodies; i++) {
      if (i != idx) {
        glm::dvec3 distanceVector = position[i] - position[idx];
        tmpAcc
            += distanceVector * ((gravityConstant * mass[i]) / pow(glm::length(distanceVector), 3));
      }
    }
    acceleration[idx] = tmpAcc;
    velocity[idx] += acceleration[idx] * timeStep;
    position[idx] += velocity[idx] * timeStep;

    devicePointer[idx * 3] = position[idx].x;
    devicePointer[(idx * 3) + 1] = position[idx].y;
    devicePointer[(idx * 3) + 2] = position[idx].z;
  }
}

void NbodyEngine::initDeviceArrays() {
  cudaMalloc(&position_d, dvecBytes);
  cudaMalloc(&acceleration_d, dvecBytes);
  cudaMalloc(&velocity_d, dvecBytes);
  cudaMalloc(&mass_d, dBytes);

  cudaMemcpy(position_d, position_h, dvecBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(acceleration_d, acceleration_h, dvecBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_d, velocity_h, dvecBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(mass_d, mass_h, dBytes, cudaMemcpyHostToDevice);
}

void NbodyEngine::calculateForces_Euler_Parallel() {
  // Host vectors

  // Device input vectors

  // Size, in bytes, of each vector

  // Allocate memory for each vector on GPU

  // Initialize vectors on host

  // Copy host vectors to device

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)state.objCount / blockSize);

  // Execute the kernel
  calculateForcesEuler<<<gridSize, blockSize>>>(
      position_d, acceleration_d, velocity_d, mass_d, state.gravityConstant, state.timeStep,
      state.objCount, (float *)sharedData.get().devicePointer);

  // Copy array back to host
  // cudaMemcpy(position_h, position_d, dvecBytes, cudaMemcpyDeviceToHost);
  // cudaMemcpy(acceleration_h, acceleration_d, dvecBytes, cudaMemcpyDeviceToHost);
  // cudaMemcpy(velocity_h, velocity_d, dvecBytes, cudaMemcpyDeviceToHost);
  // // cudaMemcpy(mass_d, mass_h, bytes, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < state.objCount; i++) {
  //   state.bodies[i].position = position_h[i];
  //   state.bodies[i].acceleration = acceleration_h[i];
  //   state.bodies[i].velocity = velocity_h[i];
  //   // state.bodies[i].mass = mass_h[i];
  // }
}
