#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "empyrean/engine/nbody_engine.hpp"

__global__ void calculateForcesEuler(glm::dvec3 *position, glm::dvec3 *acceleration,
                                     glm::dvec3 *velocity, double *mass, double gravityConstant,
                                     double timeStep, int numBodies) {
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
  }
}

// void NbodyEngine::InitCuda() {

// }

void NbodyEngine::calculateForces_Euler_Parallel() {
  // Host vectors
  glm::dvec3 *position_h;
  glm::dvec3 *acceleration_h;
  glm::dvec3 *velocity_h;
  double *mass_h;

  // Device input vectors
  glm::dvec3 *position_d;
  glm::dvec3 *acceleration_d;
  glm::dvec3 *velocity_d;
  double *mass_d;

  // Size, in bytes, of each vector
  size_t vecbytes = numBodies * sizeof(glm::dvec3);
  size_t bytes = numBodies * sizeof(double);

  // Allocate memory for each vector on host
  position_h = (glm::dvec3 *)malloc(vecbytes);
  acceleration_h = (glm::dvec3 *)malloc(vecbytes);
  velocity_h = (glm::dvec3 *)malloc(vecbytes);
  mass_h = (double *)malloc(bytes);

  // Allocate memory for each vector on GPU
  cudaMalloc(&position_d, vecbytes);
  cudaMalloc(&acceleration_d, vecbytes);
  cudaMalloc(&velocity_d, vecbytes);
  cudaMalloc(&mass_d, bytes);

  // Initialize vectors on host
  for (int i = 0; i < numBodies; i++) {
    position_h[i] = state.bodies[i].position;
    acceleration_h[i] = state.bodies[i].acceleration;
    velocity_h[i] = state.bodies[i].velocity;
    mass_h[i] = state.bodies[i].mass;
  }

  // Copy host vectors to device
  cudaMemcpy(position_d, position_h, vecbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(acceleration_d, acceleration_h, vecbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_d, velocity_h, vecbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(mass_d, mass_h, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)numBodies / blockSize);

  // Execute the kernel
  calculateForcesEuler<<<gridSize, blockSize>>>(position_d, acceleration_d, velocity_d, mass_d,
                                                state.gravityConstant, state.timeStep, numBodies);

  // Copy array back to host
  cudaMemcpy(position_h, position_d, vecbytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(acceleration_h, acceleration_d, vecbytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(velocity_h, velocity_d, vecbytes, cudaMemcpyDeviceToHost);
  // cudaMemcpy(mass_d, mass_h, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < numBodies; i++) {
    state.bodies[i].position = position_h[i];
    state.bodies[i].acceleration = acceleration_h[i];
    state.bodies[i].velocity = velocity_h[i];
    // state.bodies[i].mass = mass_h[i];
  }

  // Release device memory
  cudaFree(position_d);
  cudaFree(acceleration_d);
  cudaFree(velocity_d);
  cudaFree(mass_d);

  // Release host memory
  free(position_h);
  free(acceleration_h);
  free(velocity_h);
  free(mass_h);
}
