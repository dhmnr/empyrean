#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void calculateForcesEuler(glm::dvec3* position, glm::dvec3* acceleration,
                                     glm::dvec3* velocity, double* mass, double gravityConstant,
                                     double timeStep, int numBodies);