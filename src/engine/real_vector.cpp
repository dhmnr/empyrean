#include "empyrean/engine/real_vector.hpp"

#include <math.h>

#include <iostream>

RealVector::RealVector(double x, double y, double z) : x(x), y(y), z(z) {}

double RealVector::GetMagnitude() { return sqrt(x * x + y * y + z * z); }

RealVector RealVector::GetUnitVector() {
  double magnitude = GetMagnitude();
  return RealVector(x / magnitude, y / magnitude, z / magnitude);
}

// Operator overloading for vector addition
RealVector RealVector::operator+(RealVector& rv) {
  return RealVector(x + rv.x, y + rv.y, z + rv.z);
}

// Operator overloading for scalar multiplication
RealVector RealVector::operator*(double val) { return RealVector(x * val, y * val, z * val); }

RealVector& RealVector::operator+=(const RealVector& rv) {
  x += rv.x;
  y += rv.y;
  z += rv.z;
  return *this;
}

std::ostream& operator<<(std::ostream& os, const RealVector& vec) {
  os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
  return os;
}

// Standalone function outside the class
RealVector GetDistance(const RealVector& v1, const RealVector& v2) {
  double dx = v1.x - v2.x;
  double dy = v1.y - v2.y;
  double dz = v1.z - v2.z;

  return RealVector(dx, dy, dz);
}
