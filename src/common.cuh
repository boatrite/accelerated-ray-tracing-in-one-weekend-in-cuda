#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <cstring>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

enum class LightingType { normals, alt_approx_lambertian, true_lambertian, hacky_lambertian };

__device__ inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0;
}

__device__ inline double clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

__device__ inline double random_double(curandState *rand_state) {
  // Returns a random real in (0,1].
  return curand_uniform(rand_state);
}

__device__ inline double random_double(curandState *rand_state, double min, double max) {
  // Returns a random real in (min,max].
  return min + (max - min) * random_double(rand_state);
}

__device__ inline int random_int(curandState *rand_state, int min, int max) {
  // Returns a random integer in (min,max].
  return static_cast<int>(random_double(rand_state, min, max+1));
}

#include "ray.cuh"
#include "vec3.cuh"

#endif
