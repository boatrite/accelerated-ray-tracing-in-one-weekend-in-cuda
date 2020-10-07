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

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

enum class LightingType { normals, alt_approx_lambertian, true_lambertian, hacky_lambertian };

__device__ inline float degrees_to_radians(float degrees) {
  return degrees * pi / 180.0;
}

__device__ inline float clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

__device__ inline float random_float(curandState *rand_state) {
  // Returns a random real in (0,1].
  return curand_uniform(rand_state);
}

__device__ inline float random_float(curandState *rand_state, float min, float max) {
  // Returns a random real in (min,max].
  return min + (max - min) * random_float(rand_state);
}

__device__ inline int random_int(curandState *rand_state, int min, int max) {
  // Returns a random integer in (min,max].
  return static_cast<int>(random_float(rand_state, min, max+1));
}

#include "ray.cuh"
#include "vec3.cuh"

#endif
