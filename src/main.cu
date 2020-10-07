#include <iostream>
#include <time.h>

#include "common.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ float hit_sphere(const point3& center, double radius, const ray& r) {
  vec3 oc = r.origin() - center;
  auto a = dot(r.direction(), r.direction());
  auto b = 2.0f * dot(oc, r.direction());
  auto c = dot(oc, oc) - radius*radius;
  auto disc = b*b - 4.0f*a*c;
  if (disc < 0) {
    return -1.0f;
  } else {
    return (-b - sqrt(disc)) / (2.0f * a);
  }
}

__device__ vec3 ray_color(const ray& r) {
  float t = hit_sphere(point3(0, 0, -1), 0.5, r);
  if (t > 0.0) {
    vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
    return 0.5f * color(N.x()+1, N.y()+1, N.z()+1);
  }
  vec3 unit_direction = unit_vector(r.direction());
  t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;

  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
  fb[pixel_index] = ray_color(r);
}

int main() {
  // Image
  const float aspect_ratio = 16.0f / 9.0f;
  const int image_width = 400;
  const int image_height = static_cast<int>(image_width / aspect_ratio);

  // Camera
  float viewport_height = 2.0f;
  float viewport_width = aspect_ratio * viewport_height;
  float focal_length = 1.0f;

  auto origin = point3(0.0f, 0.0f, 0.0f);
  auto horizontal = vec3(viewport_width, 0.0f, 0.0f);
  auto vertical = vec3(0.0f, viewport_height, 0.0f);
  auto lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - vec3(0.0f, 0.0f, focal_length);

  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = image_width*image_height;
  size_t fb_size = num_pixels*sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(image_width/tx+1,image_height/ty+1);
  dim3 threads(tx,ty);
  render <<<blocks, threads>>>(fb, image_width, image_height, lower_left_corner, horizontal, vertical, origin);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
  for (int j = image_height-1; j >= 0; j--) {
    for (int i = 0; i < image_width; i++) {
      size_t pixel_index = j*image_width + i;
      float r = fb[pixel_index].r();
      float g = fb[pixel_index].g();
      float b = fb[pixel_index].b();
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
      std:: cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  checkCudaErrors(cudaFree(fb));
}
