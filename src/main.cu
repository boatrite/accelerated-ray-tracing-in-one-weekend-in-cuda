#include <iostream>
#include <time.h>
#include <curand_kernel.h>

#include "common.cuh"

#include "camera.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"

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

__device__ vec3 ray_color(const ray& r, hittable **world) {
  hit_record rec;
  if ((*world)->hit(r, 0, infinity, rec)) {
    return 0.5f * (rec.normal + color(1,1,1));
  }

  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ void write_color(vec3 *fb, int pixel_index, color pixel_color, int samples_per_pixel) {
  // Divide the color by the number of samples
  auto scale = 1.0f / samples_per_pixel;
  pixel_color *= scale;

  fb[pixel_index] = pixel_color;
}

__global__ void render(vec3 *fb, int max_x, int max_y, int samples_per_pixel, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hittable **world, camera **cam, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  curandState local_rand_state = rand_state[pixel_index];

  color pixel_color(0,0,0);
  for (int s = 0; s < samples_per_pixel; ++s) {
    float u = float(i + random_float(&local_rand_state)) / float(max_x-1);
    float v = float(j + random_float(&local_rand_state)) / float(max_y-1);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    // ray r = (*cam)->get_ray(u, v, rand_state);
    pixel_color += ray_color(r, world);
  }

  write_color(fb, pixel_index, pixel_color, samples_per_pixel);
}

__global__ void create_world(hittable **d_list, hittable **d_world, const float aspect_ratio, camera **d_camera) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
    *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *d_world    = new hittable_list(d_list, 2);

    point3 lookfrom(13, 2, 3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    *d_camera = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
  }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
  delete *(d_list);
  delete *(d_list+1);
  delete *d_world;
  delete *d_camera;
}

int main() {
  // Image
  const float aspect_ratio = 16.0f / 9.0f;
  const int image_width = 400;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 100;

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

  // allocate FB
  int num_pixels = image_width*image_height;
  size_t fb_size = num_pixels*sizeof(vec3);
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // World
  hittable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
  hittable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
  create_world<<<1,1>>>(d_list, d_world, aspect_ratio, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Rand
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

  clock_t start, stop;
  start = clock();

  // Render our buffer
  dim3 blocks(image_width/tx+1,image_height/ty+1);
  dim3 threads(tx,ty);

  render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, lower_left_corner, horizontal, vertical, origin, d_world, d_camera, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
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

  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(fb));
}
