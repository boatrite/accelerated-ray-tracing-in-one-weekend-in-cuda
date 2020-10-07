#ifndef CAMERA_H
#define CAMERA_H

#include "common.cuh"

class camera {
  public:
    __device__ camera(
      point3 lookfrom,
      point3 lookat,
      vec3 vup,
      float vfov, // vertical field-of-view in degrees
      float aspect_ratio,
      float aperture,
      float focus_dist
    ) {
      // This is the complete implementation:
      // auto theta = degrees_to_radians(vfov);
      // auto h = tan(theta/2);
      // auto viewport_height = 2.0 * h;
      // auto viewport_width = aspect_ratio * viewport_height;

      // w = unit_vector(lookfrom - lookat);
      // u = unit_vector(cross(vup, w));
      // v = cross(w, u);

      // origin = lookfrom;
      // horizontal = focus_dist * viewport_width * u;
      // vertical = focus_dist * viewport_height * v;
      // lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

      // lens_radius = aperture / 2;


      // This is the temp implementation:
      float viewport_height = 2.0f;
      float viewport_width = aspect_ratio * viewport_height;
      float focal_length = 1.0f;

      origin = point3(0.0f, 0.0f, 0.0f);
      horizontal = vec3(viewport_width, 0.0f, 0.0f);
      vertical = vec3(0.0f, viewport_height, 0.0f);
      lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - vec3(0.0f, 0.0f, focal_length);
    }

    __device__ ray get_ray(float s, float t, curandState *rand_state) const {
      // This is the complete implementation:
      // vec3 rd = lens_radius * random_in_unit_disk(rand_state);
      // vec3 offset = u * rd.x() + v * rd.y();
      // return ray(
        // origin + offset,
        // lower_left_corner + s*horizontal + t*vertical - origin - offset
      // );


      // This is the temp implementation:
      return ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
    }

  private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};
#endif
