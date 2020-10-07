#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"

class sphere : public hittable {
  public:
    __device__ sphere() {}
    __device__ sphere(point3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};
    __device__ virtual bool hit(
        const ray& r, float tmin, float tmax, hit_record& rec) const override;

  public:
    point3 center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  vec3 oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius*radius;
  auto disc = half_b*half_b - a*c;

  if (disc > 0) {
    auto root = sqrt(disc);

    auto temp = (-half_b - root) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      return true;
    }

    temp = (-half_b + root) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }

  return false;
}

#endif
