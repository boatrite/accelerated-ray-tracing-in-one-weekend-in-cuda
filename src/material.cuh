#ifndef MATERIAL_H
#define MATERIAL_H

#include "common.cuh"

#include "hittable.cuh"

class material {
  public:
    __device__ virtual bool scatter(
      const ray& r, const hit_record& rec, color& attenuation, ray& scattered, curandState *rand_state, const LightingType& lighting_type
    ) const = 0;
};

class lambertian : public material {
  public:
    __device__ lambertian(const color& a) : albedo(a) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *rand_state, const LightingType& lighting_type
      ) const override {

      vec3 scatter_direction;
      switch (lighting_type) {
        case LightingType::hacky_lambertian:
          scatter_direction = rec.normal + random_in_unit_sphere(rand_state);
          break;
        case LightingType::true_lambertian:
          scatter_direction = rec.normal + random_unit_vector(rand_state);
          break;
        case LightingType::alt_approx_lambertian:
          scatter_direction = rec.normal + random_in_hemisphere(rand_state, rec.normal);
          break;
        default:
          scatter_direction = rec.normal + random_unit_vector(rand_state);
          break;
      }

      scattered = ray(rec.p, scatter_direction);
      attenuation = albedo;
      return true;
    }

  public:
    color albedo;
};

class metal : public material {
  public:
    __device__ metal(const color& a, float f) : albedo(a), fuzz(f) {}

    __device__ virtual bool scatter(
      const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *rand_state, const LightingType& lighting_type
    ) const override {
      vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
      scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(rand_state));
      attenuation = albedo;
      return (dot(scattered.direction(), rec.normal) > 0);
    }

  public:
    color albedo;
    float fuzz;
};

class dielectric : public material {
  public:
    __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ virtual bool scatter(
      const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *rand_state, const LightingType& lighting_type
    ) const override {
      attenuation = color(1.0, 1.0, 1.0);
      float refraction_ratio = rec.front_face ? (1.0/ir) : ir;
      vec3 unit_direction = unit_vector(r_in.direction());
      float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
      float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

      bool cannot_refract = refraction_ratio * sin_theta > 1.0;
      vec3 direction;
      if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(rand_state)) {
        direction = reflect(unit_direction, rec.normal);
      } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
      }

      scattered = ray(rec.p, direction);
      return true;
    }
  public:
    float ir; // Index of refraction

  private:
    __device__ static float reflectance(float cosine, float ref_idx) {
      // Use Schlick's approximation for reflectance.
      auto r0 = (1-ref_idx) / (1+ref_idx);
      r0 = r0*r0;
      return r0 + (1-r0)*pow((1 - cosine), 5);
    }
};

#endif
