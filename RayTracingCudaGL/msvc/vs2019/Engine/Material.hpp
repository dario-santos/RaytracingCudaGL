#pragma once

struct hit_record;

#include <Engine/Ray.hpp>
#include <Engine/Hitable.hpp>

#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__
Vec3 random_in_unit_sphere(curandState* local_rand_state) 
{
  Vec3 p;

  do {
    p = 2.0f * RANDVEC3 - Vec3(1, 1, 1);
  } while (p.squared_length() >= 1.0f);

  return p;
}

class Material 
{
public:
  __device__ 
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;

  __device__
  static Vec3 Reflect(const Vec3& v, const Vec3& n)
  {
    return v - 2.0f * dot(v, n) * n;
  }

  __device__
  static float Reflectance(float cosine, float ref_idx)
  {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;

    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
  }

  __device__
  static bool Refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted)
  {
    Vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

    if (discriminant > 0)
    {
      refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
      return true;
    }

    return false;
  }
};
