#pragma once

#include <Engine/Material.hpp>

class Lambertian : public Material
{
public:
  __device__
    Lambertian(const Vec3& a)
  {
    albedo = a;
  }

  __device__
    virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
  {
    // S = P + n + random
    Vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);

    scattered = Ray(rec.p, target - rec.p);
    attenuation = albedo;

    return true;
  }

  Vec3 albedo;
};
