#pragma once

#include <Engine/Material.hpp>

class Metal : public Material
{
public:
  Vec3 albedo;
  float fuzz;

  __device__
  Metal(const Vec3& a, float f)
  {
    albedo = a;
    fuzz = f < 1 ? f : 1;
  }
  __device__
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
  {
    Vec3 reflected = Material::Reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }

};
