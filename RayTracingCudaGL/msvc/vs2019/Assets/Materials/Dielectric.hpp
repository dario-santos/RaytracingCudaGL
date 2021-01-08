#pragma once

#include <Engine/Material.hpp>

class Dielectric : public Material
{
public:
  __device__
  Dielectric(float ri)
  {
    ref_idx = ri;
  }

  __device__
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
  {
    Vec3 outward_normal;
    Vec3 reflected = Material::Reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = Vec3(1.0, 1.0, 1.0);
    Vec3 refracted;
    float reflect_prob;
    float cosine;

    // Are we inside?
    if (dot(r_in.direction(), rec.normal) > 0.0f)
    {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx;
      cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
      cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else
    {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / ref_idx;
      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }


    if (Material::Refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
      reflect_prob = Material::Reflectance(cosine, ref_idx);
    else
      reflect_prob = 1.0f;

    if (curand_uniform(local_rand_state) < reflect_prob)
      scattered = Ray(rec.p, reflected);
    else
      scattered = Ray(rec.p, refracted);

    return true;
  }

  float ref_idx;
};
