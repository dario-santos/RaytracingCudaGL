#pragma once

#include <Engine/Material.hpp>

class Metal : public Material
{
private:
  Vec3 albedo;
  float fuzz;

public:
  __device__ Metal(const Vec3& a, float f)
  {
    albedo = a;
    fuzz = f < 1 ? f : 1;
  }

  /**
   * ToScatter
   *
   * \param r_in The incident ray
   * \param rec
   * \param attenuation The attenuation of the object
   * \param scattered
   * \param randState The random object
   *
   * \return True if a ray was scattered, false otherwise
   */
  __device__ virtual bool ToScatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
  {
    Vec3 reflected = Material::Reflect(unit_vector(r_in.GetDirection()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz * RandomInUnitSphere(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.GetDirection(), rec.normal) > 0.0f);
  }
};
