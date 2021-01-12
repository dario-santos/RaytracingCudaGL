#pragma once

#include <Engine/Material.hpp>

class Lambertian : public Material
{
private:
  Vec3 albedo;

public:
  __device__ Lambertian(const Vec3& a)
  {
    albedo = a;
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
    // S = P + n + random
    Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);

    scattered = Ray(rec.p, target - rec.p);
    attenuation = albedo;

    return true;
  }
};
