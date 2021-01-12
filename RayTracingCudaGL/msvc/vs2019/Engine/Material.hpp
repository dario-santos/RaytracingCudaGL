/*********************************************************************
  * \file   Material.hpp
  * \brief  Material class, defines the super class of a material
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

struct hit_record;

#include <Engine/Ray.hpp>
#include <Engine/Hitable.hpp>

class Material 
{
public:
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
  __device__ virtual bool ToScatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;

  /**
   * Reflect
   *
   * \param v The vector to reflect
   * \param n The normal
   *
   * \return Vector v reflected
   */
  __device__ static Vec3 Reflect(const Vec3& v, const Vec3& n)
  {
    return v - 2.0f * dot(v, n) * n;
  }

  /**
   * Reflectance
   *
   * \param cosine The cossine of teta
   * \param ref_idx the refract index of the material
   *
   * \return The Schlick aproximation
   */
  __device__ static float Reflectance(float cosine, float ref_idx)
  {
    // Schlick aproximation
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;

    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
  }

  /**
   * Reflectance
   *
   * \param v The vector to refract
   * \param n The noraml
   * \param ni_over_nt The fraction of the refract indexes
   * \param refracted Output, the vector v refracted
   *
   * \return true if the vector v was refracted, false otherwise
   */
  __device__ static bool Refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted)
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

  /**
   * RandomInUnitSphere
   *
   * \param randObject The random object
   *
   * \return A random vector inside an unit sphere
   */
  __device__ static Vec3 RandomInUnitSphere(curandState* randObject)
  {
    Vec3 p;
    Vec3 randomVec;
    do {
      randomVec = Vec3(curand_uniform(randObject), curand_uniform(randObject), curand_uniform(randObject));

      p = 2.0f * randomVec - Vec3(1, 1, 1);
    } while (p.SquareLength() >= 1.0f);

    return p;
  }
};
