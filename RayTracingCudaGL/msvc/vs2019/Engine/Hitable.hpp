/*********************************************************************
  * \file   Hitable.hpp
  * \brief  Hitable class, defines the super class of a hittable object
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

#include <Engine/Ray.hpp>

class Material;

struct hit_record
{
  float t;
  Vec3 p;
  Vec3 normal;
  Material* mat_ptr;
};

class Hitable
{
public:
  /**
   * Hit
   *
   * \param r The incident ray
   * \param tmin The minimum value accepted of t
   * \param tmax The maximum value accepted of t
   * \param rec Out parameter, the the information of the ray
   *
   * \return Returns true if there was a hit, false otherwise
   */
  __device__ virtual bool Hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};
