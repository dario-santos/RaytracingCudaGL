/*********************************************************************
  * \file   HitableList.hpp
  * \brief  HitableList class, a list of hittable objects
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

#include <Engine/Hitable.hpp>
#include <Engine/Primitives/Sphere.hpp>

class HitableList : public Sphere
{
private:
  Sphere** list;
  int list_size;

public:
  __device__ HitableList() 
  {
  }

  __device__ HitableList(Sphere** l, int n) 
  { 
    list = l;
    list_size = n;
  }

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
  __device__ virtual bool Hit(const Ray& r, float tmin, float tmax, hit_record& rec) const;
};

__device__ bool HitableList::Hit(const Ray& r, float t_min, float t_max, hit_record& rec) const
{
  hit_record temp_rec;
  bool hit_anything = false;
  float closest_so_far = t_max;

  for(int i = 0; i < list_size; i++) 
  {
    if(list[i]->Hit(r, t_min, closest_so_far, temp_rec))
    {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}
