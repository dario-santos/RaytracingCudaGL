/*********************************************************************
  * \file   Sphere.hpp
  * \brief  Sphere class, represents a sphere in the scene
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

#include <Engine/Math/Vec3.hpp>
#include <Engine/Hitable.hpp>
#include <Engine/Material.hpp>

class Sphere : public Hitable 
{
private:
  Vec3 center;
  float radius;
  Material* mat_ptr;

public:
  __device__ Sphere() 
  {
  }

  __device__ Sphere(Vec3 center, float radius, Material* material)
  {
    this->center = center;
    this->radius = radius;
    this->mat_ptr = material;
  };

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

__device__ bool Sphere::Hit(const Ray& r, float t_min, float t_max, hit_record& rec) const
{
  Vec3 oc = r.GetOrigin() - center;
  float a = dot(r.GetDirection(), r.GetDirection());
  float b = dot(oc, r.GetDirection());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  
  // Was there an intersection?
  if(discriminant > 0) 
  {
    // First try the zero closer to the camera
    float temp = (-b - sqrt(discriminant)) / a;
   
    if (temp < t_max && temp > t_min) 
    {
      rec.t = temp;
      rec.p = r.GetPointAt(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }

    // Second try the zero closer to the camera
    temp = (-b + sqrt(discriminant)) / a;
    
    if (temp < t_max && temp > t_min) 
    {
      rec.t = temp;
      rec.p = r.GetPointAt(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }

  return false;
}
