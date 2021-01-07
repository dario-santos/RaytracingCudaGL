#pragma once

#include "./Math/Vec3.hpp"

class Ray
{
public:
  __device__ Ray() {}
  __device__ Ray(const Vec3& a, const Vec3& b) 
  { 
    orig = a; 
    dir = b; 
  }
  __device__ Vec3 origin() const 
  { 
    return orig; 
  }
  
  __device__ Vec3 direction() const { return dir; }
  __device__ Vec3 point_at_parameter(float t) const { return orig + t*dir; }

  Vec3 orig;
  Vec3 dir;
};
