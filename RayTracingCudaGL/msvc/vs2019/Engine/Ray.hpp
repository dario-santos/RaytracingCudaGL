/*********************************************************************
  * \file   Ray.hpp
  * \brief  Ray class, defines a ray
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

#include "./Math/Vec3.hpp"

class Ray
{
private:
  Vec3 origin;
  Vec3 direction;

public:
  __device__ Ray() 
  {
  }

  /**
   * origin
   *
   * \param origin The origin point of the ray
   * \param direction The direction vector of the ray
   */
  __device__ Ray(const Vec3& origin, const Vec3& direction) 
  { 
    this->origin = origin;
    this->direction = direction;
  }

  /**
   * GetOrigin
   *
   * \return Returns the origin point
   */
  __device__ Vec3 GetOrigin() const 
  { 
    return this->origin; 
  }
  
  /**
   * GetDirection
   *
   * \return Returns the direction vector
   */
  __device__ Vec3 GetDirection() const
  { 
    return this->direction; 
  }

  /**
   * GetPointAt
   * 
   * \param t The t to feed the ray function
   * \return Returns the point at t
   */
  __device__ Vec3 GetPointAt(float t) const 
  { 
    return this->origin + (t * this->direction); 
  }
};
