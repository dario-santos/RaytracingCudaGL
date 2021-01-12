/*********************************************************************
  * \file   Camera.hpp
  * \brief  The camera class, defines a camera
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

#include <Engine/Ray.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Camera 
{
public:
  Vec3 origin;
  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;
  float lens_radius;
  float aspect;
  float vfov;
  float focus_dist;

  __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture, float focus_dist) 
  { 
    this->aspect = aspect;
    this->vfov = vfov;
    this->focus_dist = focus_dist;

    // vfov is top to bottom in degrees
    lens_radius = aperture / 2.0f;
    float theta = vfov * ((float)M_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;
    
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;
  }

  /**
   * GetRay
   *
   * \param s Value at the x, [0, 1]
   * \param t Value at the y, [0, 1]
   * \param randState The random object
   * \return The ray to the point of the screen Vec2(s, t)
   */
  __device__ Ray GetRay(float s, float t, curandState* randState) 
  {
    Vec3 rd = lens_radius * Camera::RandomInUnitDisk(randState);

    Vec3 offset = u * rd.x() + v * rd.y();
  
    return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
  }

  /**
   * UpdatePos
   *
   * \param lookfrom Position of the camera
   * \param lookat Position that the camera is looking at
   * \param vup The orientation of the camera
   */
  __device__ void UpdatePos(Vec3 lookfrom, Vec3 lookat, Vec3 vup)
  {
    float theta = vfov * ((float)M_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;

    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
    horizontal = 2.0f * half_width * focus_dist * u;
    vertical = 2.0f * half_height * focus_dist * v;
  }

private:
  /**
   * RandomInUnitDisk
   *
   * \param randomObject The random object
   * \return The ammout to change the radius lens_radius
   */
  __device__ static Vec3 RandomInUnitDisk(curandState* randomObject)
  {
    Vec3 p;
    
    do 
    {
      p = 2.0f * Vec3(curand_uniform(randomObject), curand_uniform(randomObject), 0) - Vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    
    return p;
  }
};
