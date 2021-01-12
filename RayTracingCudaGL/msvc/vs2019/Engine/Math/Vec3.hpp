/*********************************************************************
  * \file   Vec3.hpp
  * \brief  Vec3 class, defines a vector with three components
  *
  * \author Dário Santos
  * \date   January 2021
 ***********************************************************************/
#pragma once

#include <math.h>
#include <stdlib.h>
#include <iostream>

class Vec3 
{
private:
  /** Vector with the three components of the triple */
  float e[3];

public:
  __host__ __device__ Vec3() 
  {
  }

  __host__ __device__ Vec3(float x, float y, float z) 
  { 
    this->e[0] = x; 
    this->e[1] = y;
    this->e[2] = z;
  }
  
  /**
   * x
   *
   * \return Returns the x component of the vector.
   */
  __host__ __device__ inline float x() const 
  { 
    return e[0]; 
  }
  
  /**
   * y
   *
   * \return Returns the y component of the vector.
   */
  __host__ __device__ inline float y() const 
  { 
    return e[1]; 
  }
  
  /**
   * z
   *
   * \return Returns the z component of the vector.
   */
  __host__ __device__ inline float z() const 
  { 
    return e[2]; 
  }
  
  /**
   * r
   *
   * \return Returns the r component of the vector.
   */
  __host__ __device__ inline float r() const 
  { 
    return e[0]; 
  }

  /**
   * r
   *
   * \return Returns the g component of the vector.
   */
  __host__ __device__ inline float g() const 
  { 
    return e[1]; 
  }
 
  /**
   * b
   *
   * \return Returns the b component of the vector.
   */
  __host__ __device__ inline float b() const 
  { 
    return e[2]; 
  }

  /**
   * Length
   *
   * \return Returns the length of the the vector
   */
  __host__ __device__ inline float Length() const
  {
    return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  }

  /**
   * SquareLength
   *
   * \return Returns the length of the the vector
   */
  __host__ __device__ inline float SquareLength() const
  {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  __host__ __device__ inline const Vec3& operator+() const { return *this; }
  __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }
  __host__ __device__ inline float& operator[](int i) { return e[i]; };

  __host__ __device__ inline Vec3& operator+=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator-=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator*=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator/=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator*=(const float t);
  __host__ __device__ inline Vec3& operator/=(const float t);
};

__host__ __device__ inline Vec3 operator+(const Vec3& v1, const Vec3& v2) 
{
  return Vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}

__host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2) 
{
  return Vec3(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2) 
{
  return Vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, const Vec3& v2) 
{
  return Vec3(v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z());
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) 
{
  return Vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) 
{
  return Vec3(v.x() / t, v.y() / t, v.z() / t);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t) 
{
  return Vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2) 
{
  return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2) 
{
  return Vec3((v1.y() * v2.z() - v1.z() * v2.y()),
             (-(v1.x() * v2.z() - v1.z() * v2.x())),
             (v1.x() * v2.y() - v1.y() * v2.x()));
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v) 
{
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v) 
{
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3& v) 
{
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) 
{
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) 
{

  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) 
{
  float k = 1.0f / t;

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v) 
{
  return v / v.Length();
}
