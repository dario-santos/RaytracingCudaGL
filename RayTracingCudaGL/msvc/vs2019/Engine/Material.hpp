#pragma once

struct hit_record;

#include <Engine/Ray.hpp>
#include <Engine/Hitable.hpp>

__device__ 
float schlick(float cosine, float ref_idx) 
{
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;

  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ 
bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) 
{
  Vec3 uv = unit_vector(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

  if(discriminant > 0) 
  {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  }
   
  return false;
}

#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__
Vec3 random_in_unit_sphere(curandState* local_rand_state) 
{
  Vec3 p;

  do {
    p = 2.0f * RANDVEC3 - Vec3(1, 1, 1);
  } while (p.squared_length() >= 1.0f);

  return p;
}

__device__ 
Vec3 reflect(const Vec3& v, const Vec3& n) 
{
  return v - 2.0f * dot(v, n) * n;
}

class Material {
public:
  __device__ 
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material 
{
public:
  __device__ 
  Lambertian(const Vec3& a) : albedo(a) 
  {
  }
  
  __device__ 
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const 
  {
    Vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered   = Ray(rec.p, target - rec.p);
    attenuation = albedo;
    
    return true;
  }

  Vec3 albedo;
};

class Metal : public Material {
public:
  __device__ 
  Metal(const Vec3& a, float f) : albedo(a) 
  { 
    if (f < 1) 
      fuzz = f; 
    else 
      fuzz = 1; 
  }
  __device__ 
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const 
  {
    Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }
  Vec3 albedo;
  float fuzz;
};

class Dielectric : public Material 
{
public:
  __device__ 
  Dielectric(float ri) : ref_idx(ri) 
  {
  }

  __device__ 
  virtual bool scatter(const Ray& r_in, const hit_record& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const 
  {
    Vec3 outward_normal;
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = Vec3(1.0, 1.0, 1.0);
    Vec3 refracted;
    float reflect_prob;
    float cosine;

    if(dot(r_in.direction(), rec.normal) > 0.0f) 
    {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx;
      cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
      cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else 
    {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / ref_idx;
      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
      reflect_prob = schlick(cosine, ref_idx);
    else
      reflect_prob = 1.0f;

    if (curand_uniform(local_rand_state) < reflect_prob)
      scattered = Ray(rec.p, reflected);
    else
      scattered = Ray(rec.p, refracted);

    return true;
  }

  float ref_idx;
};
