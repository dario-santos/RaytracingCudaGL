#pragma once

#include <Engine/Hitable.hpp>
#include <Engine/HitableList.hpp>
#include <Engine/Primitives/Sphere.hpp>
#include <Engine/Math/Vec3.hpp>
#include <Engine/Camera.hpp>

#include <Assets/Materials/Metal.hpp>
#include <Assets/Materials/Lambertian.hpp>
#include <Assets/Materials/Dielectric.hpp>

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Sphere** d_list, Sphere** d_world, Camera** d_camera, int nx, int ny, curandState* rand_state)
{
  if (!(threadIdx.x == 0 && blockIdx.x == 0))
    return;

  curandState local_rand_state = *rand_state;
  d_list[0] = new Sphere(Vec3(0, -1000.0, -1), 1000, new Lambertian(Vec3(0.5, 0.5, 0.5)));
  int i = 1;

  for (int a = -11; a < 11; a++)
  {
    for (int b = -11; b < 11; b++)
    {
      float choose_mat = RND;
      Vec3 center(a + RND, 0.2, b + RND);

      // Choose material
      if (choose_mat < 0.8f)
        d_list[i++] = new Sphere(center, 0.2, new Lambertian(Vec3(RND * RND, RND * RND, RND * RND)));
      else if (choose_mat < 0.95f)
        d_list[i++] = new Sphere(center, 0.2, new Metal(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
      else
        d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));

    }
  }

  d_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
  d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));
  d_list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));
  *rand_state = local_rand_state;
  *d_world = new HitableList(d_list, 22 * 22 + 1 + 3);

  Vec3 lookfrom(10, 2.5f, 15);
  Vec3 lookat(0, 0, 0);
  float dist_to_focus = (lookfrom - lookat).Length();
  float aperture = 0.2;
  *d_camera = new Camera(lookfrom, lookat, Vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
}

__global__ void update(Camera** d_cam, Vec3 v)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i == 0)
  {
    Vec3 lookfrom = Vec3(v.x(), (*d_cam)->origin.y(), (*d_cam)->origin.z());

    Vec3 lookat(0, 0, 0);

    (*d_cam)->UpdatePos(lookfrom, lookat, Vec3(0, 1, 0));
  }
}

