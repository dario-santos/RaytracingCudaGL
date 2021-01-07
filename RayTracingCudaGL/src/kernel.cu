#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>

#include <curand_kernel.h>

#include <Engine/Math/Vec3.hpp>

#include <Engine/Primitives/Sphere.hpp>

#include <Engine/Ray.hpp>
#include <Engine/HitableList.hpp>
#include <Engine/Camera.hpp>
#include <Engine/Material.hpp>

#include <Engine/Utils/CudaHelper.hpp>

#include <glm/glm.hpp>
using glm::vec3;

// Variables
Sphere** d_list;
Sphere** d_world;
Camera** d_camera;
curandState* d_rand_state;
curandState* d_rand_state2;
float alpha = 0.0f;
int numObjects = 1 * 22 + 1 + 3;

// clamp x to range [a, b]
__device__
float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

__device__
int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__
int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ 
Vec3 color(const Ray& r, Sphere** world, curandState* local_rand_state)
{
  Ray cur_ray = r;
  Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);
  
  for(int i = 0; i < 50; i++) 
  {
    hit_record rec;
    
    if((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) 
    {
      Ray scattered;
      Vec3 attenuation;
      if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) 
      {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else 
      {
        return Vec3(0.0, 0.0, 0.0);
      }
    }
    else 
    {
      Vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return Vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ 
void rand_init(curandState* rand_state) 
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    curand_init(1984, 0, 0, rand_state);
}

__global__ 
void render_init(int max_x, int max_y, curandState* rand_state) 
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y)) 
    return;
  
  int pixel_index = j * max_x + i;
  // Original: Each thread gets same seed, a different sequence number, no offset
  // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
  // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
  // performance improvement of about 2x!
  curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__
void render(unsigned int* fb, int max_x, int max_y, int ns, Camera** cam, Sphere** world, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) 
    return;
  
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  Vec3 col(0, 0, 0);
  
  for(int s = 0 ; s < ns ; s++) 
  {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

    Ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col += color(r, world, &local_rand_state);
  }

  rand_state[pixel_index] = local_rand_state;
  col /= float(ns);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);

  // VEC3 to INT
  col *= float(255);
  uchar4 c4 = make_uchar4(col.r(), col.g(), col.b(), 0);
  fb[pixel_index] = rgbToInt(c4.x, c4.y, c4.z);
}

#define RND (curand_uniform(&local_rand_state))

__global__ 
void create_world(Sphere** d_list, Sphere** d_world, Camera** d_camera, int nx, int ny, curandState* rand_state)
{
  if (!(threadIdx.x == 0 && blockIdx.x == 0))
    return;

  curandState local_rand_state = *rand_state;
  d_list[0] = new Sphere(Vec3(0, -1000.0, -1), 1000, new Lambertian(Vec3(0.5, 0.5, 0.5)));
  int i = 1;

  for(int a = 0; a < 1; a++) 
  {
    for(int b = -11; b < 11; b++) 
    {
      float choose_mat = RND;
      Vec3 center(a + RND, 0.2, b + RND);

      // Choose material
      if(choose_mat < 0.8f) 
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
  *d_world = new HitableList(d_list, 1 * 22 + 1 + 3);

  Vec3 lookfrom(13, 2, 3);
  Vec3 lookat(0, 0, 0);
  float dist_to_focus = 10.0; (lookfrom - lookat).length();
  float aperture = 0.1;
  *d_camera = new Camera(lookfrom, lookat, Vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
}

__global__
void update(Camera** d_cam, Vec3 v)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(i == 0)
  {
    Vec3 lookfrom = Vec3(v.x(), (*d_cam)->origin.z(), v.y());
      
    Vec3 lookat(0, 0, 0);
    
    (*d_cam)->UpdatePos(lookfrom, lookat, Vec3(0, 1, 0));
  }
}


extern "C"
void free_world() 
{
  for(int i = 0; i < 22 * 22 + 1 + 3; i++) 
  {
    //delete ((Sphere*)d_list[i])->mat_ptr;
    //delete d_list[i];
  }

  //delete* d_world;
  //delete* d_camera;
}

extern "C" 
void launch_cudaRender(dim3 blocks, dim3 threads, unsigned int* g_odata, int max_x, int max_y)
{
  int ns = 500;

  render_init<<<blocks, threads>>>(max_x, max_y, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Camera alfa
  alpha += 0.5f;
  Vec3 v = Vec3(glm::cos(alpha) + 5, 7.0f, 7.0f);

  update<<<blocks, threads>>>(d_camera, v);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(g_odata, max_x, max_y, ns, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

extern "C"
void create_world(int max_x, int max_y, int num_pixels)
{ 
  // allocate random state
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));  
  checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

  rand_init<<<1, 1>>>(d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make our world of hitables & the camera
  checkCudaErrors(cudaMalloc((void**)&d_list, numObjects *sizeof(Sphere*)));
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Sphere*)));
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

  create_world<<<1,1>>>(d_list, d_world, d_camera, max_x, max_y, d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}
