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

#include <Engine/Math/Util.cuh>

#include <Assets/Worlds/World1.cuh>

#include <Engine/Render/Render.cuh>

#include <glm/glm.hpp>
using glm::vec3;

// Variables
Sphere** d_list;
Sphere** d_world;

Camera** d_camera;

curandState* d_rand_state;
curandState* d_rand_state2;

float alpha = 0.0f;
int numObjects = 22 * 22 + 1 + 3;

/**
 * rand_init
 * 
 * \param rand_state The random object
 */
__global__ void rand_init(curandState* rand_state) 
{
  if(threadIdx.x == 0 && blockIdx.x == 0)
    curand_init(1984, 0, 0, rand_state);
}

extern "C" void launch_cudaRender(dim3 blocks, dim3 threads, unsigned int* g_odata, int max_x, int max_y)
{
  // NS
  int ns = 10;

  RenderRandInit<<<blocks, threads>>>(max_x, max_y, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Camera alfa
  alpha += 0.5f / 3.1415f;
  Vec3 v = Vec3(glm::cos(alpha) + 5, 2.5f, 15);

  update<<<blocks, threads >> > (d_camera, v);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  Render<<<blocks, threads>>>(g_odata, max_x, max_y, ns, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void create_world(int max_x, int max_y, int num_pixels)
{ 
  // allocate random state
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));  
  checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

  rand_init<<<1,1>>>(d_rand_state2);
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
