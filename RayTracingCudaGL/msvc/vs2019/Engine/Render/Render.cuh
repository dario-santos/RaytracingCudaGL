#pragma once

/**
 * GetRayColor
 * 
 * \param local_rand_state The random object
 * \param world The List of objects
 * \param r The ray
 * \return The color of the ray r
 */
__device__ Vec3 GetRayColor(const Ray& r, Sphere** world, curandState* local_rand_state)
{
  Ray cur_ray = r;
  Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);

  // 50 levels of depth, can't be recursive because of the nature of CUDA
  for (int i = 0; i < 50; i++)
  {
    hit_record rec;

    // Was there a hit? 
    if((*world)->Hit(cur_ray, 0.001f, FLT_MAX, rec))
    {
      Ray scattered;
      Vec3 attenuation;
      
      // Was the ray scattered by the material?
      if(rec.mat_ptr->ToScatter(cur_ray, rec, attenuation, scattered, local_rand_state))
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
      // Draw the background
      Vec3 unit_direction = unit_vector(cur_ray.GetDirection());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
      
      return cur_attenuation * c;
    }
  }

  // Exceeded the maximum depth level
  return Vec3(0.0, 0.0, 0.0);
}

/**
 * RenderRandInit
 *
 * \param max_x Width of the window
 * \param max_y Height of the window
 * \param rand_state An random object
 * \return A new random object
 */
__global__ void RenderRandInit(int max_x, int max_y, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y))
    return;

  int pixel_index = j * max_x + i;
  curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

/**
 * Render
 *
 * \param fb The output color buffer
 * \param max_x Width of the window
 * \param max_y Height of the window
 * \param ns Number of samples
 * \param cam The camera of the world
 * \param world The object list
 * \param curandState The random object
 */
__global__ void Render(unsigned int* fb, int max_x, int max_y, int ns, Camera** cam, Sphere** world, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y))
    return;

  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  Vec3 col(0, 0, 0);

  // Sends NS rays to that pixel
  for(int s = 0 ; s < ns ; s++)
  {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

    Ray r = (*cam)->GetRay(u, v, &local_rand_state);
    col += GetRayColor(r, world, &local_rand_state);
  }
  // Average of the antialiasing
  col /= float(ns);

  // Gama correction
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);

  rand_state[pixel_index] = local_rand_state;

  // VEC3 to INT
  col *= float(255);
  uchar4 c4 = make_uchar4(col.r(), col.g(), col.b(), 0);
  fb[pixel_index] = RgbToInt(c4.x, c4.y, c4.z);
}
