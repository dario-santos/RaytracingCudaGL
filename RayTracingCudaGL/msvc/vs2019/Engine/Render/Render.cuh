#pragma once

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__
Vec3 color(const Ray& r, Sphere** world, curandState* local_rand_state)
{
  Ray cur_ray = r;
  Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);

  for (int i = 0; i < 50; i++)
  {
    hit_record rec;

    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
    {
      Ray scattered;
      Vec3 attenuation;
      if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
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

  for (int s = 0; s < ns; s++)
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
