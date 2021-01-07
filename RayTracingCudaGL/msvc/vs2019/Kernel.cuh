#pragma once

#include <vector_types.h>

extern "C" void launch_cudaRender(dim3 blocks, dim3 threads, unsigned int* g_odata, int max_x, int max_y);

extern "C" void create_world(int max_x, int max_y, int num_pixels);

extern "C" void free_world();
