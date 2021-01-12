#pragma once

#include <vector_types.h>

/**
 * launch_cudaRender
 *
 * \param blocks The number of blocks
 * \param threads The number of threads
 * \param g_odata The output color vector
 * \param max_x The Width of the array
 * \param max_y The Heigth of the array
 */
extern "C" void launch_cudaRender(dim3 blocks, dim3 threads, unsigned int* g_odata, int max_x, int max_y);

/**
 * create_world
 *
 * \param max_x The Width of the array
 * \param max_y The Heigth of the array
 * \param num_pixels The number of pixels in the buffer
 */
extern "C" void create_world(int max_x, int max_y, int num_pixels);
