#pragma once

#include "cuda_runtime.h"

#include <stdlib.h>
#include <stdio.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/**
 * check_cuda
 *
 * \param result The result of a cuda call
 * \param func The function that generated the error
 * \param file The file that generated the error
 * \param line The line that generated the error
 */
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) 
{
  if (result) 
  {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";

    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}
