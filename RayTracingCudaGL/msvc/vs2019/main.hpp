#pragma once

// Implementation of CUDA simpleCUDA2GL sample - based on Cuda Samples 9.0
// Dependencies: GLFW, GLEW

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

// OpenGL
#include <GL/glew.h> // Take care: GLEW should be included before GLFW
#include <GLFW/glfw3.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "libs/helper_cuda.h"
#include "libs/helper_gl.h"

// C++ libs
#include <string>
#include <filesystem>

#include <Engine/Shader/GLSLProgram.hpp>

#include <Engine/Math/Vec3.hpp>

#include <Engine/Input/Input.hpp>

#include "gl_tools.h"
#include "glfw_tools.h"

#include "Kernel.cuh"
