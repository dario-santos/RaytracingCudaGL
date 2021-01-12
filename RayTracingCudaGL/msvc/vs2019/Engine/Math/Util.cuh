#pragma once

/**
* Clamp
* 
* \param x The value to clamp
* \param a The minimum value that x can have
* \param b the maximum value that x can have
* \return The value of x clamped to the range [a, b]
*/
__device__ float Clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

/**
 * RgbToInt
 * 
 * \param r The R component of the RGB color
 * \param g The G component of the RGB color
 * \param b The B component of the RGB color
 * \return The hexadecimal representation of a RGB color
 */
__device__ int RgbToInt(float r, float g, float b)
{
	r = Clamp(r, 0.0f, 255.0f);
	g = Clamp(g, 0.0f, 255.0f);
	b = Clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}
