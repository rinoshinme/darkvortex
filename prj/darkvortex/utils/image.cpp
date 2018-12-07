#include "image.h"
#include <cmath>
#include <cstring>
#include <cassert>

Image::Image(int w, int h, int c, bool clear)
{
	width = w;
	height = h;
	channels = c;
	data = new float[width * height * channels];
	if (clear)
		memset(data, 0, width * height * channels * sizeof(float));
}

float Image::GetPixel(int x, int y, int c)
{
	assert(x < width && y < height && c < channels);
	return data[c * width * height + y * width + x];
}

float Image::GetPixelExtend(int x, int y, int c)
{
	if (x < 0 || x >= width || y < 0 || y >= height)
		return 0;
	if (c < 0 || c >= channels)
		return 0;
	return GetPixel(x, y, c);
}

void Image::SetPixel(int x, int y, int c, float val)
{
	if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels)
		return;
	data[c * width * height + y * width + x] = val;
}

void Image::AddPixel(int x, int y, int c, float val)
{
	if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels)
		return;
	data[c * width * height + y * width + x] += val;
}

