#include "random.h"
#include <cstdlib>

float rand_uniform(float min, float max)
{
	if (max < min)
	{
		float swap = min;
		min = max;
		max = swap;
	}
	return ((float)rand() / RAND_MAX * (max - min)) + min;
}
