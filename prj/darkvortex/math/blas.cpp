#include "blas.h"
#include <cmath>

void const_cpu(int N, float alpha, float* x, int inc)
{
	for (int i = 0; i < N; ++i)
		x[i * inc] = alpha;
}

void mul_cpu(int N, float* x, int incx, float* y, int incy)
{
	for (int i = 0; i < N; ++i)
		y[i * incy] *= x[i * incx];
}

void power_cpu(int N, float alpha, float* x, int incx, float* y, int incy)
{
	for (int i = 0; i < N; ++i)
		y[i * incy] = pow(x[i * incx], alpha);
}
