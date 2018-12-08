#include "blas.h"
#include <cmath>
#include <memory>
#include <iostream>
#include <ctime>
#include <cassert>
#include "gemm.h"

void flatten(float* x, int size, int layers, int batch, bool forward)
{
	float* swap = new float[size * layers * batch];
	for (int b = 0; b < batch; ++b)
	{
		for (int c = 0; c < layers; ++c)
		{
			for (int i = 0; i < size; ++i)
			{
				int i1 = b * layers * size + c * size + i;
				int i2 = b * layers * size + i * layers + c;
				if (forward)
					swap[i2] = x[i1];
				else
					swap[i1] = x[i2];
			}
		}
	}
	memcpy(x, swap, size * layers * batch * sizeof(float));
	delete[] swap;
}

void pm(int M, int N, float* A)
{
	for (int i = 0; i < M; ++i)
	{
		printf("%d", i + 1);
		for (int j = 0; j < N; ++j)
		{
			printf("%2.4f, ", A[i * N + j]);
		}
		printf("\n");
	}
	printf("\n");
}

float* random_matrix(int rows, int cols)
{
	float* m = new float[rows * cols];
	for (int i = 0; i < rows * cols; ++i)
		m[i] = (float)rand() / RAND_MAX;
	return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
	float* a;
	if (!TA) a = random_matrix(m, k);
	else a = random_matrix(k, m);
	int lda = (!TA) ? k : m;

	float* b;
	if (!TB) b = random_matrix(k, n);
	else b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	float* c = random_matrix(m, n);
	clock_t start = clock();
	for (int i = 0; i < 10; ++i)
	{
		gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	}
	clock_t end = clock();
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
	delete[] a;
	delete[] b;
	delete[] c;
}

void reorg_cpu(float* x, int w, int h, int c, int batch, int stride, int forward, float* out)
{
	int out_c = c / (stride * stride);
	for (int b = 0; b < batch; ++b)
	{
		for (int k = 0; k < c; ++k)
		{
			for (int j = 0; j < h; ++j)
			{
				for (int i = 0; i < w; ++i)
				{
					// NCHW layout
					int in_index = i + w * (j + h * (k + c * b));
					int c2 = k % out_c;
					int offset = k / out_c;
					int w2 = i * stride + offset % stride;
					int h2 = j * stride + offset / stride;
					int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
					if (forward)
						out[out_index] = x[in_index];
					else
						out[in_index] = x[out_index];
				}
			}
		}
	}
}

void inter_cpu(int NX, float* x, int NY, float* y, int B, float* out)
{
	int index = 0;
	for (int j = 0; j < B; ++j)
	{
		for (int i = 0; i < NX; ++i)
			out[index++] = x[j * NX + i];
		for (int i = 0; i < NY; ++i)
			out[index++] = y[j * NY + i];
	}
}

void deinter_cpu(int NX, float* x, int NY, float* y, int B, float* out)
{
	int index = 0;
	for (int j = 0; j < B; ++j)
	{
		for (int i = 0; i < NX; ++i)
		{
			if (x) x[j * NX + i] = out[index];
			++index;
		}
		for (int i = 0; i < NY; ++i)
		{
			if (y) y[j * NY + i] = out[index];
			++index;
		}
	}
}

void mult_add_into_cpu(int N, float* x, float* y, float* z)
{
	for (int i = 0; i < N; ++i)
		z[i] += x[i] * y[i];
}

void const_cpu(int N, float alpha, float* x, int inc)
{
	for (int i = 0; i < N; ++i)
		x[i * inc] = alpha;
}

void pow_cpu(int N, float alpha, float* x, int incx, float* y, int incy)
{
	for (int i = 0; i < N; ++i)
		y[i * incy] = pow(x[i * incx], alpha);
}

void mul_cpu(int N, float* x, int incx, float* y, int incy)
{
	for (int i = 0; i < N; ++i)
		y[i * incy] *= x[i * incx];
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float* add,
	int w2, int h2, int c2, float s1, float s2, float* out)
{
	int stride = w1 / w2;
	int sample = w2 / w1;
	assert(stride == h1 / h2);
	assert(sample = h2 / h1);
	if (stride < 1) stride = 1;
	if (sample < 1) sample = 1;
	int minw = (w1 < w2) ? w1 : w2;
	int minh = (h1 < h2) ? h1 : h2;
	int minc = (c1 < c2) ? c1 : c2;

	for (int b = 0; b < batch; ++b)
	{
		for (int k = 0; k < minc; ++k)
		{
			for (int j = 0; j < minh; ++j)
			{
				for (int i = 0; i < minw; ++i)
				{
					int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
					int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
					out[out_index] = s1 * out[out_index] + s2 * add[add_index];
				}
			}
		}
	}
}

void mean_cpu(float* x, int batch, int filters, int spatial, float* mean)
{
	float scale = 1.0f / (batch * spatial);
	for (int i = 0; i < filters; ++i)
	{
		mean[i] = 0;
		for (int j = 0; j < batch; ++j)
		{
			for (int k = 0; k < spatial; ++k)
			{
				int index = j * filters * spatial + i * spatial + k;
				mean[i] += x[index];
			}
		}
		mean[i] *= scale;
	}
}

void variance_cpu(float* x, float* mean, int batch, int filters, int spatial, float* variance)
{
	float scale = 1.0f / (batch * spatial - 1);
	for (int i = 0; i < filters; ++i)
	{
		variance[i] = 0;
		for (j = 0; j < batch; ++j)
		{
			for (int k = 0; k < spatial; ++k)
			{
				int index = j * filters * spatial + i * spatial + k;
				variance[i] = pow(x[index] - mean[i], 2);
			}
		}
		variance[i] *= scale;
	}
}

void scale_bias(float* output, float* scales, int batch, int n, int size)
{
	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < size; ++j)
				output[(b * n + i) * size + j] *= scales[i];
		}
	}
}

void backward_scale_cpu(float* x_norm, float* delta, int batch, int n, int size, float* scale_updates)
{
	for (int f = 0; f < n; ++f)
	{
		float sum = 0;
		for (int b = 0; b < batch; ++b)
		{
			for (int i = 0; i < size; ++i)
			{
				int index = i + size * (f + n * b);
				sum += delta[index] * x_norm[index];
			}
		}
		scale_updates[f] += sum;
	}
}

void mean_delta_cpu(float* delta, float* variance, int batch, int filters, int spatial, float* mean_delta)
{
	for (int i = 0; i < filters; ++i)
	{
		mean_delta[i] = 0;
		for (int j = 0; j < batch; ++j)
		{
			for (int k = 0; k < spatial; ++k)
			{
				int index = j * filters * spatial + i * spatial + k;
				mean_delta[i] += delta[index];
			}
		}
		mean_delta[i] *= (-1.0f / sqrt(variance[i] + 0.00001f));
	}
}

void variance_delta_cpu(float* x, float* delta, float* mean, float* variance, int batch, int filters, int spatial, float* variance_delta)
{
	for (int i = 0; i < filters; ++i)
	{
		variance_delta[i] = 0;
		for (int j = 0; j < batch; ++j)
		{
			for (int k = 0; k < spatial; ++k)
			{
				int index = j * filters * spatial + i * spatial + k;
				variance_delta[i] += delta[index] * (x[index] - mean[i]);
			}
		}
		variance_delta[i] *= -.5f * pow(variance[i] + 0.00001f, (float)(-3. / 2.));
	}
}

void normalize_delta_cpu(float* x, float* mean, float* variance, float* mean_delta, float* variance_delta, int batch, int filters, int spatial, float* delta)
{
	for (int j = 0; j < batch; ++j){
		for (int f = 0; f < filters; ++f){
			for (int k = 0; k < spatial; ++k){
				int index = j*filters*spatial + f*spatial + k;
				delta[index] = delta[index] * 1. / (sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f] / (spatial*batch);
			}
		}
	}
}

void l2normalize_cpu(float* x, float* dx, int batch, int filters, int spatial)
{
	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < spatial; ++i)
		{
			float sum = 0;
			for (int f = 0; f < filters; ++f)
			{
				int index = b * filters * spatial + f * spatial + i;
				sum += powf(x[index], 2);
			}
			sum = sqrtf(sum);
			for (int f = 0; f < filters; ++f)
			{
				int index = b * filters * spatial + f * spatial + i;
				x[index] /= sum;
				dx[index] = (1 - x[index]) / sum;
			}
		}
	}
}
