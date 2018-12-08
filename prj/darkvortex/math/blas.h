/*
 * basic algebraic operations for cpu
 */

#ifndef DARKVORTEX_BLAS_H
#define DARKVORTEX_BLAS_H

// flatten only used in reorg layer
void flatten(float* x, int size, int layers, int batch, bool forward);
// print matrix with M rows and N cols
void pm(int M, int N, float* A);
// generate random matrix in [0, 1]
float* random_matrix(int rows, int cols);
// gemm timing
void time_random_matrix(int TA, int TB, int m, int k, int n);
// reorgnize from [c h w] -> [c/ss hs ws] if forward == TRUE
// or [c/ss hs ws] -> [c h w] if forward == FALSE
void reorg_cpu(float* x, int w, int h, int c, int batch, int stride, int forward, float* out);
/*
* X: BxNX, Y: BxNY, out: Bx(NX+NY)
* row interleaved[1 row from x, 1 row from y...]
*/
void inter_cpu(int NX, float* x, int NY, float* y, int B, float* out);
// split interleaved out into x and y
void deinter_cpu(int NX, float* x, int NY, float* y, int B, float* out);
// z += x * y
void mult_add_into_cpu(int N, float* x, float* y, float* z);
// set x be value of alpha, for every inc step
void const_cpu(int N, float alpha, float* x, int inc);
// y = pow(x, alpha)
void pow_cpu(int N, float alpha, float* x, int incx, float* y, int incy);
// y *= x
void mul_cpu(int N, float* x, int incx, float* y, int incy);
// out = s1 * out + s2 * add (with reshaping)
void shortcut_cpu(int batch, int w1, int h1, int c1, float* add,
	int w2, int h2, int c2, float s1, float s2, float* out);
// get mean foreach filter [batch, w, h as other dims are squeezed]
// useful for batch norm
void mean_cpu(float* x, int batch, int filters, int spatial, float* mean);
// get variance for each filter, useful for batch norm
void variance_cpu(float* x, float* mean, int batch, int filters, int spatial, float* variance);
// scale output along each channel (1 multiplier for each channel)
void scale_bias(float* output, float* scales, int batch, int n, int size);
// for batch norm
void backward_scale_cpu(float* x_norm, float* delta, int batch, int n, int size, float* scale_updates);
void mean_delta_cpu(float* delta, float* variance, int batch, int filters, int spatial, float* mean_delta);
void variance_delta_cpu(float* x, float* delta, float* mean, float* variance, int batch, int filters, int spatial, float* variance_delta);
void normalize_delta_cpu(float* x, float* mean, float* variance, float* mean_delta, float* variance_delta, int batch, int filters, int spatial, float* delta);

void l2normalize_cpu(float* x, float* dx, int batch, int filters, int spatial);

void axpy_cpu(int N, float alpha, float* x, int incx, float* y, int incy);
void scale_cpu(int N, float alpha, float* x, int incx);
inline void fill_cpu(int N, float alpha, float* x, int incx) {}
inline void copy_cpu(int N, float* x, int incx, float* y, int incy) {}

#endif
