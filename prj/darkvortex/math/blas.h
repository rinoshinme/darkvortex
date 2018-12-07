/*
 * basic algebraic operations for cpu
 */

#ifndef DARKVORTEX_BLAS_H
#define DARKVORTEX_BLAS_H

void flatten(float* x, int size, int layers, int batch, bool forward);


void const_cpu(int N, float alpha, float* x, int inc);
void mul_cpu(int N, float* x, int incx, float* y, int incy);
void power_cpu(int N, float alpha, float* x, int incx, float* y, int incy);
void axpy_cpu(int N, float alpha, float* x, int incx, float* y, int incy);
void scale_cpu(int N, float alpha, float* x, int incx);
inline void fill_cpu(int N, float alpha, float* x, int incx) {}
inline void copy_cpu(int N, float* x, int incx, float* y, int incy) {}

#endif
