#ifndef DARKVORTEX_GEMM_H
#define DARKVORTEX_GEMM_H

// binary gemm
void gemm_bin(int M, int N, int K, float ALPHA,
	char* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

// wrapper for gemm_cpu
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float BETA,
	float* C, int ldc);

// general gemm impl
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float BETA,
	float* C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float BETA,
	float* C, int ldc);
#endif

#endif
