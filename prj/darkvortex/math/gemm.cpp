#include "gemm.h"

void gemm_bin(int M, int N, int K, float ALPHA,
	char* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			char A_part = A[i * lda + k];
			if (A_part)
			{
				for (int j = 0; j < N; ++j)
					C[i * ldc + j] += B[k * ldb + j];
			}
			else
			{
				for (int j = 0; j < N; ++j)
					C[i * ldc + j] -= B[k * ldb + j];
			}
		}
	}
}

/* A: MxK, B: KxN, C: MxN 
 * C += alpha * A * B
 */
void gemm_nn(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
// #pragma omp parallel for // openMP
	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			register float A_part = ALPHA * A[i * lda + k];
			for (int j = 0; j < N; ++j)
				C[i * ldc + j] += A_part * B[k * ldb + j];
		}
	}
}

/* A: MxK, B: NxK, C: MxN
 * C += alpha * A * BT
 */
void gemm_nt(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			register float sum = 0;
			for (int k = 0; k < K; ++k)
				sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
			C[i * ldc + j] += sum;
		}
	}
}

/*
 * A: KxM, B: KxN, C: MxN
 * C += alpha * AT * B
 */
void gemm_tn(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			register float A_part = ALPHA * A[k * lda + i];
			for (int j = 0; j < N; ++j)
				C[i * ldc + j] += A_part * B[k * ldb + j];
		}
	}
}

/*
 * A: KxM, B: NxK, C: MxN
 * C += alpha * AT * BT
 */
void gemm_tt(int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc)
{
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			register float sum = 0;
			for (int k = 0; k < K; ++k)
				sum += ALPHA * A[k * lda + i] * B[j * ldb + k];
			C[i * ldc + j] += sum;
		}
	}
}

/*
 * C += ALPHA * A * B + BETA
 */
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float BETA,
	float* C, int ldc)
{
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
			C[i * ldc + j] += BETA;
	}
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB)
		gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB)
		gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else
		gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float* A, int lda,
	float* B, int ldb,
	float BETA,
	float* C, int ldc)
{
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}
