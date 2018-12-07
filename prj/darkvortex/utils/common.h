/*
 * Common data structure definitions
 */
#ifndef DARKVORTEX_COMMON_H
#define DARKVORTEX_COMMON_H

#include <string>
#include <vector>
#include "data.h"

class Image;

enum LearningRatePolicy
{
	CONSTANT,
	STEP,
	EXP,
	POLY,
	STEPS,
	SIG,
	RANDOM
};

enum LayerType
{
	CONVOLUTIONAL,
	DECONVOLUTIONAL,
	CONNECTED,
	MAXPOOL,
	SOFTMAX,
	DROPOUT,
	ACTIVATE,
};

std::string getLayerString(LayerType& a);

enum Activation
{
	LINEAR,
	LOGISTIC,
	RELU,
	TANH,
	// LEAKY,
	// SELU,
};

enum CostType
{
	SSE,
	MASKED,
	L1,
	SEG,
	SMOOTH,
	WGAN,
};

struct LoadArgs
{
	int threads;
	char** paths;
	char* path;
	int n;
	int m;
	char** labels;
	int h;
	int w;
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min, max, size;
	int classes;
	int background;
	int scale;
	int center;
	int coords;
	float jitter;
	float angle;
	float aspect;
	float saturation;
	float exposure;
	float hue;
	Data* d;
	Image* im;
	Image* resized;
};

struct UpdateArgs
{
	int batch;
	float learning_rate;
	float momentum;
	float decay;
	int adam;
	float B1;
	float B2;
	float eps;
	int t;
};

struct ImageAugmentParam
{
	int center;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;
};

struct AugmentArgs
{
	int w;
	int h;
	float scale;
	float rad;
	float dx;
	float dy;
	float aspect;
};

struct AdamArgs
{
	float* m;
	float* v;
	float* bias_m;
	float* scale_m;
	float* bias_v;
	float* scale_v;

	AdamArgs(int inputs, int outputs)
	{
		m = new float[inputs * outputs];
		memset(m, 0, inputs * outputs * sizeof(float));
		v = new float[inputs * outputs];
		memset(v, 0, inputs * outputs * sizeof(float));
		bias_m = new float[outputs];
		memset(bias_m, 0, outputs * sizeof(float));
		scale_m = new float[outputs];
		memset(scale_m, 0, outputs * sizeof(float));
		bias_v = new float[outputs];
		memset(bias_v, 0, outputs * sizeof(float));
		scale_v = new float[outputs];
		memset(scale_v, 0, outputs * sizeof(float));
	}

	~AdamArgs()
	{
		if (m) delete[] m;
		if (v) delete[] v;
		if (bias_m) delete[] bias_m;
		if (bias_v) delete[] bias_v;
		if (scale_m) delete[] scale_m;
		if (scale_v) delete[] scale_v;
	}
};

struct BatchNormArgs
{
	float* scales;
	float* scale_updates;
	float* mean;
	float* mean_delta;
	float* variance;
	float* variance_delta;
	float* rolling_mean;
	float* rolling_variance;
	float* x;
	float* x_norm;

	BatchNormArgs(int inputs, int outputs, int batch)
	{
		scales = new float[outputs];
		for (int i = 0; i < outputs; ++i)
			scales[i] = 1;
		scale_updates = new float[outputs];
		memset(scale_updates, 0, outputs * sizeof(float));

		mean = new float[outputs];
		memset(mean, 0, outputs * sizeof(float));
		mean_delta = new float[outputs];
		memset(mean_delta, 0, outputs * sizeof(float));
		variance = new float[outputs];
		memset(variance, 0, outputs * sizeof(float));
		variance_delta = new float[outputs];
		memset(variance_delta, 0, outputs * sizeof(float));

		rolling_mean = new float[outputs];
		memset(rolling_mean, 0, outputs * sizeof(float));
		rolling_variance = new float[outputs];
		memset(rolling_variance, 0, outputs * sizeof(float));

		x = new float[batch * outputs];
		memset(x, 0, outputs * sizeof(float));
		x_norm = new float[batch * outputs];
		memset(x_norm, 0, outputs * sizeof(float));
	}

	~BatchNormArgs()
	{
		if (scales) delete[] scales;
		if (scale_updates) delete[] scale_updates;
		if (mean) delete[] mean;
		if (variance) delete[] variance;
		if (mean_delta) delete[] mean_delta;
		if (variance_delta) delete[] variance_delta;
		if (rolling_mean) delete[] rolling_mean;
		if (rolling_variance) delete[] rolling_variance;
		if (x) delete[] x;
		if (x_norm) delete[] x_norm;
	}
};

struct XNorArgs
{

};

template<typename T>
inline T* CreateEmptyMemory(int size)
{
	T* memory = new T[size];
	memset(memory, 0, size * sizeof(T));
	return memory;
}

#if GPU

struct TrainArgs
{
	Network* net;
	Data d;
	float* err;
};

#endif


#endif
