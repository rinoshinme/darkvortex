#ifndef DARKVORTEX_NETWORK_H
#define DARKVORTEX_NETWORK_H

#include <vector>
#include "utils\common.h"

class Layer;
class Network
{
public:
	int n;
	int batch;
	size_t* seen;
	int* t;
	float epoch;
	int subdivision;
	std::vector< Layer* > layers;
	float* output;

	LearningRatePolicy policy;
	float learning_rate;
	float momentum;
	float decay;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;
	float* scales;
	int* steps;
	int num_steps;
	int burn_in;

	int adam;
	float B1;
	float B2;
	float eps;

	int inputs;
	int outputs;
	int truths;
	int notruths;
	int h, w, c;

	int max_crop;
	int min_crop;
	float max_ratio;
	float min_ratio;
	int center;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;
	int random;

	int gpu_index;
	// tree* hierarchy;
	
	float* input;
	float* truth;
	float* delta;
	float* workspace;
	int train;
	int index;
	float* cost;
	float clip;

#ifdef GPU
	float* input_gpu;
	float* truth_gpu;
	float* delta_gpu;
	float* output_gpu;
#endif
};

#endif
