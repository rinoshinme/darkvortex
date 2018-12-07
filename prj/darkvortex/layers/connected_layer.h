#ifndef DARKVORTEX_CONNECTED_LAYER_H
#define DARKVORTEX_CONNECTED_LAYER_H

#include "layer.h"
#include "../utils/common.h"

class ConnectedLayer : public Layer
{
public:
	ConnectedLayer(int batch, int inputs, int outputs, Activation activation, bool batch_normalize, bool adam);

	void Forward(Network& network);
	void Backward(Network& network);
	void Update(UpdateArgs& args);

private:
	void InitWeightAndBias();

private:
	float learning_rate_scale;
	int inputs;
	int outputs;
	int batch;
	bool batch_normalize;
	int h, w, c;
	int out_h, out_w, out_c;

	float* output;
	float* delta;
	float* weight_updates;
	float* bias_updates;
	float* weights;
	float* biases;

	AdamArgs* adam_args;
	BatchNormArgs* bn_args;
};

#endif
