#include "connected_layer.h"
#include "../math/blas.h"

ConnectedLayer::ConnectedLayer(int batch, int inputs, int outputs, Activation activation, bool batch_normalize, bool adam)
{
	this->activation = activation;
	this->type = CONNECTED;

	this->learning_rate_scale = 1;
	
	this->inputs = inputs;
	this->outputs = outputs;
	this->batch = batch;
	this->batch_normalize = batch_normalize;

	this->h = 1;
	this->w = 1;
	this->c = inputs;
	this->out_w = 1;
	this->out_h = 1;
	this->out_c = outputs;

	this->output = new float[this->batch * this->outputs];
	memset(this->output, 0, this->batch * this->outputs * sizeof(float));
	this->delta = new float[this->batch * this->outputs];
	memset(this->delta, 0, this->batch * this->outputs * sizeof(float));

	this->weights = new float[this->outputs * this->inputs];
	memset(this->weights, 0, this->outputs * this->inputs * sizeof(float));
	this->biases = new float[this->outputs];
	memset(this->biases, 0, this->outputs * sizeof(float));

	this->weight_updates = new float[this->outputs * this->inputs];
	memset(this->weight_updates, 0, this->outputs * this->inputs * sizeof(float));
	this->bias_updates = new float[this->outputs];
	memset(this->bias_updates, 0, this->outputs * sizeof(float));

	InitWeightAndBias();

	if (adam)
		this->adam_args = new AdamArgs(this->inputs, this->outputs);
	else
		this->adam_args = NULL;

	if (batch_normalize)
		this->bn_args = new BatchNormArgs(this->inputs, this->outputs, this->batch);
	else
		this->bn_args = NULL;
}

void ConnectedLayer::Update(UpdateArgs& args)
{
	float learning_rate = args.learning_rate * this->learning_rate_scale;
	float momentum = args.momentum;
	float decay = args.decay;
	int batch = args.batch;
	axpy_cpu(this->outputs, learning_rate / batch, this->bias_updates, 1, this->biases, 1);
	scale_cpu(this->outputs, momentum, this->bias_updates, 1);

	if (this->batch_normalize)
	{
		axpy_cpu(this->outputs, learning_rate / batch, this->bn_args->scale_updates, 1, this->bn_args->scales, 1);
		scale_cpu(this->outputs, momentum, this->bn_args->scale_updates, 1);
	}

	axpy_cpu(this->inputs * this->outputs, -decay / batch, this->weights, 1, this->weight_updates, 1);
	axpy_cpu(this->inputs * this->outputs, learning_rate / batch, this->weight_updates, 1, this->weights, 1);
	scale_cpu(this->inputs * this->outputs, momentum, this->weight_updates, 1);
}

void ConnectedLayer::Forward(Network& network)
{
	fill_cpu(this->outputs * this->batch, 0, this->output, 1);
	int m = this->batch;
	int k = this->inputs;
	int n = this->outputs;
	float* a = network.input;
	float* b = this->weights;
	float* c = this->output;
	/*
	gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

	if (this->batch_normalize)
		forward_batchnorm_layer(network);
	else
		add_bias(this->output, this->biases, this->batch, this->outputs, 1);
	
	activate_array(this->output, this->outputs * this->batch, this->activation);
	*/
}

void ConnectedLayer::Backward(Network& network)
{

}

