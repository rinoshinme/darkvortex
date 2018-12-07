#include "activation_layer.h"
#include "../math/blas.h"
#include "../utils/activations.h"

ActivationLayer::ActivationLayer(int batch, int inputs, Activation activation)
{
	this->inputs = inputs;
	this->outputs = inputs;
	this->batch = batch;
	this->activation = activation;

	this->output = new float[this->inputs];
	memset(this->output, 0, this->inputs * sizeof(float));
	this->delta = new float[this->inputs];
	memset(this->delta, 0, this->inputs * sizeof(float));

#ifdef LAYER_DEBUG_INFO
	std::cerr << "Activation Layer: " << inputs << " inputs\n";
#endif
}

void ActivationLayer::Forward(Network& net)
{
	copy_cpu(this->outputs * this->batch, net.input, 1, this->output, 1);
	activate_array(this->output, this->outputs * this->batch, this->activation);
}

void ActivationLayer::Backward(Network& net)
{
	gradient_array(this->output, this->outputs * this->batch, this->activation, this->delta);
	copy_cpu(this->outputs * this->batch, this->delta, 1, net.delta, 1);
}

#ifdef GPU
void ActivationLayer::ForwardGPU(Network& net)
{
}

void ActivationLayer::BackwardGPU(Network& net)
{
}

#endif
