#ifndef DARKVORTEX_ACTIVATION_LAYER_H
#define DARKVORTEX_ACTIVATION_LAYER_H

#include "layer.h"

// separate input and output data memory
class ActivationLayer : public Layer
{
public:
	ActivationLayer(int batch, int inputs, Activation activation);

	virtual void Forward(Network& network);
	virtual void Backward(Network& network);
	virtual void Update(UpdateArgs& args) { /* Do nothing */ }

private:
	int inputs;
	int outputs;
	int batch;

	float* output;
	float* delta;
};

#endif
