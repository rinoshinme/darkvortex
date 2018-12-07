#ifndef DARKVORTEX_SOFTMAX_LAYER_H
#define DARKVORTEX_SOFTMAX_LAYER_H

#include "layer.h"

class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer();
	void Create(float* input, int n, float temp, float* output);
	void Forward(Network& network);
	void Backward(Network& network);
	void Update(Network& network) {}
};

#endif
