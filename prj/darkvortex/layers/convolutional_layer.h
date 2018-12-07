#ifndef DARKVORTEX_CONVOLUTIONAL_LAYER_H
#define DARKVORTEX_CONVOLUTIONAL_LAYER_H

#include "layer.h"

class ConvolutionalLayer : public Layer
{
public:
	ConvolutionalLayer();
	void Create();
	void Forward(Network& network);
	void Backward(Network& network);
	void Update(Network& network);
};

#endif
