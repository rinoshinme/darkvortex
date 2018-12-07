#ifndef DARKVORTEX_LAYER_H
#define DARKVORTEX_LAYER_H

#include "network.h"
#include "utils\common.h"
#include <cstdlib>

#define LAYER_DEBUG_INFO

#ifdef LAYER_DEBUG_INFO
#include <iostream>
#endif

class Layer
{
protected:
	// basic information
	LayerType type;
	Activation activation;
	CostType cost_type;

	Layer() {}
	virtual ~Layer() {}

	virtual void Forward(Network& net) = 0;
	virtual void Backward(Network& net) = 0;
	virtual void Update(UpdateArgs& args) = 0;
#if GPU
	virtual void ForwardGPU(Network& net) = 0;
	virtual void BackwardGPU(Network& net) = 0;
	virtual void UpdateGPU(UpdateArgs& args) = 0;
#endif
};

#endif
