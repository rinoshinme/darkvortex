#include "activations.h"

Activation get_activation(const std::string& s)
{
	if (s == "linear")
		return Activation::LINEAR;
	else if (s == "logistic")
		return Activation::LOGISTIC;
	else if (s == "relu")
		return Activation::RELU;
	else if (s == "tanh")
		return Activation::TANH;
	// going with default relu activation
	return Activation::RELU;
}

std::string get_activation_string(Activation a)
{
	switch (a)
	{
	case LINEAR:
		return "linear";
	case LOGISTIC:
		return "logistic";
	case RELU:
		return "relu";
	case TANH:
		return "tanh";
	default:
		break;
	}
	return "relu";
}

float activate(float x, Activation a)
{
	switch (a)
	{
	case LINEAR:
		return linear_activate(x);
	case LOGISTIC:
		return logistic_activate(x);
	case RELU:
		return relu_activate(x);
	case TANH:
		return tanh_activate(x);
	default:
		break;
	}
	return 0;
}

float gradient(float x, Activation a)
{
	switch (a)
	{
	case LINEAR:
		return linear_gradient(x);
	case LOGISTIC:
		return logistic_gradient(x);
	case RELU:
		return relu_gradient(x);
	case TANH:
		return tanh_gradient(x);
	default:
		break;
	}
	return 0;
}

void activate_array(float* x, int n, Activation a)
{
	for (int i = 0; i < n; ++i)
		x[i] = activate(x[i], a);
}

void gradient_array(float* x, int n, Activation a, float* delta)
{
	for (int i = 0; i < n; ++i)
		delta[i] = gradient(x[i], a);
}
