#include "common.h"

std::string getLayerString(LayerType& a)
{
	switch (a)
	{
	case CONVOLUTIONAL:
		return "convolutional";
	case ACTIVATE:
		return "activation";
	default:
		break;
	}
	return "none";
}
