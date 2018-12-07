#ifndef DARKVORTEX_ACTIVATIONS_H
#define DARKVORTEX_ACTIVATIONS_H

#include <string>
#include <cmath>
#include "common.h"

Activation get_activation(const std::string& s);
std::string get_activation_string(Activation a);

float activate(float x, Activation a);
float gradient(float x, Activation a);
void activate_array(float* x, int n, Activation a);
// 这里的x是activation的输出
void gradient_array(float* x, int n, Activation a, float* delta);


// activation functions
static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1.f / (1.f + exp(-x)); }
static inline float relu_activate(float x) { return x * (x > 0); }
static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }

static inline float linear_gradient(float x) { return 1; }
static inline float logistic_gradient(float x) { return (1 - x) * x; }
static inline float relu_gradient(float x) { return (x > 0); }
static inline float tanh_gradient(float x) { return 1 - x * x; }

#endif
