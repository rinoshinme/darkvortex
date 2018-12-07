/*
 * parse network and various types of layers
 */
#ifndef DARKVORTEX_PARSER_H
#define DARKVORTEX_PARSER_H

#include "option_list.h"
#include "../network.h"
#include "../layers/activation_layer.h"

void parse_network_options(Network& net, const List<kvp>& options);

Layer* parse_activation(const List<kvp>& options);
Layer* parse_connected(const List<kvp>& options);

#endif
