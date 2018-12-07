/*
 * config file io
 * 3 types of configuration files:
 *		1. model config
 *		2. data config
 *		3. metadata [names of all classes]
 */
#ifndef DARKVORTEX_OPTION_LIST_H
#define DARKVORTEX_OPTION_LIST_H

#include <string>
#include "list.h"

struct kvp
{
	std::string key;
	std::string val;
	int used;
};

struct section
{
	std::string val;
	List<kvp> options;
};

// data configuration
bool read_option(const std::string& line, kvp& kv);

List<kvp> read_data_cfg(const std::string& filename);

bool option_find(const List<kvp>& l, const std::string& key, std::string& val);
std::string option_find_str(const List<kvp>& l, const std::string& key, const std::string& default_val = "");

int option_find_int(const List<kvp>& l, const std::string& key, int default_val);
int option_find_int_quiet(const List<kvp>& l, const std::string& key, int default_val);
float option_find_float(const List<kvp>& l, const std::string& key, float default_val);
float option_find_float_quiet(const List<kvp>& l, const std::string& key, float default_val);

// model configuration
List<section> read_model_cfg(const std::string& filename);

#endif
