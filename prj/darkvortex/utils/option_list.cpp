#include "option_list.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "utils.h"

bool read_option(const std::string& line, kvp& kv)
{
	int pos = line.find('=');
	if (pos == std::string::npos)
		return false;
	kv.key = string_strip(line.substr(0, pos));
	kv.val = string_strip(line.substr(pos + 1));
	return true;
}

List<kvp> read_data_cfg(const std::string& filename)
{
	std::ifstream stream;
	stream.open(filename);
	if (!stream.is_open())
		file_error(filename.c_str());
	
	std::string line;
	int nu = 0;
	List<kvp> list;

	std::getline(stream, line);
	while (!line.empty())
	{
		++nu;
		// strip(line)
		switch (line[0])
		{
		case '\0':
		case '#':
		case ';':
			break;
		default:
			// read option on this line
			kvp kv;
			read_option(line, kv);
			list.insert(kv);
			break;
		}
		std::getline(stream, line);
	}
	stream.close();

	return list;
}

bool option_find(const List<kvp>& l, const std::string& key, std::string& val)
{
	Node<kvp>* n = l.front_node();
	while (n)
	{
		if (key == n->val.key)
		{
			n->val.used = 1;
			val = n->val.val;
			return true;
		}
		n = n->next;
	}
	return false;
}

std::string option_find_str(const List<kvp>& l, const std::string& key, const std::string& default_val)
{
	std::string val;
	bool ret = option_find(l, key, val);
	if (ret)
		return val;
	else
	{
		fprintf(stderr, "%s: Using default '%s'\n", key, default_val);
		return default_val;
	}
}
int option_find_int(const List<kvp>& l, const std::string& key, int default_val)
{
	std::string val;
	bool ret = option_find(l, key, val);
	if (ret)
		return atoi(val.c_str());
	else
	{
		fprintf(stderr, "%s: Using default '%s'\n", key, default_val);
		return default_val;
	}
}

int option_find_int_quiet(const List<kvp>& l, const std::string& key, int default_val)
{
	std::string val;
	bool ret = option_find(l, key, val);
	if (ret)
		return atoi(val.c_str());
	else
		return default_val;
}

float option_find_float(const List<kvp>& l, const std::string& key, float default_val)
{
	std::string val;
	bool ret = option_find(l, key, val);
	if (ret)
		return atof(val.c_str());
	else
	{
		fprintf(stderr, "%s: Using default '%s'\n", key, default_val);
		return default_val;
	}
}

float option_find_float_quiet(const List<kvp>& l, const std::string& key, float default_val)
{
	std::string val;
	bool ret = option_find(l, key, val);
	if (ret)
		return atof(val.c_str());
	else
		return default_val;
}

List<section> read_model_cfg(const std::string& filename)
{
	std::ifstream stream;
	stream.open(filename);
	if (!stream.is_open())
		file_error(filename.c_str());

	std::string line;
	int nu = 0;
	List<section> sections;
	section current;
	
	std::getline(stream, line);
	while (!line.empty())
	{
		++nu;
		string_strip(line);
		switch (line[0])
		{
		case '[':
			// new section
			if (!current.val.empty())
				sections.insert(current);
			current.val = "";
			current.options = List<kvp>();
			break;
		case '\0':
		case '#':
		case ';':
			break;
		default:
			kvp kv;
			read_option(line, kv);
			current.options.insert(kv);
		}
	}
	if (!current.val.empty())
		sections.insert(current);
	return sections;
}
