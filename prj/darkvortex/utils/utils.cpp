#include "utils.h"

void file_error(const char* s)
{
	fprintf(stderr, "Cannot open file: %s\n", s);
	exit(0);
}

std::string string_strip(const std::string& str)
{
	if (str.size() == 0)
		return str;

	int front_pos = 0;
	char val = str[front_pos];
	while ((val == ' ' || val == '\t' || val == '\n') && front_pos < str.size())
	{
		front_pos += 1;
		val = str[front_pos];
	}

	int back_pos = str.size() - 1;
	val = str[back_pos];
	while ((val == ' ' || val == '\t' || val == '\n') && back_pos >= 0)
	{
		back_pos -= 1;
		val = str[back_pos];
	}
	return str.substr(front_pos, back_pos - front_pos + 1);
}
