/*
 * input, output related data structures.
 * Image structure has its own implementation
 */
#ifndef DARKVORTEX_DATA_H
#define DARKVORTEX_DATA_H

struct Box
{
	float x, y, w, h;
};

struct Matrix
{
	int rows;
	int cols;
	std::vector<std::vector<float> > vals;
};

struct Data
{
	int w;
	int h;
	Matrix X;
	Matrix y;
	int shallow;
	// int* num_boxes;
	// box** boxes;
};

struct Detection
{
	Box bbox;
	int classes;
	std::vector<float> prob;
	std::vector<float> mask;
	float objectness;
	int sort_class;
};

#endif
