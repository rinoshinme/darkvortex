#ifndef DARKVORTEX_IMAGE_H
#define DARKVORTEX_IMAGE_H

#include <string>
#include "image_helper.h"
#include "common.h"

// global functions

class Image
{
public:
	int width;
	int height;
	int channels;
	float* data;

public:
	Image(int w, int h, int c, bool clear = true);

	// functions are divided into 2 categories:
	// 1. generate a new image
	// 2. modify the original image
	void DrawBox(int x1, int y1, int x2, int y2, ColorRGB& color);
	void DrawBBox(Box& box, int w, ColorRGB& color);
	// void WriteLabel(int r, int c);
	void Scale(float s);
	Image RotateCrop(float rad, float s, int w, int h, float dx, float dy, float aspect);
	Image RandomCrop(int w, int h);
	Image RandomAugment(float angle, float aspect, int low, int high, int w, int h);
	AugmentArgs RandomAugmentArgs(float angle, float aspect, int low, int high, int w, int h);
	void LetterboxImageInto();
	Image ResizeMax(int max);
	void Translate(float s);
	void Embed(Image& dest, int dx, int dy);
	void Place(int w, int h, int dx, int dy, Image& canvas);
	void Saturate(float sat);
	void Exposure(float sat);
	void Distort(float hue, float sat, float val);
	void SaturateExposure(float sat, float exposure);
	void RgbToHsv();
	void HsvToRgb();
	void YuvToRgb();
	void RgbToYuv();
	
private:
	float GetPixel(int x, int y, int c);
	float GetPixelExtend(int x, int y, int c);
	void SetPixel(int x, int y, int c, float val);
	void AddPixel(int x, int y, int c, float val);
	float BilinearInterpolate(float x, float y, int c);
};

Image ImageDistance(const Image& a, const Image& b);

#ifdef OPENCV
#include "opencv2\opencv.hpp"

void* openVideoStream(const std::string& f, int c, int w, int h, int fps);
Image getImageFromStream(void* p);
Image loadImageCV(const std::string& filename, int channels);
int showImageCV(const Image& im, const std::string& name, int ms);

#endif

#endif
