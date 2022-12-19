#ifndef INPAINTOR_H_
#define INPAINTOR_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/fuzzy.hpp>

namespace place_recognition
{
class Pillar
{
public:
	Pillar() :
		start(0), end(0) {}

	Pillar(int _start,int _end) :
		start(_start), end(_end) {}
	
	int start;
	int end;
private:
};

class Pillars
{
public:
	Pillars() {}

	Pillar first;
	Pillar second;
	Pillar third;
	Pillar fourth;

private:
};

class Inpaintor
{
public:
	Inpaintor() { }

	void inpaint_img(cv::Mat input_img,cv::Mat& output_img)
	{
		// resize
		resize_img(input_img);

		// mask img
		cv::Mat mask_img = cv::Mat::zeros(input_img.size(),CV_8UC3);
		cv::rectangle(mask_img,cv::Point(pillars_.first.start,0),cv::Point(pillars_.first.end,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);
		cv::rectangle(mask_img,cv::Point(pillars_.second.start,0),cv::Point(pillars_.second.end,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);
		cv::rectangle(mask_img,cv::Point(pillars_.third.start,0),cv::Point(pillars_.third.end,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);
		cv::rectangle(mask_img,cv::Point(pillars_.fourth.start,0),cv::Point(pillars_.fourth.end,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);	
		cv::cvtColor(mask_img,mask_img,cv::COLOR_BGR2GRAY);

		// inpaint img
		cv::Mat inpaint_img;
		cv::inpaint(input_img,mask_img,inpaint_img,3,cv::INPAINT_TELEA);
		output_img = inpaint_img;		
	}

	void resize_img(cv::Mat& img)
	{
		cv::Rect rect(cv::Point(0,265),cv::Size(img.cols,235));
		img = img(rect);
	}

	void set_params(Pillars pillars) { pillars_ = pillars; }

private:
	Pillars pillars_;
};
}

#endif	// INPAINTOR_H_