#ifndef INPAINTOR_H_
#define INPAINTOR_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/fuzzy.hpp>

namespace place_recognition
{
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
		cv::rectangle(mask_img,cv::Point(150,0),cv::Point(210,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);
		cv::rectangle(mask_img,cv::Point(420,0),cv::Point(490,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);
		cv::rectangle(mask_img,cv::Point(800,0),cv::Point(840,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);
		cv::rectangle(mask_img,cv::Point(1060,0),cv::Point(1120,input_img.rows),cv::Scalar(255,255,255),-1,cv::LINE_AA);	
		cv::cvtColor(mask_img,mask_img,cv::COLOR_BGR2GRAY);

		// inpaint img
		cv::Mat inpaint_img;
		cv::inpaint(input_img,mask_img,inpaint_img,3,cv::INPAINT_TELEA);
		output_img = inpaint_img;		
	}

	void resize_img(cv::Mat& img)
	{
		// int col_size = img.cols;
		// int row_size = img.rows;
		cv::Rect rect(cv::Point(0,265),cv::Size(img.cols,235));
		img = img(rect);
	}

private:
};
}

#endif	// INPAINTOR_H_