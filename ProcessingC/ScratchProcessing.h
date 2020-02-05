#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
//#define DEBUG
void DrawRotatedRectangle(Mat& image, RotatedRect rotatedRectangle, Scalar color)
{
	// Create the rotated rectangle
	// We take the edges that OpenCV calculated for us
	Point2f vertices2f[4];
	rotatedRectangle.points(vertices2f);

	// Convert them so we can use them in a fillConvexPoly
	for (int i = 0; i < 4; i++) {
		line(image, vertices2f[i], vertices2f[(i + 1) % 4], color, 1, 8);
	}
}
Mat sobel_detector(Mat src)
{
	if (!src.data)
	{
		return Mat();
	}

	/// Reduce noise with a kernel 3x3
	Mat detected_edges;
	static int count = 0;
#ifdef DEBUG
	String path = "C:/Users/nvnhu/Downloads/ThorLabsSampleCaptured/Test/Result/";
	imwrite(path + to_string(count++) + "src.png", src);
#endif
	/// Convert the image to grayscale
	Mat gray;
	cvtColor(src, gray, CV_RGB2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 3, 0, BORDER_DEFAULT);
	medianBlur(gray, gray, 5);
	//sobel
	/// Generate grad_x and grad_y
	int scale = 1;
	int delta = 0;
	int ddepth = CV_64F;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Sobel(gray, grad_x, ddepth, 1, 0, 5, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	Sobel(gray, grad_y, ddepth, 0, 1, 5, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	Mat sobel;
	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);
	GaussianBlur(sobel, sobel, Size(5, 5), 0, 0, BORDER_DEFAULT);
	medianBlur(sobel, sobel, 5);
	return sobel;
}
bool detectScratch(vector<Mat> vtImage) {
#ifdef DEBUG
	String path = "C:/Users/nvnhu/Downloads/ThorLabsSampleCaptured/Test/Result/";
#endif
	for (int i = 0; i < vtImage.size(); i++) {
		Mat img_src = vtImage.at(i);
		Mat img_gray, img_sobel;
		Mat img_thresold, img_thresold_black_defect, img_dilate, img_erode, img_median;
		cvtColor(img_src, img_gray, CV_RGB2GRAY);

		// detect sobel

		// increase edge defect 
		Mat img_sharpen;
		cv::GaussianBlur(img_src, img_sharpen, cv::Size(0, 0), 5);
		cv::addWeighted(img_src, 1.5, img_sharpen, -0.5, 0, img_sharpen);
#ifdef DEBUG
		imwrite(path + to_string(i) + "img_sharpen.png", img_sharpen);
#endif // DEBUG

		
		// end 

		img_sobel = sobel_detector(img_sharpen);
		// thresold sobel
		threshold(img_sobel, img_thresold, 50, 255, CV_THRESH_BINARY);
		Dilate(img_thresold, img_dilate, 9);
		Erode(img_dilate, img_erode, 9);
#ifdef DEBUG
		imwrite(path + to_string(i) + "img_sobel.png", img_sobel);
		imwrite(path + to_string(i) + "img_thresold.png", img_thresold);
#endif // DEBUG
		// thresold src remove black defect and outline
		GaussianBlur(img_gray, img_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
		threshold(img_gray, img_thresold_black_defect, 100, 255, CV_THRESH_BINARY_INV);
		Erode(img_thresold_black_defect, img_thresold_black_defect, 15);
		Dilate(img_thresold_black_defect, img_thresold_black_defect, 35);
		bitwise_not(img_thresold_black_defect, img_thresold_black_defect);
		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;

		// and to remove outline
		Mat img_and;
		bitwise_and(img_thresold_black_defect, img_erode, img_and);
#ifdef DEBUG
		imwrite(path + to_string(i) + "img_and.png", img_and);
		imwrite(path + to_string(i) + "img_thresold_black_defect.png", img_thresold_black_defect);
		imwrite(path + to_string(i) + "img_erode.png", img_erode);
#endif // DEBUG
		
		contours.clear();
		hierarchy.clear();
		Mat img_result = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);
		Dilate(img_and, img_and, 9);
		Erode(img_and, img_and, 9);
		findContours(img_and, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		cout << contours.size();
		for (int j = 0; j < contours.size(); j++) {
			RotatedRect rectTmp = minAreaRect(contours.at(j));
			// compare area and width height defect
			if (rectTmp.boundingRect().area() > 500 && (double)(MAX(rectTmp.size.height, rectTmp.size.width) >= (double)(MIN(rectTmp.size.height, rectTmp.size.width)*2)) &&
				(MAX(rectTmp.size.height, rectTmp.size.width) >= 100)) {
				drawContours(img_result, contours, j,  Scalar(255, 255, 255), CV_FILLED);
				DrawRotatedRectangle(img_src, minAreaRect(contours[j]), Scalar(255, 0, 0));
			}
			else if (rectTmp.boundingRect().area() > 50000) {
				drawContours(img_result, contours, j, Scalar(255, 255, 255), CV_FILLED);
				DrawRotatedRectangle(img_src, minAreaRect(contours[j]), Scalar(255, 0, 0));
			}
		}
#ifdef DEBUG
		imwrite(path + to_string(i) + "img_result.png", img_result);
		imwrite(path + to_string(i) + "img_src_result.png", img_src);
#endif // DEBUG

	}
	return true;
};