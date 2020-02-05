#pragma once
/// Apply the dilate operation
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
void readImage(String path, vector<Mat>& vtImage) {
    vector<cv::String> fn;
    glob(path + "*", fn, false);
    size_t count = fn.size();
    for (size_t i = 0; i < count; i++) {
		Mat img_read = imread(fn[i]);
		if (img_read.data != NULL) {
			vtImage.push_back(img_read);
		}
    }
}
void Dilate(Mat img_src, Mat& img_dst, int kernelSize) {
    Mat element = getStructuringElement(MORPH_RECT,
        Size(2 * kernelSize + 1, 2 * kernelSize + 1),
        Point(kernelSize, kernelSize));
    dilate(img_src, img_dst, element);
}
/// Apply the erosion operation
void Erode(Mat img_src, Mat& img_dst, int kernelSize) {
    Mat element = getStructuringElement(MORPH_RECT,
        Size(2 * kernelSize + 1, 2 * kernelSize + 1),
        Point(kernelSize, kernelSize));
    erode(img_src, img_dst, element);

}
float DistancePoints(Point pt1, Point pt2)
{
    int dx, dy;

    dx = abs(pt1.x - pt2.x);
    dy = abs(pt1.y - pt2.y);

    int temp = (dx * dx) + (dy * dy);
    float distance;
    if (temp > 0)
    {
        distance = sqrt(temp);
    }
    else
    {
        distance = 0;
    }

    return distance;
}
bool dilateLighting(Mat img_src, Mat& img_dst, Rect rectMask, Mat img_mask) {
    Mat img_thresold;
    threshold(img_src, img_thresold, 200, 255, CV_THRESH_BINARY);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat img_zero = Mat::zeros(img_src.rows, img_src.cols, CV_8UC3);
    findContours(img_thresold.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for (int i = 0; i < contours.size(); i++) {
        Rect rectTmp = boundingRect(contours.at(i));
        Rect rect_and = rectTmp & rectMask;
        Mat img_rectTmp = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);
        drawContours(img_rectTmp, contours, i, Scalar(255, 255, 255), -1);
        Mat img_and;
        bitwise_and(img_rectTmp, img_mask, img_and);
        if (rect_and.area() > 0 &&
            (((double)rect_and.area() / (double)MIN(rectTmp.area(), rectMask.area())) >= 0.8) && (rectTmp.area() < rectMask.area()) && countNonZero(img_and) > 0)
        {
            rectangle(img_zero, boundingRect(contours.at(i)), Scalar(255, 255, 255), -1);
            //drawContours(img_zero, contours, i, Scalar(255, 255, 255), -1);
        }

    }
    cvtColor(img_zero, img_zero, COLOR_RGB2GRAY);
    img_dst = img_zero;

    return true;
}
void imhist(Mat image, int histogram[])
{
	cout << "-------------------" << endl;
	// initialize all intensity values to 0
	for (int i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}

	// calculate the no of pixels for each intensity values
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			cout << (int)image.at<uchar>(y, x) << " ";
			histogram[(int)image.at<uchar>(y, x)]++;
		}
		cout << endl;
	}
}

bool processToSearchingLight(vector<Mat> vtSource, Mat& img_mask, Mat& img_result)
{
	String path = "C:/Users/nvnhu/Downloads/45Degree/45Degree/Lower/Test/Result/";
    vector <Mat> vtImgDilate, vtImgRmDefect;
	if (vtSource.size() < 1) {
		return false;
	}
    for (int i = 0; i < vtSource.size(); i++) {
        Mat img_src = vtSource.at(i).clone();
        Mat img_gray = Mat();
        // convert image
        cvtColor(img_src, img_gray, CV_RGB2GRAY);
		int histogram[256];
		imhist(img_gray.clone(), histogram);

        Mat img_blur, img_erode, img_dilate, img_blurOr, img_median, img_rmDefect;

        //filter Gaussian
        GaussianBlur(img_gray, img_blur, Size(5, 5), 0, 0);

        // filter image
        medianBlur(img_blur, img_dilate, 13);
        Dilate(img_dilate, img_dilate, 13);
        Erode(img_dilate, img_dilate, 13);

        // get background merge
        medianBlur(img_blur, img_median, 9);
        Erode(img_median, img_rmDefect, 9);
        Dilate(img_rmDefect, img_rmDefect, 9);

        // append image into list to process later
        vtImgDilate.push_back(img_dilate);
        vtImgRmDefect.push_back(img_rmDefect);
        Mat img_thresold, img_adaptiveThreshold;
        threshold(img_dilate, img_thresold, 80, 255, CV_THRESH_BINARY);
		//adaptiveThreshold(img_gray, img_adaptiveThreshold, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, img_gray.size().width - 1, -5);
		//imwrite(path + to_string(i) + "img_adaptiveThreshold.png", img_adaptiveThreshold);
		bitwise_not(img_thresold, img_thresold);
        if (i == 0) {
            img_mask = img_thresold;
            continue;
        }
        bitwise_or(img_mask, img_thresold, img_mask);
    }
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(img_mask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    Rect rectMaxContour(0, 0, 0, 0);
    float min_distance = max(img_mask.size().width, img_mask.size().height);
    int indexMaxContour = 0;
    Point pCenter = Point(img_mask.size().width / 2, img_mask.size().height / 2);
    Mat img_max_contour = Mat::zeros(img_mask.rows, img_mask.cols, CV_8UC1);
    for (int i = 0; i < contours.size(); i++) {
        Rect rectTmp = boundingRect(contours.at(i));
        if (rectTmp.size().width > img_mask.size().width / 3) {
            Point pointCntr = Point(rectTmp.x, rectTmp.y);
            float distance = DistancePoints(pCenter, pointCntr);
            if (min_distance > distance) {
                min_distance = distance;
                rectTmp = boundingRect(contours.at(i));
                rectMaxContour = rectTmp;
                indexMaxContour = i;
            }
        }
    }
    drawContours(img_max_contour, contours, indexMaxContour, Scalar(255), -1);
    img_mask = img_max_contour; // find mask

    // find light and merge image
    for (int i = 0; i < vtSource.size(); i++) {
        Mat img_src = vtSource.at(i).clone();
        Mat img_gray = Mat();
        cvtColor(img_src, img_gray, CV_RGB2GRAY);
        Mat img_dilate = vtImgDilate.at(i);
        Mat img_dilateLighting, img_thresold, img_and_original;
        dilateLighting(img_dilate, img_dilateLighting, rectMaxContour, img_mask);
        Dilate(img_dilateLighting, img_dilateLighting, 9);
        Mat img_rmDefect = vtImgRmDefect.at(i);

        contours.clear();
        hierarchy.clear();
        threshold(img_dilate, img_thresold, 80, 255, CV_THRESH_BINARY);


        bitwise_or(img_thresold, img_dilateLighting, img_thresold);
        Mat img_dilate_thresold = img_thresold;
        bitwise_or(img_dilate_thresold, img_rmDefect, img_dilate_thresold);
        bitwise_not(img_dilate_thresold, img_dilate_thresold);
        bitwise_and(img_dilate_thresold, img_gray, img_and_original);
        if (i == 0) {
            img_result = img_and_original * 0.5;
            continue;
        }
        img_result += img_and_original * 0.5;
    }
    return true;
}
void processingMerge() {
    vector<Mat> vtSource;
    Mat img_mask;
    Mat img_result;
    processToSearchingLight(vtSource, img_mask, img_result);
}
