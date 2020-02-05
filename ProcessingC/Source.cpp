#include <iostream>
#include "ImageProcessing.h"
#include "ScratchProcessing.h"
//#define MDEBUG
int main(int argc, const char* argv[]) {
    // insert code here...
    std::cout << "Hi Project!\n";
    //String path = "C:/Users/nvnhu/Downloads/45Degree/45Degree/Lower/Test/";
	String path = "C:/Users/nvnhu/Downloads/ThorLabsSampleCaptured/Test/";
    vector<Mat> vtImage;
    Mat img_result, img_mask;
    readImage(path, vtImage);
    //processToSearchingLight(vtImage, img_mask, img_result);
    //imwrite(path + "Result/img_img_mask.png", img_mask);
    //imwrite(path + "Result/img_result.png", img_result);
	detectScratch(vtImage);
    return 0;
}