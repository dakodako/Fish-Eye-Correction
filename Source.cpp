#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>


#include <math.h>
#include <sstream>
#include <string>
#include <fstream>

#include <time.h>
#define PI 3.1415926
#define IMAGEWIDTH 4500 //NEED TO MEASURE THE IMAGE_WIDTH
using namespace std;
using namespace cv;
Mat correct(Mat src,double focal,double fov)
{
	//int width = src.cols;
	//int height = src.rows;
	
	
	Size imgSize(src.cols, src.rows);
	// imgArea(0, 0, src.cols, src.rows);
	Mat retImg(imgSize, CV_8UC3, Scalar(0, 0, 0));
	Mat_<Vec3b> _retImg = retImg;
	Mat_<Vec3b> _src = src;
	double xf, yf;
	for (int j = 0; j < imgSize.height; j++)
	{
		for (int i = 0; i < imgSize.width; i++)
		{
			//double xp = 1.0f * (i - 960);
			//double yp = 1.0f * (j + 540);
			//printf("xp = %f,yp = %f\n", xp, yp);
			double lambda = 1.0f*(sqrt((i - 960)* (i - 960) + (j + 540)*(j + 540))) / focal;
			if (lambda != 0)
			{
				xf = (2.0f * (i - 960) * sin(atan(lambda*1.0f/2))) / lambda + 960;
				yf = (2.0f * (j + 540) * sin(atan(lambda*1.0f/2))) / lambda - 540;
			}
		//	printf("xf = %f,yf = %f\n", xf, yf);
			//xf = xf + 960;
			//yf = yf - 540;

			if (xf >= 0 && xf < src.cols && yf >= 0 && yf < src.rows)
			{
				_retImg.at<Vec3b>(j, i) = _src.at<Vec3b>(yf, xf);
			}
		}
	}
	return retImg;
	
}
int main(int argc, char** argv)
{	
	time_t start, end;

	const string location = "D:\\Documents\\Images\\attempt3_4.jpg";
	Mat src  = imread(location);

	double FOV = (175.0/180)*PI;
	double f =IMAGEWIDTH / (4.0*sin(FOV/2));
	//double f = 5.0;
	cout << "f is " << f << endl;
	time(&start);
	src = correct(src,f,FOV);
	time(&end);
	printf("time spend %d\n", end - start);
	printf("fps = %f\n", 1.0 / (end - start));
	imshow("1", src);
	imwrite("D:\\Documents\\Images\\attemptserror.jpg", src);
	waitKey(0);
	return 0;
}