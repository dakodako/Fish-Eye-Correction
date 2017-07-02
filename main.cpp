//
//  main.cpp
//  CameraCalibration
//
//  Created by Didi Chi on 1/7/17.
//  Copyright © 2017 dako. All rights reserved.
//

#include <iostream>
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <stdio.h>
using namespace cv;
string videoPath = "CameraCalibration2.mp4";
int main()
{
    int numBoards = 2;
    Mat test_image = imread("testCalibration11.jpg");
    Mat test_image_2 = imread("testCalibration10.jpg");
    //Mat test_image_3 = imread("");
    vector<Mat> test_images;
    test_images.push_back(test_image);
    test_images.push_back(test_image_2);
    //test_images.push_back(test_image_3);
    int numCornersHor = 7;
    int numCornersVer = 7;
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor,numCornersVer);
    vector<vector<Point3f>> object_points;//3D points, the physical position of the corners
    vector<vector<Point2f>> image_points;//2D points, the location of the corners on the image (in 2 Dimentsions)
    vector<Point2f> corners;// keep track of successfully capturing a chessboard and save into the lists declared above
    int successes = 0;
    vector<Point3f> obj;
    for (int j = 0; j < numSquares; j++)
    {
        obj.push_back(Point3f(j/numCornersHor,j%numCornersHor,0.0f));
    }
    Mat gray_image;
    for (int i = 0; i < numBoards; i++)
    {
        cvtColor(test_images[i], gray_image, CV_BGR2GRAY);
        bool found = findChessboardCorners(test_images[0], board_sz, corners,CV_CALIB_CB_ADAPTIVE_THRESH);
        if(found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
            successes = successes + 1;
            image_points.push_back(corners);
            object_points.push_back(obj);
            i = i + 1;
        }
    }
    
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;

    calibrateCamera(object_points, image_points, test_images[1].size(), intrinsic, distCoeffs, rvecs, tvecs);
    Mat distortedImage;
    Mat undistortedImage;
    distortedImage = imread("distorted3.jpg");
    undistort(distortedImage, undistortedImage, intrinsic, distCoeffs);
    //imshow("win1", distortedImage);
    //imshow("win2", undistortedImage);
    imwrite("undistorted3.jpg", undistortedImage);
    //return 0;
    /*
    cvtColor(test_images[0], gray_image, CV_BGR2GRAY);
    //Mat blurred;
    //GaussianBlur(gray_image,blurred, Size(0,0), 5) ;
    //imshow("blurred",blurred);
    //waitKey(0);
    
    printf("finding corners\n");
    //bool found = findChessboardCorners(test_images[0], board_sz, corners,CV_CALIB_CB_ADAPTIVE_THRESH);
    printf("ahhhh what happend!!!\n");
    /*
    if(found)
    {
        cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
        drawChessboardCorners(gray_image, board_sz, corners, found);
        printf("draw board\n");
        //imshow("win1", test_image);
        //imshow("win2", gray_image);
        imwrite("corners11.jpg",gray_image);
        //waitKey(0);
        
    }
    else
    {
        printf("not found\n");
    }
    */
    return 0;
}
/*int main()
{
    int numBoards = 3;
    int numCornersHor = 8;
    int numCornersVer = 5;
    
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);
    VideoCapture capture = VideoCapture(videoPath);
    vector<vector<Point3f>> object_points;//3D points, the physical position of the corners
    vector<vector<Point2f>> image_points;//2D points, the location of the corners on the image (in 2 Dimentsions)
    vector<Point2f> corners;// keep track of successfully capturing a chessboard and save into the lists declared above
    int successes = 0;
    Mat image;
    Mat gray_image;
    capture >> image;
    vector<Point3f> obj;
    for (int j = 0; j < numSquares; j++)
    {
        obj.push_back(Point3f(j/numCornersHor,j%numCornersHor,0.0f));
    }
    
    while (successes < numBoards)
    {
        cvtColor(image, gray_image, CV_BGR2GRAY);
        bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        if (found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
            imshow("win1", image);
            imshow("win2", gray_image);
        }
        
        capture >> image;
        int key = waitKey(1);
        if (key == 27)
            return 0;
        if (key == ' ' && found != 0)
        {
            image_points.push_back(corners);
            object_points.push_back(obj);
            printf("Snap stored!");
            successes++;
            printf("number of successes %d\n",successes);
            if (successes >= numBoards)
                break;
        }
    }
    capture.release();
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    
    calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
    Mat distortedImage;
    Mat undistortedImage;
    distortedImage = imread("distorted1.jpg");
    undistort(distortedImage, undistortedImage, intrinsic, distCoeffs);
    //imshow("win1", distortedImage);
    //imshow("win2", undistortedImage);
    imwrite("undistorted1.jpg", undistortedImage);
    return 0;
}*/
