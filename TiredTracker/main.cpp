#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include "constants.h"

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat prevgray;

void findEyes(Mat frame_gray, Rect face);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double scale, const Scalar& color);

int main()
{
	CascadeClassifier face_cascade, eye_cascade;
	if (!face_cascade.load(frontalFacePath)) {
		printf("Error loading cascade file for face");
		return 1;
	}

	namedWindow(myWholeFaceWin, CV_WINDOW_NORMAL);
	moveWindow(myWholeFaceWin, 10, 100);
	//namedWindow(leftEyeWin, CV_WINDOW_NORMAL);
	//moveWindow(leftEyeWin, 10, 100);
	//namedWindow(rightEyeWin, CV_WINDOW_NORMAL);
	//moveWindow(rightEyeWin, 10, 100);
	namedWindow(leftEyeFloatWin, CV_WINDOW_NORMAL);
	moveWindow(leftEyeFloatWin, 10, 100);


	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		printf("error to initialize camera");
		return 1;
	}

	Mat cap_img, gray_img;
	vector<Rect> faces;
	while (1)
	{
		capture >> cap_img;

		if (cap_img.empty()) {												//sometimes first or second image from camera is empty (camera is loading)
			cout << "captured image is empty";
			continue;
		}

		flip(cap_img, cap_img, 1);											//for left eye to be on left side and right eye on the right side image must be vertically flipped

		cvtColor(cap_img, gray_img, CV_BGR2GRAY);
		equalizeHist(gray_img, gray_img);

		face_cascade.detectMultiScale(gray_img, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, Size(150, 150), Size(200, 200));

		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(cap_img, faces[i], 1234);
			findEyes(gray_img, faces[i]);
		}

		imshow("Result", cap_img);

		char c = waitKey(3);
		if (c == 27)
			break;
	}
	return 0;
}

float euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double scale, const Scalar& color)
{
	//float flx = 0;
	//float fly = 0;
	for (int y = 0; y < cflowmap.rows; y += step)
	{
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);

			int x2 = cvRound(x + fxy.x);
			int y2 = cvRound(y + fxy.y);

			//line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
			//circle(cflowmap, Point(x, y), 2, color, -1);
			//float flx = x - fxy.x;
			//cout << fxy.x;
			//cout << "\n";
			float distance = euclideanDist(Point(x, y), Point(x2, y2));
			//cout << distance;

			if (distance > 1.5) {
				if (fxy.y < 0) {
					line(cflowmap, Point(x, y), Point(x2, y2), CV_RGB(0, 0, 255));			//modre, ukazuje pohyb smerom vlavo (fxy.x < 0) alebo hore (fxy.y < 0)
					//circle(cflowmap, Point(x, y), 2, color, -1);
				}
				else {
					line(cflowmap, Point(x, y), Point(x2, y2), color);
				}
			}
		}
	}

}

void findEyes(Mat frame_gray, Rect face) {
	Mat faceROI = frame_gray(face);

	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth / 100.0);
	int eye_region_height = face.width * (kEyePercentHeight / 100.0);
	int eye_region_top = face.height * (kEyePercentTop / 100.0);
	Rect leftEyeRegion(face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width, eye_region_height);
	Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width, eye_region_height);


	Mat newFace;
	faceROI.copyTo(newFace);
	rectangle(newFace, leftEyeRegion, 200);
	rectangle(newFace, rightEyeRegion, 200);
	//imshow(myWholeFaceWin, newFace);

	Mat leftEye = faceROI(leftEyeRegion);

	/*circle(leftEye, Point(30, 30), 30, CV_RGB(255, 255, 255), 20, 8, 0);*/

	Mat rightEye = faceROI(rightEyeRegion);

	//imshow(leftEyeWin, leftEye);
	//imshow(rightEyeWin, rightEye);

	

	if (prevgray.data)
	{
		Mat flow, cflow;

		//Mat Transform;
		//Mat Transform_avg = Mat::eye(2, 3, CV_64FC1);
		//Mat warped;
		//Transform = estimateRigidTransform(leftEye, prevgray, 0);
		//Transform(Range(0, 2), Range(0, 2)) = Mat::eye(2, 2, CV_64FC1);
		//Transform_avg += (Transform - Transform_avg) / 2.0;
		//warpAffine(leftEye, warped, Transform_avg, Size(leftEye.cols, leftEye.rows));

		//imshow("Camw", warped);

		resize(leftEye, leftEye, prevgray.size());

		circle(leftEye, Point(prevgray.size().width / 2, prevgray.size().height / 2), 45, CV_RGB(255, 255, 255), 40, 8, 0);

		//calcOpticalFlowFarneback(prevgray, leftEye, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		//cvtColor(prevgray, cflow, CV_GRAY2BGR);
		////drawOptFlowMap(flow, cflow, 16, 1.5, CV_RGB(0, 255, 0));
		//drawOptFlowMap(flow, cflow, 4, 0, CV_RGB(0, 255, 0));
		//imshow(leftEyeFloatWin, cflow);
		////imshow("flow", flow);

		imshow(leftEyeFloatWin + "1", leftEye);

		int darkestPixel = 255;			// 255 - white, 0 - black

		for (int j = 0; j<leftEye.rows; j++)
		{
			for (int i = 0; i<leftEye.cols; i++)
			{
				if (leftEye.at<uchar>(j, i) < darkestPixel) {
					darkestPixel = leftEye.at<uchar>(j, i);
				}
			}
		}

		for (int j = 0; j<leftEye.rows; j++)
		{
			for (int i = 0; i<leftEye.cols; i++)
			{
				if (leftEye.at<uchar>(j, i) < darkestPixel + 1) {			//if the pixel is darker
					leftEye.at<uchar>(j, i) = 0;
				}
			}
		}

		cout << darkestPixel;
		cout << "\n";

		cv::threshold(leftEye, leftEye, 0, 100, cv::THRESH_BINARY);

		imshow(leftEyeFloatWin, leftEye);
	}

	
	swap(prevgray, leftEye);
	

	
}
