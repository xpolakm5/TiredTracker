/*
Semestralny projekt - Pocitacove videnie
2014/2015
xpolakm5
*/

#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

#include <iomanip>
#include <chrono>
#include <ctime>
#include <thread>

#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include "constants.h"

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

/* Global variables */
//Mat prevgray;					// first attempt

Mat previousFace;
Mat currentFace;

bool leftEyeOpen = true;
bool rightEyeOpen = true;
int calibrationFace = calibrationDefault;
int blinkNumberLeft = 0;
int blinkNumberRight = 0;

clock_t leftEyeCloseTime;
clock_t rightEyeCloseTime;


/* Functions */

//void findEyes(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace);			// first attempt
//void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double scale, const Scalar& color);								// first attempt

void headTracing(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace, Rect &detectedFaceRegion);
void calcFlow(const Mat& flow, Mat& cflowmap, double scale, int &globalMovementX, int &globalMovementY);
Rect findBiggestFace(Mat matCapturedGrayImage, CascadeClassifier cascFace);
void eyeTracking(Mat &matCurrentEye, Mat &matPreviousEye);
void getEyesFromFace(Mat &matFace, Mat &matLeftEye, Mat &matRightEye);
void detectBlink(Mat &matEyePrevious, Mat &matEyeCurrent, String eye, bool &eyeOpen, int &blinkNumber, clock_t &closeTime);


/**************************************************************************   to_string_with_precision   **************************************************************************/

template <typename T>
///<summary>From (for example double) value makes string with just whole numbers</summary>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out << std::setprecision(n) << a_value;
	return out.str();
}


/**************************************************************************   main   **************************************************************************/

///<summary> Application will detect eye blinking by calculating optical flow of face and eyes </summary>
int main()
{
	CascadeClassifier cascFace, cascEye;
	if (!cascFace.load(frontalFacePath)) {
		printf("Error loading cascade file for face");
		return 1;
	}
	if (!cascEye.load(eyePath)) {
		printf("Error loading cascade file for eye");
		return 1;
	}

	Rect detectedFaceRegion;														// have to be created here; if it was inside while the image would be still changing

	cout << "\n\tESC - turn this program off\n\tf - recalibrate face\n\tc - reset counter\n\n";

	namedWindow("Result", CV_WINDOW_NORMAL);

	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		printf("error to initialize camera");
		return 1;
	}

	Mat matCapturedImage;

	while (1)
	{
		Mat matCapturedGrayImage;
		capture >> matCapturedImage;

		if (matCapturedImage.empty()) {													//sometimes first or second image from camera is empty (camera is loading)
			//cout << "captured image is empty\n";
			continue;
		}

		flip(matCapturedImage, matCapturedImage, 1);									//for left eye to be on left side and right eye on the right side image must be vertically flipped
		cvtColor(matCapturedImage, matCapturedGrayImage, CV_BGR2GRAY);
		//equalizeHist(matCapturedGrayImage, matCapturedGrayImage);						// disabled; problems with dark areas

		//findEyes(matCapturedGrayImage, matCapturedImage, cascEye, cascFace);							// first approach; not working
		headTracing(matCapturedGrayImage, matCapturedImage, cascEye, cascFace, detectedFaceRegion);		// second approach


		switch (waitKey(3)) {
		case 27:																		// ESC key - end this program
			return 0;
			break;
		
		case 102:																		// f key - recalibrate face
			calibrationFace = 0;
			break;

		case 99:																		// c key - reset Counters
			leftEyeOpen = true;
			rightEyeOpen = true;
			blinkNumberLeft = 0;
			blinkNumberRight = 0;
			break;
		}
	}
	return 0;
}


/**************************************************************************   calcFlow   **************************************************************************/

///<summary> Measure face movement, return values are globalMovementX and globalMovementY ("clean" values that means real pixel movement by for example 2 pixels) </summary>
void calcFlow(const Mat& flow, Mat& cflowmap, int step, int &globalMovementX, int &globalMovementY)
{
	int localMovementX = 0;
	int localMovementY = 0;

	for (int y = 0; y < cflowmap.rows; y += step)
	{
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);

			localMovementX = localMovementX + fxy.x;
			localMovementY = localMovementY + fxy.y;
		}
	}

	globalMovementX = (localMovementX / (cflowmap.cols * cflowmap.rows))*2;							//these are usable values for global movement of face (for example + 2 pixels to axis x)
	globalMovementY = (localMovementY / (cflowmap.rows * cflowmap.cols))*2;
}


/**************************************************************************   calcFlow   **************************************************************************/

///<summary> Measure eye movement, return values are movementX and movementY (raw values of movement, for example 500 pixels moved to the right) </summary>
void calcFlowEyes(const Mat& flow, Mat& cflowmap, int step, int &movementX, int &movementY)
{
	movementX = 0;
	movementY = 0;

	for (int y = 0; y < cflowmap.rows; y += step)
	{
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);

			movementX = movementX + fxy.x;														//these are raw values of movement (not divided by number of pixels)
			movementY = movementY + fxy.y;
		}
	}
}

/**************************************************************************   headTracing   **************************************************************************/

///<summary> On input there are captured images, classifiers and output is saved in detectedFaceRegion. This have to be this way, otherwise the image was changing (have to be reference)</summary>
void headTracing(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace, Rect &detectedFaceRegion) {
	
	Rect face = findBiggestFace(matCapturedGrayImage, cascFace);
	if (face.width == 0 && face.height == 0) {
		imshow("Result", matCapturedImage);									// just face
		return;																// no face was found
	}
	
	calibrationFace = calibrationFace - 1;

	if (detectedFaceRegion.height == 0 || calibrationFace < 1) {			//first frame cannot calculate flow; there have to be previous frame to do that; face is calibrated each "x" frames (here)
		detectedFaceRegion = face;
		previousFace = matCapturedGrayImage(face);
		calibrationFace = calibrationDefault;								//reset calibration number to default (from constants.h)
	}
	else {																	//first frame captured with face; now will be calculated optical flow
		
		currentFace = matCapturedGrayImage(detectedFaceRegion);

		//imshow("currentFace", currentFace);
		//imshow("previousFace", previousFace);

		Mat flow, cflow;
		calcOpticalFlowFarneback(previousFace, currentFace, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		cvtColor(previousFace, cflow, CV_GRAY2BGR);

		int globalMovementX, globalMovementY;

		calcFlow(flow, cflow, 1, globalMovementX, globalMovementY);


		detectedFaceRegion.x = detectedFaceRegion.x + globalMovementX;		//move rectangle to a new place (according to calculated optical flow)
		detectedFaceRegion.y = detectedFaceRegion.y + globalMovementY;

		if (detectedFaceRegion.x < 0) {										//when the rectangle is out of visible window; so it doesn't crash
			detectedFaceRegion.x = 0;
		}
		if (detectedFaceRegion.y < 0) {
			detectedFaceRegion.y = 0;
		}

		if (detectedFaceRegion.x + detectedFaceRegion.width > matCapturedImage.size().width - 1) {				// when the rectangle is out of visible window; so it doesn't crash
			detectedFaceRegion.x = matCapturedImage.size().width - detectedFaceRegion.width - 1;
		}
		if (detectedFaceRegion.y + detectedFaceRegion.height > matCapturedImage.size().height - 1) {
			detectedFaceRegion.y = matCapturedImage.size().height - detectedFaceRegion.height - 1;
		}

		rectangle(matCapturedImage, detectedFaceRegion, 12);				//na povodnom obrazku sa ukaze novo posunuty rectangle
		currentFace = matCapturedGrayImage(detectedFaceRegion);				//currentFace sa posunie na novu poziciu, na ktoru podla porovnania s previousFace patri

		/* teraz je nutne porovnat predch. tvar a aktualnu a pozriet float uz konkretnych oci */

		eyeTracking(currentFace, previousFace);								//when we have two stabilised faces (previous and current) now we can calculate their movement
		swap(previousFace, currentFace);									//previousFace is now currentFace and vice versa
	}

	rectangle(matCapturedImage, face, 1234);								//make rectangle around face

	if (leftEyeOpen) {																	//when the eye is open, circle will be drawn on top of the image
		circle(matCapturedImage, Point(40, 40), 20, Scalar(102, 255, 51), 40, 8, 0);
	}
	else {																				//when the eye is closed, time from closing is calculated and drawn on the image
		circle(matCapturedImage, Point(40, 40), 20, Scalar(0, 0, 255), 40, 8, 0);

		double diffticks = clock() - leftEyeCloseTime;
		double diffms = diffticks / (CLOCKS_PER_SEC / 1000);
		putText(matCapturedImage, to_string_with_precision(diffms), cvPoint(80, 100), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(50, 50, 50), 1, CV_AA);

		if (diffms > 5000) cout << "\a";												//when the time of closed is is bigger than 5 seconds, beeping starts

	}

	if (rightEyeOpen) {
		circle(matCapturedImage, Point(600, 40), 20, Scalar(102, 255, 51), 40, 8, 0);
	}
	else {
		circle(matCapturedImage, Point(600, 40), 20, Scalar(0, 0, 255), 40, 8, 0);

		double diffticks = clock() - rightEyeCloseTime;
		double diffms = diffticks / (CLOCKS_PER_SEC / 1000);
		putText(matCapturedImage, to_string_with_precision(diffms), cvPoint(520, 100), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(50, 50, 50), 1, CV_AA);

		if (diffms > 5000) cout << "\a";
	}

	putText(matCapturedImage, to_string(blinkNumberLeft), cvPoint(100, 45), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(100, 100, 100), 1, CV_AA);		//number of eye blinks
	putText(matCapturedImage, to_string(blinkNumberRight), cvPoint(520, 45), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(100, 100, 100), 1, CV_AA);

	imshow("Result", matCapturedImage);										//show face with rectangle
}


/**************************************************************************   eyeTracking   **************************************************************************/

///<summary> Select eyes from previous and current faces and detect blink </summary>
void eyeTracking(Mat &matCurrentFace, Mat &matPreviousFace) {

	Mat matLeftEyePrevious;
	Mat matRightEyePrevious;
	Mat matLeftEyeCurrent;
	Mat matRightEyeCurrent;

	getEyesFromFace(matPreviousFace, matLeftEyePrevious, matRightEyePrevious);
	getEyesFromFace(matCurrentFace, matLeftEyeCurrent, matRightEyeCurrent);

	//imshow("matLeftEyePrevious", matLeftEyePrevious);
	//imshow("matRightEyePrevious", matRightEyePrevious);
	//imshow("matLeftEyeCurrent", matLeftEyeCurrent);
	//imshow("matRightEyeCurrent", matRightEyeCurrent);

	detectBlink(matLeftEyePrevious, matLeftEyeCurrent, "left", leftEyeOpen, blinkNumberLeft, leftEyeCloseTime);				// each eye have its own blinking detection, timer and blink counter
	detectBlink(matRightEyePrevious, matRightEyeCurrent, "right", rightEyeOpen, blinkNumberRight, rightEyeCloseTime);
}

/**************************************************************************   detectBlink   **************************************************************************/

///<summary> Blink detection for previous and current frame of an eye, will calculate flow and increment counters of eye blinking </summary>
void detectBlink(Mat &matEyePrevious, Mat &matEyeCurrent, String eye, bool &eyeOpen, int &blinkNumber, clock_t &closeTime) {
	Mat leftFlow, leftCflow;
	calcOpticalFlowFarneback(matEyePrevious, matEyeCurrent, leftFlow, 0.5, 3, 15, 3, 5, 1.2, 0);
	cvtColor(matEyePrevious, leftCflow, CV_GRAY2BGR);
	int movementX, movementY;

	calcFlowEyes(leftFlow, leftCflow, 1, movementX, movementY);


	if (movementY == 0) {
		return;
	}

	if (movementY > 0 && eyeOpen) {						//eye is now closed

		closeTime = clock();

		eyeOpen = false;
		blinkNumber = blinkNumber + 1;					//increment blink count number for current eye
		//cout << eye;
		//cout << "IS CLOSED, localmovementX=";
		//cout << movementX;
		//cout << ", localmovementY=";
		//cout << movementY;
		//cout << "\n";
		//cout << '\a';
	}
	else if (movementY < 0 && !eyeOpen){				//eye is now open
		eyeOpen = true;
		//cout << eye;
		//cout << "IS OPEN, localmovementX=";
		//cout << movementX;
		//cout << ", localmovementY=";
		//cout << movementY;
		//cout << "\n";
		////cout << '\a';
	}
}

/**************************************************************************   getEyesFromFace   **************************************************************************/

///<summary>Get left and right eye from face (in one frame)</summary>
void getEyesFromFace(Mat &matFace, Mat &matLeftEye, Mat &matRightEye) {

	Size faceSize = matFace.size();

	int eye_region_width = faceSize.width * (kEyePercentWidth / 100.0);
	int eye_region_height = faceSize.width * (kEyePercentHeight / 100.0);
	int eye_region_top = faceSize.height * (kEyePercentTop / 100.0);
	Rect leftEyeRegion(faceSize.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width, eye_region_height);
	Rect rightEyeRegion(faceSize.width - eye_region_width - faceSize.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width, eye_region_height);

	matLeftEye = matFace(leftEyeRegion);
	matRightEye = matFace(rightEyeRegion);
}

/**************************************************************************   findBiggestFace   **************************************************************************/

///<summary>From grayscale image captured on camera will find one biggest face. If there is no face, returnValue will be empty.</summary>
Rect findBiggestFace(Mat matCapturedGrayImage, CascadeClassifier cascFace)  {

	Rect returnValue;

	vector<Rect> faces;
	cascFace.detectMultiScale(matCapturedGrayImage, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, Size(150, 150), Size(300, 300));

	if (faces.size() > 0) {
		return faces[0];
	}
	else {
		return returnValue;
	}
}












///*********************************************************************   NEPOUZIVANA CAST KODU - prve testovania   *********************************************************************/
//
//
//
//
///**************************************************************************   euclideanDist   **************************************************************************/
//
//float euclideanDist(Point& p, Point& q) {
//	Point diff = p - q;
//	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
//}
//
//
///**************************************************************************   drawOptFlowMap   **************************************************************************/
//
//void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double scale, const Scalar& color)
//{
//	//float flx = 0;
//	//float fly = 0;
//	for (int y = 0; y < cflowmap.rows; y += step)
//	{
//		for (int x = 0; x < cflowmap.cols; x += step)
//		{
//			const Point2f& fxy = flow.at<Point2f>(y, x);
//
//			int x2 = cvRound(x + fxy.x);
//			int y2 = cvRound(y + fxy.y);
//
//			//line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
//			//circle(cflowmap, Point(x, y), 2, color, -1);
//			//float flx = x - fxy.x;
//			//cout << fxy.x;
//			//cout << "\n";
//			float distance = euclideanDist(Point(x, y), Point(x2, y2));
//			//cout << distance;
//
//			if (distance > 1.5) {
//				if (fxy.y < 0) {
//					line(cflowmap, Point(x, y), Point(x2, y2), CV_RGB(0, 0, 255));			//modre, ukazuje pohyb smerom vlavo (fxy.x < 0) alebo hore (fxy.y < 0)
//					//circle(cflowmap, Point(x, y), 2, color, -1);
//				}
//				else {
//					line(cflowmap, Point(x, y), Point(x2, y2), color);
//				}
//			}
//		}
//	}
//}
//
///**************************************************************************   findEyes   **************************************************************************/
//
//void findEyes(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace) {
//
//	Rect face = findBiggestFace(matCapturedGrayImage, cascFace);
//	if (face.width == 0 && face.height == 0) {
//		return;											// no face was found
//	}
//
//	rectangle(matCapturedImage, face, 1234);
//	imshow("TiredTracker", matCapturedImage);
//
//	Mat matFoundFace = matCapturedGrayImage(face);
//
//	//-- Find eye regions and draw them
//	int eye_region_width = face.width * (kEyePercentWidth / 100.0);
//	int eye_region_height = face.width * (kEyePercentHeight / 100.0);
//	int eye_region_top = face.height * (kEyePercentTop / 100.0);
//	Rect leftEyeRegion(face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width, eye_region_height);
//	Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide / 100.0), eye_region_top, eye_region_width, eye_region_height);
//
//	Mat matLeftEyeRegion = matFoundFace(leftEyeRegion);
//	Mat matRightEyeRegion = matFoundFace(rightEyeRegion);
//
//	imshow(leftEyeWin, matLeftEyeRegion);
//	imshow(rightEyeWin, matRightEyeRegion);
//
//
//	if (prevgray.data)
//	{
//		vector<Rect> vecFoundEyes;
//		cascEye.detectMultiScale(matLeftEyeRegion, vecFoundEyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//
//		//Point center(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
//		//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
//		//circle(leftEye, center, radius, Scalar(255, 0, 0), 2, 8, 0);
//		
//		if (vecFoundEyes.size() > 0) {			
//			Mat matLeftEye = matLeftEyeRegion(vecFoundEyes[0]);
//
//			circle(matLeftEye, Point(matLeftEye.size().width / 2, matLeftEye.size().height / 2), 43, CV_RGB(255, 255, 255), 40, 8, 0);
//
//			imshow("matLeftEye", matLeftEye);
//		}
//
//
//		Mat flow, cflow;
//		resize(matLeftEyeRegion, matLeftEyeRegion, prevgray.size());
//
//		//circle(leftEye, Point(prevgray.size().width / 2, prevgray.size().height / 2), 45, CV_RGB(255, 255, 255), 40, 8, 0);
//
//		//calcOpticalFlowFarneback(prevgray, matLeftEyeRegion, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//		//cvtColor(prevgray, cflow, CV_GRAY2BGR);
//		////drawOptFlowMap(flow, cflow, 16, 1.5, CV_RGB(0, 255, 0));
//		//drawOptFlowMap(flow, cflow, 4, 0, CV_RGB(0, 255, 0));
//		//imshow(leftEyeFloatWin, cflow);
//		////imshow("flow", flow);
//
//		//imshow(leftEyeFloatWin + "1", leftEye);
//
//		int darkestPixel = 255;			// 255 - white, 0 - black
//
//		for (int j = 0; j<matLeftEyeRegion.rows; j++)
//		{
//			for (int i = 0; i<matLeftEyeRegion.cols; i++)
//			{
//				if (matLeftEyeRegion.at<uchar>(j, i) < darkestPixel) {
//					darkestPixel = matLeftEyeRegion.at<uchar>(j, i);
//				}
//			}
//		}
//
//		for (int j = 0; j<matLeftEyeRegion.rows; j++)
//		{
//			for (int i = 0; i<matLeftEyeRegion.cols; i++)
//			{
//				if (matLeftEyeRegion.at<uchar>(j, i) < darkestPixel + 1) {			//if the pixel is darker
//					matLeftEyeRegion.at<uchar>(j, i) = 0;
//				}
//			}
//		}
//
//		cout << darkestPixel;
//		cout << "\n";
//
//		cv::threshold(matLeftEyeRegion, matLeftEyeRegion, 0, 100, cv::THRESH_BINARY);
//
//		//imshow(leftEyeFloatWin, matLeftEyeRegion);
//	}
//
//	
//	swap(prevgray, matLeftEyeRegion);
//}