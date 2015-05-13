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

/* Public variables */
Mat prevgray;

Mat previousFace;
Mat currentFace;

bool leftEyeOpen = true;
bool rightEyeOpen = true;
int calibrationFace = calibrationDefault;


/* Functions */
//void findEyes(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double scale, const Scalar& color);
void headTracing(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace, Rect &detectedFaceRegion);
void calcFlow(const Mat& flow, Mat& cflowmap, double scale, int &globalMovementX, int &globalMovementY);
Rect findBiggestFace(Mat matCapturedGrayImage, CascadeClassifier cascFace);
void eyeTracking(Mat &matCurrentEye, Mat &matPreviousEye);
void getEyesFromFace(Mat &matFace, Mat &matLeftEye, Mat &matRightEye);
void detectBlink(Mat &matEyePrevious, Mat &matEyeCurrent, String eye, bool &eyeOpen);

/**************************************************************************   main   **************************************************************************/

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

	Rect detectedFaceRegion;

	////namedWindow(myWholeFaceWin, CV_WINDOW_NORMAL);
	////moveWindow(myWholeFaceWin, 10, 100);
	//namedWindow(leftEyeWin, CV_WINDOW_NORMAL);
	//moveWindow(leftEyeWin, 10, 100);
	//namedWindow(rightEyeWin, CV_WINDOW_NORMAL);
	//moveWindow(rightEyeWin, 10, 100);
	//namedWindow(leftEyeFloatWin, CV_WINDOW_NORMAL);
	//moveWindow(leftEyeFloatWin, 10, 100);

	//namedWindow("matLeftEye", CV_WINDOW_NORMAL);
	//moveWindow("matLeftEye", 10, 100);
	////namedWindow("leftEye", CV_WINDOW_NORMAL);
	////moveWindow("leftEye", 10, 100);

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

		if (matCapturedImage.empty()) {											//sometimes first or second image from camera is empty (camera is loading)
			cout << "captured image is empty\n";
			continue;
		}

		flip(matCapturedImage, matCapturedImage, 1);							//for left eye to be on left side and right eye on the right side image must be vertically flipped
		cvtColor(matCapturedImage, matCapturedGrayImage, CV_BGR2GRAY);
		//equalizeHist(matCapturedGrayImage, matCapturedGrayImage);

		//findEyes(matCapturedGrayImage, matCapturedImage, cascEye, cascFace);
		headTracing(matCapturedGrayImage, matCapturedImage, cascEye, cascFace, detectedFaceRegion);

		char c = waitKey(3);
		if (c == 27)															//pressed ESC
			break;
	}
	return 0;
}


/**************************************************************************   euclideanDist   **************************************************************************/

float euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}


/**************************************************************************   drawOptFlowMap   **************************************************************************/

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


/**************************************************************************   calcFlow   **************************************************************************/

///<summary> Vypocita pohyb tvare a v premennych globalMovementX a globalMovementY vrati vypocitane hodnoty </summary>
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

	globalMovementX = (localMovementX / (cflowmap.cols * cflowmap.rows))*2;
	globalMovementY = (localMovementY / (cflowmap.rows * cflowmap.cols))*2;
}


/**************************************************************************   calcFlow   **************************************************************************/

///<summary> Vypocita pohyb tvare a v premennych globalMovementX a globalMovementY vrati vypocitane hodnoty </summary>
void calcFlowEyes(const Mat& flow, Mat& cflowmap, int step, int &movementX, int &movementY)
{
	movementX = 0;
	movementY = 0;

	for (int y = 0; y < cflowmap.rows; y += step)
	{
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);

			movementX = movementX + fxy.x;
			movementY = movementY + fxy.y;
		}
	}
}

/**************************************************************************   headTracing   **************************************************************************/

void headTracing(Mat matCapturedGrayImage, Mat matCapturedImage, CascadeClassifier cascEye, CascadeClassifier cascFace, Rect &detectedFaceRegion) {
	
	Rect face = findBiggestFace(matCapturedGrayImage, cascFace);
	if (face.width == 0 && face.height == 0) {
		imshow("Result", matCapturedImage);				// just face
		return;											// no face was found
	}
	
	calibrationFace = calibrationFace - 1;

	if (detectedFaceRegion.height == 0 || calibrationFace < 1) {
		detectedFaceRegion = face;
		previousFace = matCapturedGrayImage(face);
		cout << "first time\n";
		calibrationFace = calibrationDefault;
	}
	else {												//uz je raz zachyteny frame s tvarou; teraz idem hladat pohyb oproti predchadzajucemu v tejto ploche
		
		currentFace = matCapturedGrayImage(detectedFaceRegion);

		imshow("currentFace", currentFace);
		imshow("previousFace", previousFace);

		Mat flow, cflow;
		calcOpticalFlowFarneback(previousFace, currentFace, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		cvtColor(previousFace, cflow, CV_GRAY2BGR);

		int globalMovementX, globalMovementY;

		calcFlow(flow, cflow, 1, globalMovementX, globalMovementY);


		detectedFaceRegion.x = detectedFaceRegion.x + globalMovementX;		//na zaklade analyzovaneho posunu previousFace a currentFace sa nastavia nove hodnoty, kde sa tvar nachadza
		detectedFaceRegion.y = detectedFaceRegion.y + globalMovementY;

		if (detectedFaceRegion.x < 0) {										//ked ide rectangle mimo zobrazenej plochy
			detectedFaceRegion.x = 0;
		}
		if (detectedFaceRegion.y < 0) {
			detectedFaceRegion.y = 0;
		}

		if (detectedFaceRegion.x + detectedFaceRegion.width > matCapturedImage.size().width - 1) {										//ked ide rectangle mimo zobrazenej plochy
			detectedFaceRegion.x = matCapturedImage.size().width - detectedFaceRegion.width - 1;
		}
		if (detectedFaceRegion.y + detectedFaceRegion.height > matCapturedImage.size().height - 1) {
			detectedFaceRegion.y = matCapturedImage.size().height - detectedFaceRegion.height - 1;
		}

		rectangle(matCapturedImage, detectedFaceRegion, 12);				//na povodnom obrazku sa ukaze novo posunuty rectangle
		currentFace = matCapturedGrayImage(detectedFaceRegion);				//currentFace sa posunie na novu poziciu, na ktoru podla porovnania s previousFace patri

		/* teraz je nutne porovnat predch. tvar a aktualnu a pozriet float uz konkretnych oci */

		eyeTracking(currentFace, previousFace);

		swap(previousFace, currentFace);									//previousFace je od teraz currentFace
	}

	rectangle(matCapturedImage, face, 1234);								//make rectangle around face

	if (leftEyeOpen) {
		putText(matCapturedImage, "Left eye open", cvPoint(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(50, 50, 50), 1, CV_AA);
	}
	else {
		putText(matCapturedImage, "Left eye closed", cvPoint(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(50, 50, 50), 1, CV_AA);
	}

	if (rightEyeOpen) {
		putText(matCapturedImage, "Right eye open", cvPoint(450, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(50, 50, 50), 1, CV_AA);
	}
	else {
		putText(matCapturedImage, "Right eye closed", cvPoint(450, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(50, 50, 50), 1, CV_AA);
	}

	imshow("Result", matCapturedImage);										//show face with rectangle
}


/**************************************************************************   eyeTracking   **************************************************************************/

void eyeTracking(Mat &matCurrentFace, Mat &matPreviousFace) {

	Mat matLeftEyePrevious;
	Mat matRightEyePrevious;
	Mat matLeftEyeCurrent;
	Mat matRightEyeCurrent;

	getEyesFromFace(matPreviousFace, matLeftEyePrevious, matRightEyePrevious);
	getEyesFromFace(matCurrentFace, matLeftEyeCurrent, matRightEyeCurrent);

	imshow("matLeftEyePrevious", matLeftEyePrevious);
	imshow("matRightEyePrevious", matRightEyePrevious);
	imshow("matLeftEyeCurrent", matLeftEyeCurrent);
	imshow("matRightEyeCurrent", matRightEyeCurrent);


	detectBlink(matLeftEyePrevious, matLeftEyeCurrent, "left", leftEyeOpen);
	detectBlink(matRightEyePrevious, matRightEyeCurrent, "right", rightEyeOpen);
}


void detectBlink(Mat &matEyePrevious, Mat &matEyeCurrent, String eye, bool &eyeOpen) {
	Mat leftFlow, leftCflow;
	calcOpticalFlowFarneback(matEyePrevious, matEyeCurrent, leftFlow, 0.5, 3, 15, 3, 5, 1.2, 0);
	cvtColor(matEyePrevious, leftCflow, CV_GRAY2BGR);
	int movementX, movementY;

	calcFlowEyes(leftFlow, leftCflow, 1, movementX, movementY);


	if (movementY == 0) {
		return;
	}

	if (movementY > 0 && eyeOpen) {
		eyeOpen = false;
		cout << eye;
		cout << "IS CLOSED, localmovementX=";
		cout << movementX;
		cout << ", localmovementY=";
		cout << movementY;
		cout << "\n";
		cout << '\a';
	}
	else if (movementY < 0 && !eyeOpen){
		eyeOpen = true;
		cout << eye;
		cout << "IS OPEN, localmovementX=";
		cout << movementX;
		cout << ", localmovementY=";
		cout << movementY;
		cout << "\n";
		cout << '\a';
	}
}

/**************************************************************************   getEyesFromFace   **************************************************************************/

///<description>get left and right eye from face (in one frame)</description>
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
//	imshow("Result", matCapturedImage);
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
//		////int darkestPixel = 255;			// 255 - white, 0 - black
//
//		////for (int j = 0; j<matLeftEyeRegion.rows; j++)
//		////{
//		////	for (int i = 0; i<matLeftEyeRegion.cols; i++)
//		////	{
//		////		if (matLeftEyeRegion.at<uchar>(j, i) < darkestPixel) {
//		////			darkestPixel = matLeftEyeRegion.at<uchar>(j, i);
//		////		}
//		////	}
//		////}
//
//		////for (int j = 0; j<matLeftEyeRegion.rows; j++)
//		////{
//		////	for (int i = 0; i<matLeftEyeRegion.cols; i++)
//		////	{
//		////		if (matLeftEyeRegion.at<uchar>(j, i) < darkestPixel + 1) {			//if the pixel is darker
//		////			matLeftEyeRegion.at<uchar>(j, i) = 0;
//		////		}
//		////	}
//		////}
//
//		////cout << darkestPixel;
//		////cout << "\n";
//
//		////cv::threshold(matLeftEyeRegion, matLeftEyeRegion, 0, 100, cv::THRESH_BINARY);
//
//		//imshow(leftEyeFloatWin, matLeftEyeRegion);
//	}
//
//	
//	swap(prevgray, matLeftEyeRegion);
//}



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