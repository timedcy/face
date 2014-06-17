/*****************************************************************************
* 
* Copyright (c) 2008 by Yao Wei, all rights reserved.
*
* Author:      	Yao Wei
* Contact:     	njustyw@gmail.com
* 
* This software is partly based on the following open source: 
*  
*		- OpenCV 
* 
****************************************************************************/

#include <cstdio>
#include <string>
#include <highgui.h>

#include "AAM_IC.h"
#include "AAM_basic.h"
#include "AAM_VJFaceDetect.h"
#include "FacePredict.h"

using namespace std;

static std::string resultDir = "trainingSets/tmp/";
//std::string resultDir = "../test2/";

static void usage()
{
	printf("AAMFitting int string int int\n"
		"---- int:		0:search shape by aam, 1:read shape from file"
		"---- string:	image file\n"
		"---- int:		current age group\n"
		"---- int:		predicted age group\n");
		//"Group0:[0,1), Group1:[1,2), Group2:[2,3), Group3:[3,6), Group4:[6,10), Group5:[10,18]";
		"Group0:[0,2), Group1:[2,4), Group2:[4,6), Group3:[6,10), Group4:[10,15), Group5:[15,20), Group6:[20, 30), Group7: [30, 40), Group8: [40, 68) ";
	exit(0);
}


extern "C"
char* faceAging(string imgPath, int curAgeG, int newAgeG)
{
	static std::string resultDir = "trainingSets/tmp/";
    //load image
	IplImage* originalImage = cvLoadImage(imgPath.c_str(), 1);
	if(originalImage==0){
		fprintf(stderr, "ERROR(%s, %d): Cannot open image file %s!\n",
			__FILE__, __LINE__, imgPath.c_str());
		exit(0);
	}
	
	IplImage * image = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);
	cvCopy(originalImage, image);
	AAM_Shape Shape;
	AAM_Shape ShapeF;
	AAM_Shape ShapeM;


	//search shape by aam
	AAM * aam = NULL;
	int type;
	char tmpChar[2];

	itoa(curAgeG, tmpChar, 10);
	std::string aamFileName = resultDir + "Group" + std::string(tmpChar) + ".aam_ic";
	std::ifstream fs(aamFileName);
	if(fs == 0) {
		fprintf(stderr, "ERROR(%s: %d): Cannot open file %s!\n", __FILE__, __LINE__, resultDir+"Group"+ "_x.aam_ic");
		exit(0);
	}
	fs >> type;

	//aam-basic
	if(type == 0)		aam = new AAM_Basic;
	else if(type == 1)  aam = new AAM_IC;
	
	//read model from file
	aam->Read(fs);
	fs.close();

	//intial face detector
	AAM_VJFaceDetect fjdetect;
	fjdetect.LoadCascade("haarcascade_frontalface_alt2.xml");

	//detect face for intialization
	Shape = fjdetect.Detect(image, aam->GetMeanShape());

	//do image alignment
	aam->Fit(image, Shape, 30, false);  //if true, show process

	ofstream outfile;
	outfile.open( resultDir + "aam_result.txt" );
	Shape.Write( outfile );
	outfile.close();

	//show GUI
	cvNamedWindow("AAMFitting");
	aam->Draw(image, 0);
	cvShowImage("AAMFitting", image);
	

	//resize the current image
	cvSetImageROI(originalImage, cvRect(Shape.MinX(), Shape.MinY(), Shape.GetWidth(), Shape.GetHeight()));
	IplImage *facialImage = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);
	cvCopy(originalImage, facialImage, NULL);
	cvResetImageROI(originalImage);

	CvSize stdsize;
	stdsize.width = stdwidth;
	stdsize.height = stdwidth / facialImage->width * facialImage->height;
	IplImage *stdImage = cvCreateImage(stdsize, originalImage->depth, originalImage->nChannels);
	cvResize(facialImage, stdImage, CV_INTER_LINEAR);

	cvNamedWindow("CurrentFacialImage");
	cvShowImage("CurrentFacialImage", stdImage);

	//draw the shape
	CvSize ssize;
	ssize.width = 130;
	ssize.height = 130;
	IplImage *shapeImg = cvCreateImage(ssize, originalImage->depth, originalImage->nChannels);
	cvSet(shapeImg, CV_RGB(0,0,0));
	AAM_Shape temShape = Shape;
	double orgwid = temShape.MaxX() - temShape.MinX();
	double orghei = temShape.MaxY() - temShape.MinY();
	for (int i = 0; i < 68; i++) {
		temShape[i].x = (temShape[i].x - Shape.MinX()) * stdwidth / orgwid;
		temShape[i].y = (temShape[i].y - Shape.MinY()) * stdsize.height / orghei;
	}
	temShape.Sketch(shapeImg);
	cvShowImage("shape", shapeImg);
	cvReleaseImage(&shapeImg);


	//Facial Prediction
	FacePredict face_predict;
	std::string mfile = resultDir + "facial.predict_model";
	std::ifstream model(mfile);
	face_predict.Read(model);
	model.close();
	//predict(const AAM_Shape& shape, const IplImage& curImage, int curAgeG, int newAgeG, bool save)
	IplImage* newImage = face_predict.predict(Shape, *originalImage, /*ShapeF, *ImageF, ratioF, ShapeM, *ImageM, ratioM,*/ curAgeG, newAgeG, true);
	std::string newfile = std::string(imgPath);
	newfile = newfile.insert(newfile.find_last_of('/')+1, "result_" );
	newfile = newfile.insert(newfile.find_last_of('.'), std::string("_G"));
	cvSaveImage(newfile.c_str(), newImage);
	
	cvNamedWindow("PredictedFacialImage");
	cvShowImage("PredictedFacialImage", newImage);
	cvWaitKey(0);

	cvReleaseImage(&image);
	char* result = new char[128];
	strcpy(result, newfile.c_str());
	return result;
}

int main(int argc, char** argv)
{
	//if(argc < 5) usage();

	////load image
	//IplImage* originalImage = cvLoadImage(argv[2], 1);
	//if(originalImage==0){
	//	fprintf(stderr, "ERROR(%s, %d): Cannot open image file %s!\n",
	//		__FILE__, __LINE__, argv[2]);
	//	exit(0);
	//}

	//IplImage* ImageF; 
	//IplImage* ImageM; 
	//double ratioF = 0;
	//double ratioM = 0;
	//if (argc == 9) {
	//	ImageF = cvLoadImage(argv[5], 1);
	//	ImageM = cvLoadImage(argv[7], 1);
	//	double ratioF = atof(argv[6]);
	//	double ratioM = atof(argv[8]);
	//}
	//
	//IplImage * image = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);
	//cvCopy(originalImage, image);
	//AAM_Shape Shape;
	//AAM_Shape ShapeF;
	//AAM_Shape ShapeM;

	//if (atoi(argv[1]) == 0) {
	//	//search shape by aam
	//	AAM * aam = NULL;
	//	int type;
	//	std::string aamFileName = resultDir + "Group" + std::string(argv[3]) + ".aam_ic";
	//	std::ifstream fs(aamFileName);
	//	if(fs == 0) {
	//		fprintf(stderr, "ERROR(%s: %d): Cannot open file %s!\n", __FILE__, __LINE__, resultDir+"Group"+ std::string(argv[3]) +".aam_ic");
	//		exit(0);
	//	}
	//	fs >> type;

	//	//aam-basic
	//	if(type == 0)		aam = new AAM_Basic;
	//	else if(type == 1)  aam = new AAM_IC;
	//
	//	//read model from file
	//	aam->Read(fs);
	//	fs.close();

	//	//intial face detector
	//	AAM_VJFaceDetect fjdetect;
	//	fjdetect.LoadCascade("haarcascade_frontalface_alt2.xml");

	//	//detect face for intialization
	//	Shape = fjdetect.Detect(image, aam->GetMeanShape());

	//	//do image alignment
	//	aam->Fit(image, Shape, 30, false);  //if true, show process

	//	ofstream outfile;
	//	outfile.open( resultDir + "aam_result.txt" );
	//	Shape.Write( outfile );
	//	outfile.close();

	//	//show GUI
	//	cvNamedWindow("AAMFitting");
	//	aam->Draw(image, 0);
	//	cvShowImage("AAMFitting", image);
	//}
	//
	//else {
	//	//read shape from file
	//	std::string filename = std::string(argv[2]);
	//	filename = filename.substr(0, filename.length()-3) + "pts";
	//	Shape.ReadPTS(filename);

	//	if (argc == 9) {
	//		//read father's shape
	//		std::string filenameF = std::string(argv[5]);
	//		filenameF = filenameF.substr(0, filenameF.length()-3) + "pts";
	//		ShapeF.ReadPTS(filenameF);
	//		
	//		//read mother's shape
	//		std::string filenameM = std::string(argv[7]);
	//		filenameM = filenameM.substr(0, filenameM.length()-3) + "pts";
	//		ShapeM.ReadPTS(filenameM);
	//	}
	//}

	////resize the current image
	//cvSetImageROI(originalImage, cvRect(Shape.MinX(), Shape.MinY(), Shape.GetWidth(), Shape.GetHeight()));
	//IplImage *facialImage = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);
	//cvCopy(originalImage, facialImage, NULL);
	//cvResetImageROI(originalImage);

	//CvSize stdsize;
	//stdsize.width = stdwidth;
	//stdsize.height = stdwidth / facialImage->width * facialImage->height;
	//IplImage *stdImage = cvCreateImage(stdsize, originalImage->depth, originalImage->nChannels);
	//cvResize(facialImage, stdImage, CV_INTER_LINEAR);

	//cvNamedWindow("CurrentFacialImage");
	//cvShowImage("CurrentFacialImage", stdImage);

	////draw the shape
	//CvSize ssize;
	//ssize.width = 130;
	//ssize.height = 130;
	//IplImage *shapeImg = cvCreateImage(ssize, originalImage->depth, originalImage->nChannels);
	//cvSet(shapeImg, CV_RGB(0,0,0));
	//AAM_Shape temShape = Shape;
	//double orgwid = temShape.MaxX() - temShape.MinX();
	//double orghei = temShape.MaxY() - temShape.MinY();
	//for (int i = 0; i < 68; i++) {
	//	temShape[i].x = (temShape[i].x - Shape.MinX()) * stdwidth / orgwid;
	//	temShape[i].y = (temShape[i].y - Shape.MinY()) * stdsize.height / orghei;
	//}
	//temShape.Sketch(shapeImg);
	//cvShowImage("shape", shapeImg);
	//cvReleaseImage(&shapeImg);


	////Facial Prediction
	//FacePredict face_predict;
	//std::string mfile = resultDir + "facial.predict_model";
	//std::ifstream model(mfile);
	//face_predict.Read(model);
	//model.close();
	////predict(const AAM_Shape& shape, const IplImage& curImage, int curAgeG, int newAgeG, bool save)
	//IplImage* newImage = face_predict.predict(Shape, *originalImage, /*ShapeF, *ImageF, ratioF, ShapeM, *ImageM, ratioM,*/ atoi(argv[3]), atoi(argv[4]), true);
	//std::string newfile = std::string(argv[2]);
	//newfile = newfile.insert(newfile.find_last_of('/')+1, "result_" );
	//newfile = newfile.insert(newfile.find_last_of('.'), std::string("_G" + std::string(argv[4])));
	//cvSaveImage(newfile.c_str(), newImage);
	//
	//cvNamedWindow("PredictedFacialImage");
	//cvShowImage("PredictedFacialImage", newImage);
	//cvWaitKey(0);

	//cvReleaseImage(&image);

	cout << faceAging("input.jpg", 2, 3) << endl;
	system("pause");
	return 0;
}
