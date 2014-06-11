/****************************************************************************
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

using namespace std;

static void usage()
{
	printf("AAMFitting string string\n"
		"---- string:	trained AAM model\n"
		"---- string:	image file\n");
	exit(0);
}

int main(int argc, char** argv)
{
	if(argc != 3) usage();

	AAM * aam;
	int type;

	std::ifstream fs(argv[1]);
	if(fs == 0) {
		fprintf(stderr, "ERROR(%s: %d): Cannot open file %s!\n", __FILE__, __LINE__, argv[1]);
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
	
	//load image
	IplImage* image = cvLoadImage(argv[2], 1);
	if(image==0){
		fprintf(stderr, "ERROR(%s, %d): Cannot open image file %s!\n",
			__FILE__, __LINE__, argv[2]);
		exit(0);
	}
	
	//detect face for intialization
	AAM_Shape Shape = fjdetect.Detect(image, aam->GetMeanShape());

	//do image alignment
	aam->Fit(image, Shape, 30, true);

	ofstream outfile;
	outfile.open( "test.txt" );

	Shape.Write( outfile );

	outfile.close();
	
	//show GUI
	cvNamedWindow("AAMFitting");
	aam->Draw(image, 1);
	cvShowImage("AAMFitting", image);
	cvWaitKey(0);
	
	cvReleaseImage(&image);

	return 0;
}