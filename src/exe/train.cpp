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
#include "AAM_Basic.h"

using namespace std;

enum{ TYPE_AAM_BASIC = 0, TYPE_AAM_IC = 1};

static void usage()
{
	printf("aambuilding int string string string string\n"
		"---- int:	0(AAM_Basic), 1(AAM_IC)\n"
		"---- string:	train path that contains images and annotations\n"
		"---- string:	image extension(e.g. jpg, bmp)\n"
		"---- string:	points extension(e.g. pts, asf)\n"
		"---- string:	output model of AAM\n");
	exit(0);
}

int main(int argc, char** argv)
{
	if(argc != 6)		usage();
	
    string trainPath = argv[2];
    string IMMmodelData = "model/model-IMM.asm";
	std::vector<std::string> m_vimgFiles = ScanNSortDirectory(trainPath, argv[3]);
	std::vector<std::string> m_vptsFiles = ScanNSortDirectory(trainPath, argv[4]);

	if(m_vptsFiles.size() != m_vimgFiles.size())
	{
		fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
			__FILE__, __LINE__);
		exit(0);
	}

	std::vector<AAM_Shape> AllShapes;
	AAM_Shape referenceShape;
	std::vector<IplImage*> AllImages;

	if(strstr(m_vptsFiles[0].c_str(), "pts"))
	{
		for(int i = 0; i < m_vimgFiles.size(); i++)
		{
			AllImages.push_back(cvLoadImage(m_vimgFiles[i].c_str(), 1));
			referenceShape.ReadPTS(m_vptsFiles[i]);
			AllShapes.push_back(referenceShape);
		}
	}

	else if(strstr(m_vptsFiles[0].c_str(), "asf"))
	{
		for(int i = 0; i < m_vimgFiles.size(); i++)
		{
			AllImages.push_back(cvLoadImage(m_vimgFiles[i].c_str(), 1));
			referenceShape.ReadASF(m_vptsFiles[i]);
			referenceShape.ScaleXY(AllImages[i]->width, AllImages[i]->height);
			AllShapes.push_back(referenceShape);
		}
	}
	
	//train Basic AAM
	if(atoi(argv[1])==0)
	{
		//build model
		AAM_Basic aam; aam.Train(AllShapes, AllImages);

		//write model to file
		std::string aamfile = string(argv[5])+".aam_basic";
		std::ofstream fs(aamfile.c_str());
		fs << TYPE_AAM_BASIC << std::endl;
		aam.Write(fs);
		fs.close();
	}

	//train AAM Inverse compositional
	if(atoi(argv[1])==1)
	{
		//build model
		AAM_IC aam_ic; aam_ic.Train(AllShapes, AllImages);
		
		//write model to file
		std::string aamfile = string(argv[5])+".aam_ic";
		std::ofstream fs(aamfile.c_str());
		fs << TYPE_AAM_IC << std::endl;
		aam_ic.Write(fs);
		fs.close();
	}

	else printf("Un-Supported AAM type!\n");
	
	for(int j = 0; j < AllImages.size(); j++)
	{
		cvReleaseImage(&AllImages[j]);
		AllShapes[j].clear();
	}

	return 0;
}