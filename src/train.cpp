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
#include "FacePredict.h"

using namespace std;

enum{ TYPE_AAM_BASIC = 0, TYPE_AAM_IC = 1};

std::string resultDir = "../trainingSets/";
//std::string resultDir = "../test2/";

//CvMat* AllTextures = cvCreateMat(315, 85000, CV_64FC1);

string int2string(int i) {
	string s;
	char c[10];
	itoa(i,c,10);
	return string(c);
}

static void usage()
{
	printf("train int string string string string...\n"
		"---- int:	0(AAM_Basic), 1(AAM_IC)\n"
		"---- string:	train path that contains images and landmarks\n"
		"---- string:	image extension(e.g. jpg, bmp)\n"
		"---- string:	points extension(e.g. pts, asf)\n");
	exit(0);
}

int main(int argc, char** argv)
{
	if(argc != 5)		 
		usage();
	
	//==================== Read in the images and points data====================
	std::vector<std::string> trainPaths = ScanNSortAllDirectorys(argv[2]);
	std::vector<std::string> m_vimgFiles, m_vptsFiles;
	std::vector<std::string> t_imgFiles, t_ptsFiles;
	int nG_Samples[AGE_AREA] = {0};

	for (int i = 0; i < trainPaths.size(); i++) {
		t_imgFiles = ScanNSortDirectory(trainPaths[i], argv[3]);
		t_ptsFiles = ScanNSortDirectory(trainPaths[i], argv[4]);

		if(t_imgFiles.size() != t_ptsFiles.size())
		{
			fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
				__FILE__, __LINE__);
			exit(0);
		}
		int age_group = getAgeGroup(trainPaths[i]);
		//int age_group = i;
		nG_Samples[age_group] = t_imgFiles.size();
		m_vimgFiles.insert(m_vimgFiles.end(), t_imgFiles.begin(), t_imgFiles.end());
		m_vptsFiles.insert(m_vptsFiles.end(), t_ptsFiles.begin(), t_ptsFiles.end());
	}

	std::vector<AAM_Shape> AllShapes;
	AAM_Shape referenceShape;
	std::vector<IplImage*> AllImages;

	for(int i = 0; i < m_vimgFiles.size(); i++)
	{
		AllImages.push_back(cvLoadImage(m_vimgFiles[i].c_str(), 1));
		referenceShape.ReadPTS(m_vptsFiles[i]);
		AllShapes.push_back(referenceShape);
	}

	//============================== train AAM ===============================
	if (atoi(argv[1])!=0 && atoi(argv[1])!=1)
		printf("Un-Supported AAM type!\n");

	else {
		int group_size = 0;
		std::vector<AAM_Shape> GroupShapes;	
		std::vector<AAM_Shape>::iterator itr_shape = AllShapes.begin();
		std::vector<IplImage*> GroupImages;
		std::vector<IplImage*>::iterator itr_image = AllImages.begin();

		for (int i = 0; i < NGROUPS; i++) {

			//get the samples in a designated age group
			group_size = 0;
			for (int j = AGE_GROUPS[i][0]; j <= AGE_GROUPS[i][1]; j++)
				group_size += nG_Samples[j];

			for (int j = 0; j < group_size; j++, itr_shape++, itr_image++) {
				GroupShapes.push_back(*itr_shape);
				GroupImages.push_back(*itr_image);
			}

			//train Basic AAM
			if(atoi(argv[1])==0) {
				AAM_Basic aam; aam.Train(GroupShapes, GroupImages);

				std::string aamfile = resultDir + "Group" + int2string(i) +".aam_basic";
				std::ofstream fs(aamfile.c_str());
				fs << TYPE_AAM_BASIC << std::endl;
				aam.Write(fs);
				fs.close();
			}

			//train AAM Inverse compositional
			else {
				AAM_IC aam_ic; aam_ic.Train(GroupShapes, GroupImages);
		
				std::string aamfile = resultDir + "Group" + int2string(i) +".aam_ic";
				std::ofstream fs(aamfile.c_str());
				fs << TYPE_AAM_IC << std::endl;
				aam_ic.Write(fs);
				fs.close();
			}
			GroupShapes.clear();
			GroupImages.clear();
		}
	}

	//==========================train Face Predict model===================
	FacePredict face_predict;
	face_predict.Train(AllShapes, AllImages, nG_Samples, /*AllTextures,*/ 0.95, 0.95);

	std::ofstream file(resultDir + "facial.predict_model");

	face_predict.Write(file);
	file.close();

	
	for(int j = 0; j < AllImages.size(); j++) {
		cvReleaseImage(&AllImages[j]);
		AllShapes[j].clear();
	}
	
	return 0;
}