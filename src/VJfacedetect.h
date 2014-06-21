/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#ifndef VJ_FACE_DETECT_H
#define VJ_FACE_DETECT_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "cv.h"
#include "highgui.h"
#include "AAM_Shape.h"

// Viola and Jones's AdaBoost Haar-like Face Detector
class VJfacedetect  
{
public:
	VJfacedetect();
	~VJfacedetect();
	
	// Detect face for get a good intial shape
	bool DetectFace(std::vector<AAM_Shape> &Shape, const IplImage* image);

	// Load adaboost cascade file for detect face 
	void LoadCascade(const char* cascade_name = "haarcascade_frontalface_alt2.xml");

	// add by wuxuef	
	//detect most central face in  image
	AAM_Shape Detect(const IplImage* image, const AAM_Shape& MeanShape);
private:
	// add by wuxuef
	/**************************************************************************/
	/* The following two functions are borrowed from Stephen Milborrow's stasm*/
	/**************************************************************************/
	// Make the ones face box smaller and move it down a bit.
	void AdjustViolaJonesShape (AAM_Shape &Shape);

	// align MeanShape to the Viola Jones global detector face box
	void AlignToViolaJones(AAM_Shape &StartShape, const AAM_Shape &DetShape, 
		const AAM_Shape& MeanShape);	

private:
	CvMemStorage* __storage;
	CvHaarClassifierCascade* __cascade;

};

#endif  // asmutil_h

