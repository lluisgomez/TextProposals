#include "region.h"

#include <algorithm>
#include <cassert>
#include <limits>

using namespace std;
using namespace cv;

Region::Region() 
{
}


void Region::extract_features(Mat& _lab_img, Mat& _grey_img, Mat& _gradient_magnitude, Mat& mask, bool conf[])
{

  //TODO following line is not reallistic at all because the group classifier still use statistics of this features, so either we buils a classifier without them OR we calculate them allways
  if ((!conf[1])&&(!conf[2])&&(!conf[3])&&(!conf[4])) return;

  // Expanded box 5pix.
  Rect bbox = bbox_ + Size(10,10);
  bbox = bbox - Point(5,5);
  bbox = bbox & Rect(0,0,mask.cols,mask.rows);

  uchar* mptr = (uchar*)mask.data;
  for ( int j = 0; j < (int)pixels_.size(); j++ )
  {
    mptr[pixels_[j].x+pixels_[j].y*mask.cols] = 255;
  }

  Scalar m;

  if (conf[1])
  {
    m = mean(_grey_img(bbox_),mask(bbox_));
    intensity_mean_ = m[0];
  }

  Mat tmp_mask, tmp_mask2;
  Mat element = getStructuringElement( MORPH_RECT, Size(5, 5), Point(2, 2) );
  dilate(mask(bbox), tmp_mask, element);

  if (conf[4])
  {
    distanceTransform(tmp_mask, tmp_mask2, CV_DIST_L1,3); //L1 gives distance in round integers
    double mm;
    minMaxLoc(tmp_mask2,NULL,&mm,NULL,NULL,tmp_mask);
    stroke_mean_ = (int)mm;
  }


  if (conf[2])
  {
    absdiff(tmp_mask, mask(bbox), tmp_mask2);	
	
    m = mean(_grey_img(bbox), tmp_mask2);
    boundary_intensity_mean_ = m[0];
  }
	
  if (conf[3])
  {
    erode(mask(bbox), tmp_mask2, element);
    absdiff(tmp_mask, tmp_mask2, tmp_mask);	
	
    m = mean(_gradient_magnitude(bbox), tmp_mask);
    gradient_mean_ = m[0];
  }
  
  mask(bbox_) = Scalar(0); //clean the mask

}

