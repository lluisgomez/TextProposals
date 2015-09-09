#include "region.h"
#include "voronoi.h" //voronoi_skeleton by Arnaud Ramey <arnaud.a.ramey@gmail.com>

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

  bbox_ = boundingRect(pixels_);

  if ((!conf[1])&&(!conf[2])&&(!conf[3])&&(!conf[4])) return;

  // Expanded box 5pix.
  Rect bbox = bbox_ + Size(10,10);
  bbox = bbox - Point(5,5);
  bbox = bbox & Rect(0,0,mask.cols,mask.rows);

  uchar* mptr = (uchar*)mask.data;
  for ( int j = 0; j < (int)pixels_.size(); j++ )
  {
    Point pt = pixels_[j];
    mptr[pt.x+pt.y*mask.cols] = 255;
  }

  Scalar m;

  if (conf[1])
  {
    m = mean(_grey_img(bbox_),mask(bbox_));
    mask(bbox_) = Scalar(0); //clean the mask
    intensity_mean_ = m[0];
  }

  Mat tmp_mask;

  if (conf[4])
  {
    Mat dt;
    distanceTransform(mask(bbox_), dt, CV_DIST_L1,3); //L1 gives distance in round integers
    VoronoiSkeleton skel;
    skel.thin(mask(bbox),IMPL_GUO_HALL_FAST,false);
    skel.get_skeleton()(Rect(bbox_.x-bbox.x,bbox_.y-bbox.y,
      bbox_.width,bbox_.height)).copyTo(tmp_mask); // TODO is this efficient?
    m = mean(dt,tmp_mask);
    stroke_mean_ = m[0];
  }

  Mat element = getStructuringElement( MORPH_RECT, Size(5, 5), Point(2, 2) );

  if (conf[2])
  {
    dilate(mask(bbox), tmp_mask, element);
    absdiff(tmp_mask, mask(bbox), tmp_mask);	
	
    m = mean(_grey_img(bbox), tmp_mask);
    boundary_intensity_mean_ = m[0];
  }
	
  if (conf[3])
  {
    Mat tmp2;
    dilate(mask(bbox), tmp_mask, element);
    erode(mask(bbox), tmp2, element);
    absdiff(tmp_mask, tmp2, tmp_mask);	
	
    m = mean(_gradient_magnitude(bbox), tmp_mask);
    gradient_mean_ = m[0];
  }

}

