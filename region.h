
#ifndef REGION_H
#define REGION_H

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include <vector>
#include <stdint.h>

/// A Maximally Stable Extremal Region.
class Region
{
public:
	/// Constructor.
	Region();

	std::vector<cv::Point> pixels_;

	/// Extract_features.
	/// @param[in] lab_img L*a*b* color image to extract color information
	/// @param[in] grey_img Grey level version of the original image 
	/// @param[in] gradient_magnitude of the original image
	void extract_features(cv::Mat& _lab_img, cv::Mat& _grey_img, cv::Mat& _gradient_magnitude, cv::Mat& canvas, bool conf[]);

	cv::Rect bbox_;		///< Axis aligned bounding box
        float intensity_mean_;	///< mean intensity of the whole region
        float boundary_intensity_mean_;	///< mean intensity of the boundary of the region
	int   stroke_mean_;	///< mean stroke of the whole region
	double gradient_mean_;	///< mean gradient magnitude of the whole region
};

#endif
