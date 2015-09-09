#ifndef UTILS_H
#define UTILS_H
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

void accumulate_evidence(vector<int> *meaningful_cluster, int grow, Mat *co_occurrence)
{
	//for (int k=0; k<meaningful_clusters->size(); k++)
	   for (int i=0; i<meaningful_cluster->size(); i++)
	   	for (int j=i; j<meaningful_cluster->size(); j++)
			if (meaningful_cluster->at(i) != meaningful_cluster->at(j))
			{
			    co_occurrence->at<double>(meaningful_cluster->at(i), meaningful_cluster->at(j)) += grow;
			    co_occurrence->at<double>(meaningful_cluster->at(j), meaningful_cluster->at(i)) += grow;
			}
}

void get_gradient_magnitude(Mat& _grey_img, Mat& _gradient_magnitude)
{
	cv::Mat C = cv::Mat_<double>(_grey_img);

	cv::Mat kernel = (cv::Mat_<double>(1,3) << -1,0,1);
	cv::Mat grad_x;
	filter2D(C, grad_x, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

	cv::Mat kernel2 = (cv::Mat_<double>(3,1) << -1,0,1);
	cv::Mat grad_y;
	filter2D(C, grad_y, -1, kernel2, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

    magnitude( grad_x, grad_y, _gradient_magnitude);

}

#endif
