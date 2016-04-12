#ifndef UTILS_H
#define UTILS_H
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

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

/* FOREGROUND COLORS use as follows:
   cout << KRED << "I'm red text." << KRST << endl;
   cout << KBOLD << KBLU << "I'm blue-bold." << KRST << endl; */
#define KRST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define KBOLD "\x1B[1m"
#define KUNDL "\x1B[4m"

#endif
