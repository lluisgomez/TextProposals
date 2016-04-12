#ifndef AGGLOMERATIVE_CLUSTERING_H
#define AGGLOMERATIVE_CLUSTERING_H

#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "region.h"
#include "fast_clustering.cpp"
#include "nfa.cpp"
#include "min_bounding_box.h"

using namespace std;
using namespace cv;
using namespace ml;

typedef struct {
    int num_elem;		 // number of elements
    vector<int> elements;// elements of the cluster (region IDs)
    vector<int> childs;// dendrogram childs of the cluster (clusters IDs)
    vector<float> elements_intensities;
    vector<float> elements_b_intensities;
    vector<int> elements_strokes;
    vector<int> elements_diameters;
    vector<double> elements_gradients;
    vector<Point> points; // nD points in this cluster
    vector<float> feature_vector;
    int node1;  // child
    int node2;  // child
    cv::Rect rect;  // each group defines a bounding box
    double probability;
    float  cnn_probability;
    string cnn_recognition;
    int    nfa;	// the number of false alarms for this merge (we are using only the nfa exponent so this is an int)
    int    inherit_cnn_probability; // when a cluster node has the same bbox as one of its childs there is no need to re-evaluate the CNN classifier
    float  branch_max_probability;
    int    branch_maxIdx;

    void write(FileStorage& fs) const //Write serialization for this class
    {
      fs << "{" << "num_elem" << num_elem;
      fs << "childs" << "[";
      for (size_t i=0; i<childs.size(); i++)
        fs << childs[i];
      fs << "]";
      /*fs << "elements" << "[";
      for (size_t i=0; i<elements.size(); i++)
        fs << elements[i];
      fs << "]";
      fs << "points" << "[";
      for (size_t i=0; i<points.size(); i++)
        fs << points[i];
      fs << "]";*/
      fs << "node1" << node1 << "node2" << node2 << "rect" << rect
         << "probability" << probability << "cnn_probability" << cnn_probability
         << "cnn_recognition" << cnn_recognition << "nfa" << nfa 
         << "inherit_cnn_probability" << inherit_cnn_probability 
         << "}";
    }

    void read(const FileNode& node) //Read serialization for this class
    {
      num_elem = (int)node["num_elem"];
      FileNode n = node["childs"];
      FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
      for (; it != it_end; ++it)
        childs.push_back((int)*it);
      /*FileNode n = node["elements"];
      FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
      for (; it != it_end; ++it)
        elements.push_back((int)*it);
      n = node["points"];
      it = n.begin(), it_end = n.end(); // Go through the node
      for (; it != it_end; ++it)
      {
        Point p;
        *it >> p;
        points.push_back(p);
      }*/
      node1 = (int)node["node1"];
      node2 = (int)node["node2"];
      node["rect"] >> rect;
      probability = (double)node["probability"];
      cnn_probability = (float)node["cnn_probability"];
      cnn_recognition = (string)node["cnn_recognition"];
      nfa = (int)node["nfa"];
      inherit_cnn_probability = (int)node["inherit_cnn_probability"];
    }

} HCluster;



void write(FileStorage& fs, const std::string&, const HCluster& x);
void read(const FileNode& node, HCluster& x, const HCluster& default_value = HCluster());


class HierarchicalClustering
{
public:
	
	/// Constructor.
	HierarchicalClustering(vector<Region> &regions);
	
	/// Does hierarchical clustering
	/// @param[in] data The data feature vectors to be analyzed.
	/// @param[in] Num  Number of data samples.
	/// @param[in] dim  Dimension of the feature vectors.
	/// @param[in] method Clustering method.
	/// @param[in] metric Similarity metric for clustering.
	/// @param[out] merge_info stores the resulting dendrogram information using struct HClust
	void operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector<HCluster> &merge_info, float x_coord_mult, cv::Size imsize);

	/// Helper function
	void build_merge_info(t_float *dendogram, t_float *data, int num, int dim, vector<HCluster> &merge_info, float x_coord_mult, cv::Size imsize);
	

private:
    vector<Region> &regions;
    Ptr<Boost> boost;
    
};

#endif
