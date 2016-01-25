#ifndef AGGLOMERATIVE_CLUSTERING_H
#define AGGLOMERATIVE_CLUSTERING_H

#include <vector>

#include <opencv2/ml.hpp>
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
    vector<vector<float> > points; // nD points in this cluster
    int node1;  // child
    int node2;  // child
    cv::Rect rect;  // each group defines a bounding box
    double probability;
    int nfa;	// the number of false alarms for this merge (we are using only the nfa exponent so this is an int)
} HCluster;

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
	void operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector<HCluster> &merge_info);

	/// Helper function
	void build_merge_info(t_float *dendogram, t_float *data, int num, int dim, vector<HCluster> &merge_info);
	

private:
    vector<Region> &regions;
    Ptr<Boost> boost;
    
};

#endif
