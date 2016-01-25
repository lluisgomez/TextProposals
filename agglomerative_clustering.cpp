#include "agglomerative_clustering.h"

HierarchicalClustering::HierarchicalClustering(vector<Region> &_regions): regions(_regions)
{
    boost = StatModel::load<Boost>( "./trained_boost_groups.xml" );
    if( boost.empty() )
    {
        cout << "Could not read the classifier ./trained_boost_groups.xml" << endl;
        CV_Error(Error::StsBadArg, "Could not read the default classifier!");
    }
}

//For feature space
void HierarchicalClustering::operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector<HCluster> &merge_info)
{
    
    t_float *Z = (t_float*)malloc(((num-1)*4) * sizeof(t_float)); // we need 4 floats foreach merge
    linkage_vector(data, (int)num, dim, Z, method, metric);
    
    build_merge_info(Z, data, (int)num, dim, merge_info);

    free(Z);
}


void HierarchicalClustering::build_merge_info(t_float *Z, t_float *X, int N, int dim, vector<HCluster> &merge_info)
{

    // walk the whole dendogram
    for (int i=0; i<(N-1)*4; i=i+4)
    {
        HCluster cluster;
        cluster.num_elem = Z[i+3]; //number of elements

        int node1  = Z[i];
        int node2  = Z[i+1];
        float dist = Z[i+2];
    
        if (node1<N) // child node1 is a single region
        {
            cluster.elements.push_back((int)node1);
            cluster.rect = regions.at(node1).bbox_;
            vector<float> point;
	    for (int n=0; n<dim; n++)
              point.push_back(X[node1*dim+n]);
            cluster.points.push_back(point);
        }
        else // child node1 is a cluster
        {
            cluster.points.insert(cluster.points.end(),
                                  merge_info.at(node1-N).points.begin(),
                                  merge_info.at(node1-N).points.end());
            cluster.elements.insert(cluster.elements.end(),
                                    merge_info.at(node1-N).elements.begin(),
                                    merge_info.at(node1-N).elements.end());
            cluster.rect = merge_info.at(node1-N).rect;
        }
        if (node2<N) // child node2 is a single region
        {
            vector<float> point;
	    for (int n=0; n<dim; n++)
              point.push_back(X[node2*dim+n]);
            cluster.points.push_back(point);
            cluster.elements.push_back((int)node2);
            cluster.rect = cluster.rect | regions.at(node2).bbox_; // min. area rect containing node 1 and node2
        }
        else // child node2 is a cluster
        {
            cluster.points.insert(cluster.points.end(),
                                  merge_info.at(node2-N).points.begin(),
                                  merge_info.at(node2-N).points.end());
            cluster.elements.insert(cluster.elements.end(),
                                    merge_info.at(node2-N).elements.begin(),
                                    merge_info.at(node2-N).elements.end());
            cluster.rect = cluster.rect | merge_info.at(node2-N).rect; // min. area rect containing node 1 and node2
        }

    
        cluster.node1 = node1;
        cluster.node2 = node2;

        Minibox mb;
        for (int i=0; i<cluster.points.size(); i++)
          mb.check_in(&cluster.points.at(i));	
        
        long double volume = mb.volume(); 
        if (volume >= 1) volume = 0.999999;
        if (volume == 0) volume = 0.000001; //TODO is this the minimum we can get?
		
        cluster.nfa = -1*(int)NFA( N, cluster.points.size(), (double) volume, 0); //this uses an approximation for the nfa calculations (faster)

        /* predict group class with boost */
        vector<float> sample;
        sample.push_back(0);
	Mat diameters      ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat strokes        ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat gradients      ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat fg_intensities ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat bg_intensities ( cluster.elements.size(), 1, CV_32F, 1 );
	for (int i=cluster.elements.size()-1; i>=0; i--)
	{

	  diameters.at<float>(i,0)      = (float)max(regions.at(cluster.elements.at(i)).bbox_.width,
                                                   regions.at(cluster.elements.at(i)).bbox_.height);
	  strokes.at<float>(i,0)        = (float)regions.at(cluster.elements.at(i)).stroke_mean_;
	  gradients.at<float>(i,0)      = (float)regions.at(cluster.elements.at(i)).gradient_mean_;
	  fg_intensities.at<float>(i,0) = (float)regions.at(cluster.elements.at(i)).intensity_mean_;
	  bg_intensities.at<float>(i,0) = (float)regions.at(cluster.elements.at(i)).boundary_intensity_mean_;
        }

	Scalar mean,std;
	meanStdDev( diameters, mean, std );
	sample.push_back(std[0]/mean[0]);
	meanStdDev( strokes, mean, std );
	sample.push_back(std[0]/mean[0]); 
	meanStdDev( gradients, mean, std );
	sample.push_back(std[0]); 
	meanStdDev( fg_intensities, mean, std );
	sample.push_back(std[0]); 
	meanStdDev( bg_intensities, mean, std );
	sample.push_back(std[0]); 

        float votes_group = boost->predict( Mat(sample), noArray(), DTrees::PREDICT_SUM | StatModel::RAW_OUTPUT);
        cluster.probability = (double)1-(double)1/(1+exp(-2*votes_group));

        merge_info.push_back(cluster);

    }
}
