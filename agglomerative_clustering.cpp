#include "agglomerative_clustering.h"


void write(FileStorage& fs, const std::string&, const HCluster& x)
{
x.write(fs);
}

void read(const FileNode& node, HCluster& x, const HCluster& default_value)
{
if(node.empty())
    x = default_value;
else
    x.read(node);
}



HierarchicalClustering::HierarchicalClustering(vector<Region> &_regions): regions(_regions)
{
#ifndef _TRAIN_
    boost = StatModel::load<Boost>( "./trained_boost_groups.xml" );
    if( boost.empty() )
    {
        cout << "Could not read the classifier ./trained_boost_groups.xml" << endl;
        CV_Error(Error::StsBadArg, "Could not read the default classifier!");
    }
#endif
}

//For feature space
void HierarchicalClustering::operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector<HCluster> &merge_info, float x_coord_mult, cv::Size imsize)
{
    
    t_float *Z = (t_float*)malloc(((num-1)*4) * sizeof(t_float)); // we need 4 floats foreach merge
    linkage_vector(data, (int)num, dim, Z, method, metric);
    
    build_merge_info(Z, data, (int)num, dim, merge_info, x_coord_mult, imsize);

    free(Z);
}


void HierarchicalClustering::build_merge_info(t_float *Z, t_float *X, int N, int dim, vector<HCluster> &merge_info, float x_coord_mult, cv::Size imsize)
{

    int this_node = 0;
    // walk the whole dendogram
    for (int i=0; i<(N-1)*4; i=i+4)
    {
        HCluster cluster;
        cluster.childs.push_back(this_node);
        cluster.num_elem = Z[i+3]; //number of elements

        int node1  = Z[i];
        int node2  = Z[i+1];
        float dist = Z[i+2];
    
        if (node1<N) // child node1 is a single region
        {
            cluster.elements.push_back((int)node1);
            cluster.elements_intensities.push_back(regions[node1].intensity_mean_);
            cluster.elements_b_intensities.push_back(regions[node1].boundary_intensity_mean_);
            cluster.elements_strokes.push_back(regions[node1].stroke_mean_);
            cluster.elements_gradients.push_back(regions[node1].gradient_mean_);
            cluster.elements_diameters.push_back(max(regions[node1].bbox_.width,regions[node1].bbox_.height));
            cluster.rect = regions.at(node1).bbox_;
            //vector<float> point;
	    //for (int n=0; n<dim; n++)
            //  point.push_back(X[node1*dim+n]);
            cluster.points.push_back(Point((int)((X[node1*dim]/x_coord_mult)* imsize.width),(int)(X[node1*dim+1]*imsize.height)));
        }
        else // child node1 is a cluster
        {
            cluster.childs.insert(cluster.childs.end(),
                                    merge_info.at(node1-N).childs.begin(),
                                    merge_info.at(node1-N).childs.end());
            cluster.points.insert(cluster.points.end(),
                                  merge_info.at(node1-N).points.begin(),
                                  merge_info.at(node1-N).points.end());
            cluster.elements.insert(cluster.elements.end(),
                                    merge_info.at(node1-N).elements.begin(),
                                    merge_info.at(node1-N).elements.end());
            cluster.elements_intensities.insert(cluster.elements_intensities.end(),
                                    merge_info.at(node1-N).elements_intensities.begin(),
                                    merge_info.at(node1-N).elements_intensities.end());
            cluster.elements_b_intensities.insert(cluster.elements_b_intensities.end(),
                                    merge_info.at(node1-N).elements_b_intensities.begin(),
                                    merge_info.at(node1-N).elements_b_intensities.end());
            cluster.elements_strokes.insert(cluster.elements_strokes.end(),
                                    merge_info.at(node1-N).elements_strokes.begin(),
                                    merge_info.at(node1-N).elements_strokes.end());
            cluster.elements_gradients.insert(cluster.elements_gradients.end(),
                                    merge_info.at(node1-N).elements_gradients.begin(),
                                    merge_info.at(node1-N).elements_gradients.end());
            cluster.elements_diameters.insert(cluster.elements_diameters.end(),
                                    merge_info.at(node1-N).elements_diameters.begin(),
                                    merge_info.at(node1-N).elements_diameters.end());
            cluster.rect = merge_info.at(node1-N).rect;
        }
        if (node2<N) // child node2 is a single region
        {
            //vector<float> point;
	    //for (int n=0; n<dim; n++)
            //  point.push_back(X[node2*dim+n]);
            //cluster.points.push_back(point);
            cluster.points.push_back(Point((int)((X[node2*dim]/x_coord_mult)* imsize.width),(int)(X[node2*dim+1]*imsize.height)));
            cluster.elements.push_back((int)node2);
            cluster.elements_intensities.push_back(regions[node2].intensity_mean_);
            cluster.elements_b_intensities.push_back(regions[node2].boundary_intensity_mean_);
            cluster.elements_strokes.push_back(regions[node2].stroke_mean_);
            cluster.elements_gradients.push_back(regions[node2].gradient_mean_);
            cluster.elements_diameters.push_back(max(regions[node2].bbox_.width,regions[node2].bbox_.height));
            cluster.rect = cluster.rect | regions.at(node2).bbox_; // min. area rect containing node 1 and node2
        }
        else // child node2 is a cluster
        {
            cluster.childs.insert(cluster.childs.end(),
                                    merge_info.at(node2-N).childs.begin(),
                                    merge_info.at(node2-N).childs.end());
            cluster.points.insert(cluster.points.end(),
                                  merge_info.at(node2-N).points.begin(),
                                  merge_info.at(node2-N).points.end());
            cluster.elements.insert(cluster.elements.end(),
                                    merge_info.at(node2-N).elements.begin(),
                                    merge_info.at(node2-N).elements.end());
            cluster.elements_intensities.insert(cluster.elements_intensities.end(),
                                    merge_info.at(node2-N).elements_intensities.begin(),
                                    merge_info.at(node2-N).elements_intensities.end());
            cluster.elements_b_intensities.insert(cluster.elements_b_intensities.end(),
                                    merge_info.at(node2-N).elements_b_intensities.begin(),
                                    merge_info.at(node2-N).elements_b_intensities.end());
            cluster.elements_strokes.insert(cluster.elements_strokes.end(),
                                    merge_info.at(node2-N).elements_strokes.begin(),
                                    merge_info.at(node2-N).elements_strokes.end());
            cluster.elements_gradients.insert(cluster.elements_gradients.end(),
                                    merge_info.at(node2-N).elements_gradients.begin(),
                                    merge_info.at(node2-N).elements_gradients.end());
            cluster.elements_diameters.insert(cluster.elements_diameters.end(),
                                    merge_info.at(node2-N).elements_diameters.begin(),
                                    merge_info.at(node2-N).elements_diameters.end());
            cluster.rect = cluster.rect | merge_info.at(node2-N).rect; // min. area rect containing node 1 and node2
        }

        if ( ((node1<N) && (cluster.rect == regions.at(node1).bbox_)) ||
             ((node1>N) && (cluster.rect == merge_info.at(node1-N).rect)) )
          cluster.inherit_cnn_probability = 1;
        else if ( ((node2<N) && (cluster.rect == regions.at(node2).bbox_)) ||
                  ((node2>N) && (cluster.rect == merge_info.at(node2-N).rect)) )
          cluster.inherit_cnn_probability = 2;
        else
          cluster.inherit_cnn_probability = 0;
    
        cluster.node1 = node1-N;
        cluster.node2 = node2-N;

        //Minibox mb;
        //for (int i=0; i<cluster.points.size(); i++)
        //  mb.check_in(&cluster.points.at(i));	        
        //long double volume = mb.volume();

        Rect centers_rect = boundingRect(cluster.points);
        long double volume = (long double)centers_rect.area()/(imsize.area());
        long double ext_volume = (long double)cluster.rect.area()/(imsize.area());
        if (volume >= 1) volume = 0.999999; //TODO this may never happen!!!
        if (volume == 0) volume = 0.000001; //TODO is this the minimum we can get? // better if we just quantize to a given grid of possible volumes ... 
		
        cluster.nfa = (int)NFA( N, cluster.points.size(), (double) volume, 0); //this uses an approximation for the nfa calculations (faster)

        /* predict group class with boost */
        int nfa2 = (int)NFA( N, cluster.points.size(), (double) ext_volume, 0); //this uses an approximation for the nfa calculations (faster)
        cluster.feature_vector.push_back(cluster.elements.size());

        cluster.feature_vector.push_back(cluster.nfa);
        cluster.feature_vector.push_back(nfa2);
        if (cluster.nfa == 0)
         cluster.feature_vector.push_back(-1.0);
        else
         cluster.feature_vector.push_back((float)nfa2/cluster.nfa);

        cluster.feature_vector.push_back((float)(volume/ext_volume));

        cluster.feature_vector.push_back((float)centers_rect.width/cluster.rect.width);
        int left_diff = centers_rect.x - cluster.rect.x;
        int right_diff = (cluster.rect.x+cluster.rect.width) - (centers_rect.x+centers_rect.width);
        cluster.feature_vector.push_back((float)left_diff/cluster.rect.width);
        cluster.feature_vector.push_back((float)right_diff/cluster.rect.width);
        if (max(left_diff,right_diff) == 0)
          cluster.feature_vector.push_back(1.0);
        else
          cluster.feature_vector.push_back((float)min(left_diff,right_diff)/max(left_diff,right_diff));

        cluster.feature_vector.push_back((float)centers_rect.height/cluster.rect.height);
        int top_diff = centers_rect.y - cluster.rect.y;
        int bottom_diff = (cluster.rect.y+cluster.rect.height) - (centers_rect.y+centers_rect.height);
        cluster.feature_vector.push_back((float)top_diff/cluster.rect.height);
        cluster.feature_vector.push_back((float)bottom_diff/cluster.rect.height);
        if (max(top_diff,bottom_diff) == 0)
          cluster.feature_vector.push_back(1.0);
        else
        cluster.feature_vector.push_back((float)min(top_diff,bottom_diff)/max(top_diff,bottom_diff));

	Scalar mean,std;
	meanStdDev( cluster.elements_diameters, mean, std );
	cluster.feature_vector.push_back(std[0]/mean[0]);
	meanStdDev( cluster.elements_strokes, mean, std );
	cluster.feature_vector.push_back(std[0]/mean[0]); 
	meanStdDev( cluster.elements_gradients, mean, std );
	cluster.feature_vector.push_back(std[0]); 
	meanStdDev( cluster.elements_intensities, mean, std );
	cluster.feature_vector.push_back(std[0]); 
	meanStdDev( cluster.elements_b_intensities, mean, std );
	cluster.feature_vector.push_back(std[0]); 

#ifndef _TRAIN_
        float votes_group = boost->predict( Mat(cluster.feature_vector), noArray(), DTrees::PREDICT_SUM | StatModel::RAW_OUTPUT);
        cluster.probability = (double)1-(double)1/(1+exp(-2*votes_group));
#endif

        merge_info.push_back(cluster);
        this_node++;

    }
}
