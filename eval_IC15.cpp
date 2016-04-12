#define _MAIN

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <tinyxml.h>

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"

#include "region.h"
#include "agglomerative_clustering.h"
#include "stopping_rule.h"
#include "utils.h"

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

#define VDEBUG 0 // visual debug for manual inspection of intermediate results

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

bool nmsHClusterSort (HCluster i,HCluster j) { return (i.rect.area()>j.rect.area()); }

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& label_file,
             int batch_size = 100);

  std::vector<Prediction> Classify(const vector<cv::Mat> &images);
  cv::Size getInputSize();

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int batch_size_;
  std::vector<string> labels_;
};
  
cv::Size Classifier::getInputSize() {return input_geometry_;}
  
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file,
                       int batch_size) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 1)
    << "Input layer should have 1 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  batch_size_ = batch_size;
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}


/* Return the top predictions. */
std::vector<Prediction> Classifier::Classify(const vector<cv::Mat> &images) {

  Blob<float>* input_layer = net_->input_blobs()[0];

  if (images.size() != batch_size_)
  {
    batch_size_ = images.size();
    input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
  }

  std::vector<cv::Mat> input_channels;
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < batch_size_; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    images[i].convertTo(channel, CV_32FC1);
    input_channels.push_back(channel);
    input_data += width * height;
  }

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* output_data = output_layer->cpu_data();
  cv::Mat output(batch_size_, output_layer->channels(), CV_32FC1, (void*)output_data);

  std::vector<Prediction> predictions;
  for (int i = 0; i < batch_size_; ++i) {
      Point maxIdx;
      double maxVal;
      minMaxLoc(output.row(i), NULL, &maxVal, NULL, &maxIdx);
      predictions.push_back(std::make_pair(labels_[maxIdx.x], (float)maxVal));
  }
  
  return predictions;
}


/* Diversivication Configurations :                                     */
/* These are boolean values, indicating whenever to use a particular    */
/*                                   diversification strategy or not    */

#define PYRAMIDS     1 // Use spatial pyramids
#define CUE_D        1 // Use Diameter grouping cue
#define CUE_FGI      1 // Use ForeGround Intensity grouping cue
#define CUE_BGI      1 // Use BackGround Intensity grouping cue
#define CUE_G        1 // Use Gradient magnitude grouping cue
#define CUE_S        1 // Use Stroke width grouping cue
#define CHANNEL_I    0 // Use Intensity color channel
#define CHANNEL_R    1 // Use Red color channel
#define CHANNEL_G    1 // Use Green color channel
#define CHANNEL_B    1 // Use Blue color channel


static inline bool is_not_alnum(char c)
{
    return !(isalnum(c));
}


int main( int argc, char** argv )
{


   // Global Stats
   int correct = 0; //true positives
   int misses  = 0; //false negatives
   int err     = 0; //false prositives

    ofstream outfile("RES_evalAll_cnn.txt", ofstream::out | ofstream::app);

    vector<string> full_lex; //global lexicon
    vector<string> empty_lex; //empty lexicon for the generic recognition case
    vector<string> lex50;

    ifstream labels("lex/ICDAR2015/ch4_test_vocabulary.txt");
    string line;
    while (getline(labels, line))
    {
      std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      full_lex.push_back(line);
    }


  // Params
  float x_coord_mult              = 0.25; // a value of 1 means rotation invariant
  float weak_classifier_threshold = 0.4;
  float cnn_classifier_threshold  = 0.95;
  int   min_word_lenght           = 3;
  float nms_IoU_threshold         = 0.2;
  float nms_I_threshold           = 0.5;
  int   lexicon = atoi(argv[1]);

  switch (lexicon)
  {
	case 0:
	case 1:
	  weak_classifier_threshold = 0.01;
	  cnn_classifier_threshold  = 0.85;
	  break;
  }

  // TextProposals Pipeline configuration
  bool conf_channels[4]={CHANNEL_R,CHANNEL_G,CHANNEL_B,CHANNEL_I};
  bool conf_cues[5]={CUE_D,CUE_FGI,CUE_BGI,CUE_G,CUE_S};

  /* initialize random seed: */
  srand (time(NULL));

  double t_cnn_load = (double)getTickCount();

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = string("dictnet_vgg_deploy.prototxt");
  string trained_file = string("dictnet_vgg.caffemodel");
  string label_file   = string("lex.txt");
  int batch_size = 128;
  Classifier classifier(model_file, trained_file, label_file, batch_size);

  t_cnn_load = ((double)getTickCount() - t_cnn_load) / getTickFrequency();






    for (size_t im_id=1; im_id < 501; im_id++) // foreach image
    {
    
        stringstream GTimagename_s;
        GTimagename_s << "img_" << im_id;
        string GTimagename = GTimagename_s.str()+string(".jpg");
        string in_imagename = string("data/ICDAR2015/test/")+GTimagename;
        cout << in_imagename << endl;
        outfile << in_imagename << endl;
        string  gt_fname = string("data/ICDAR2015/test/gt_")+GTimagename_s.str()+string(".txt");
        string lex_fname = string("lex/ICDAR2015/voc_")+GTimagename_s.str()+string(".txt");
        ifstream labels(lex_fname.c_str());
        string line;
        lex50.clear();
        while (getline(labels, line))
        {
          std::transform(line.begin(), line.end(), line.begin(), ::tolower);
          lex50.push_back(line);
        }
        
        vector<Rect> gt_rects;
        vector<string> gt_words;
        vector<Rect> dc_rects; // do not care
        vector<string> dc_words; // do not care

        ifstream gt(gt_fname.c_str());
        while (getline(gt, line))
        {
          std::stringstream  lineStream(line);
          std::string        cell;
          int rect_coords[8];
          int count = 0;
          while(std::getline(lineStream,cell,','))
          {
            if (count < 8)
              rect_coords[count] = atoi(cell.c_str());
            if (count == 8)
            {
              Rect box = Rect(Point(rect_coords[0],rect_coords[1]),Size(rect_coords[4]-rect_coords[0],rect_coords[5]-rect_coords[1]));
              string tag_s = cell;

                  std::transform(tag_s.begin(), tag_s.end(), tag_s.begin(), ::tolower);
                  //if ((tag_s.size() >= 3) && (find_if(tag_s.begin(), tag_s.end(), is_not_alnum) == tag_s.end()))
                  if (find (full_lex.begin(), full_lex.end(), tag_s) != full_lex.end())
                  {
                    gt_words.push_back(tag_s);
                    gt_rects.push_back(box);
                  } else {
                    dc_words.push_back(tag_s);
                    dc_rects.push_back(box);
                  }
            }
            count++;
          }
        }

        // Local Stats (x image)
        int im_correct = 0; //true positives
        int im_misses  = 0; //false negatives
        int im_err     = 0; //false prositives


///// BEGIN loopable code (x image)

  // Stats
  int nodes_total = 0;
  int nodes_evaluated = 0;
  int nodes_rejected_by_inheritance = 0;
  int nodes_rejected_by_weak_classifier = 0;
  int nodes_rejected_by_hash = 0;
  double t_algorithm_0 = (double)getTickCount();
  double t_mser        = 0;
  double t_feat        = 0;
  double t_clustering  = 0;
  double t_cnn         = 0;
  double t_sr          = 0;
  double t_nms         = 0;

  // Hash table for proposals probabilities by their bbox keys
  std::map<string, Prediction> pmap;

  vector<HCluster> max_clusters;

    Mat src, src_vis, src_grey, img, grey, lab_img, gradient_magnitude;

    img = imread(in_imagename);
    img.copyTo(src);
    if (VDEBUG)
      img.copyTo(src_vis);

    int delta = 13;
    int img_area = img.cols*img.rows;
    Ptr<MSER> cv_mser = MSER::create(delta,(int)(0.00002*img_area),(int)(0.11*img_area),55,0.);

    cvtColor(img, grey, CV_BGR2GRAY);
    grey.copyTo(src_grey);
    cvtColor(img, lab_img, CV_BGR2Lab);
    gradient_magnitude = Mat_<double>(img.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    vector<Mat> channels;
    split(img, channels);
    channels.push_back(grey);
    int num_channels = channels.size();

    if (PYRAMIDS)
    {
      for (int c=0; c<num_channels; c++)
      {
        Mat pyr;
        resize(channels[c],pyr,Size(channels[c].cols/2,channels[c].rows/2));
        channels.push_back(pyr);
      }
      /*for (int c=0; c<num_channels; c++)
      {
        Mat pyr;
        resize(channels[c],pyr,Size(channels[c].cols/4,channels[c].rows/4));
        channels.push_back(pyr);
      }*/
    }

    cout << "Go!" << endl;

    for (int c=0; c<channels.size(); c++)
    {

        if (!conf_channels[c%4]) continue;

        if (channels[c].size() != grey.size()) // update sizes for smaller pyramid lvls
        {
          resize(grey,grey,Size(channels[c].cols,channels[c].rows));
          resize(lab_img,lab_img,Size(channels[c].cols,channels[c].rows));
          resize(gradient_magnitude,gradient_magnitude,Size(channels[c].cols,channels[c].rows));
        }

        // TODO you want to try single pass MSER?
        //channels[c] = 255 - channels[c];
        //cv_mser->setPass2Only(true);

        /* Initial over-segmentation using MSER algorithm */
        vector<vector<Point> > contours;
        vector<Rect>  mser_bboxes;
        double t_mser0 = (double)getTickCount();
        cv_mser->detectRegions(channels[c], contours, mser_bboxes);
        //cout << " OpenCV MSER found " << contours.size() << " regions in " << ((double)getTickCount() - t_mser0)/getTickFrequency() << " s." << endl;
        t_mser += ((double)getTickCount() - t_mser0) / getTickFrequency();
   

        /* Extract simple features for each region */ 
        double t_feat0 = (double)getTickCount();
        vector<Region> regions;
        Mat mask = Mat::zeros(grey.size(), CV_8UC1);
        int max_stroke = 0;
        for (int i=contours.size()-1; i>=0; i--)
        {
            Region region;
            region.pixels_.push_back(Point(0,0)); //cannot swap an empty vector
            region.pixels_.swap(contours[i]);
            region.bbox_ = mser_bboxes[i];
            region.extract_features(lab_img, grey, gradient_magnitude, mask, conf_cues);
            max_stroke = max(max_stroke, region.stroke_mean_);
            regions.push_back(region);
        }
        t_feat += ((double)getTickCount() - t_feat0) / getTickFrequency();
          
        unsigned int N = regions.size();
        if (N<3) continue;
        int dim = 3;
        t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));

        /* Single Linkage Clustering for each individual cue */
        for (int cue=0; cue<5; cue++)
        {

          if (!conf_cues[cue]) continue;
    
          int count = 0;
          for (int i=0; i<regions.size(); i++)
          {
            data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/channels[c].cols*x_coord_mult;
            data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/channels[c].rows;
            switch(cue)
            {
              case 0:
                data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(channels[c].rows,channels[c].cols);
                break;
              case 1:
                data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
                break;
              case 2:
                data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;
                break;
              case 3:
                data[count+2] = (t_float)regions.at(i).gradient_mean_/255;
                break;
              case 4:
                data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;
                break;
            }
            count = count+dim;
          }
      
          double t_clustering0 = (double)getTickCount();
          HierarchicalClustering h_clustering(regions);
          vector<HCluster> dendrogram;
          h_clustering(data, N, dim, (unsigned char)0, (unsigned char)3, dendrogram, x_coord_mult, channels[c].size());
          nodes_total += dendrogram.size();
          t_clustering += ((double)getTickCount() - t_clustering0) / getTickFrequency();

          int ml = 1; // a multiplier to update regions sizes for smaller pyramid lvls
          if (c>=num_channels) ml=2;
          if (c>=2*num_channels) ml=4;

          //CNN evaluation of proposals
          double t_cnn0 = (double)getTickCount();
          int node_idx=0;
          while (node_idx < dendrogram.size())
          {
             vector<Mat> batch;
             vector<int> batch_node_indexes;
             while ((batch.size() < batch_size) && (node_idx < dendrogram.size()))
             {
               if (dendrogram[node_idx].inherit_cnn_probability > 0) 
               {
                 nodes_rejected_by_inheritance++;
                 node_idx++;
                 continue; 
               }
               if (dendrogram[node_idx].probability < weak_classifier_threshold) 
               {
                 dendrogram[node_idx].cnn_probability = 0;
                 nodes_rejected_by_weak_classifier++;
                 node_idx++;
                 continue;
               }

               Rect proposal_roi = Rect(dendrogram[node_idx].rect.x*ml,
                                      dendrogram[node_idx].rect.y*ml,
                                      dendrogram[node_idx].rect.width*ml,
                                      dendrogram[node_idx].rect.height*ml);
               proposal_roi = (proposal_roi + Size(10,10)) - Point(5,5); // add a bit of space in the border 
               proposal_roi = proposal_roi & Rect(0,0,src_grey.cols,src_grey.rows);
               dendrogram[node_idx].rect = proposal_roi;

               // check if we already have a result for this bounding box in the hash table
               stringstream sstr_key;
               sstr_key << proposal_roi.x << "x" << proposal_roi.y << "x" 
                        << proposal_roi.width << "x" << proposal_roi.height;
               if (pmap.count(sstr_key.str()) > 0)
               {
                 // TODO maintain a list of bbox correspondences between dendrograms so we can build the heterarchy
                 Prediction p = pmap[sstr_key.str()];
                 dendrogram[node_idx].cnn_probability = p.second;
                 dendrogram[node_idx].cnn_recognition = p.first;
                 nodes_rejected_by_hash++;
                 node_idx++;
                 continue;
               }

               /* we apply here the holistic word recognition CNN to each proposal */
               Mat proposal;
               resize(src_grey(proposal_roi),proposal,classifier.getInputSize());

               // image normalization as in Jaderberg etal.
	       Scalar mean,std;
               proposal.convertTo(proposal, CV_32FC1);
	       meanStdDev( proposal, mean, std );
               proposal = (proposal - mean[0]) / ((std[0] + 0.0001) /128);

               batch.push_back(proposal);
               batch_node_indexes.push_back(node_idx);

               node_idx++;
             }
             if (batch.empty()) break;

             nodes_evaluated += batch.size();

             //cout << " Resize and normalization time (batch) " << (getTickCount() - e1)/ getTickFrequency() << endl;
             //e1 = getTickCount();
             std::vector<Prediction> predictions = classifier.Classify(batch);
             //cout << " CNN time (batch) " << (getTickCount() - e1)/ getTickFrequency() << endl;

             for (int ib=0; ib<predictions.size(); ib++)
             {
                Prediction p = predictions[ib];
                dendrogram[batch_node_indexes[ib]].cnn_probability = p.second;
                dendrogram[batch_node_indexes[ib]].cnn_recognition = p.first;

                Rect proposal_roi = dendrogram[batch_node_indexes[ib]].rect;
                // TODO when in pyr lvls the bbox may be not exactly the same but we must be able to find it
                //      a possible solution would be to insert the same value with a range of different keys
                //      that span along the near vicinity of the bbox, e.g. +/- 5 pixels
                stringstream sstr_key;
                sstr_key << proposal_roi.x << "x" << proposal_roi.y << "x" 
                        << proposal_roi.width << "x" << proposal_roi.height;
                pmap[sstr_key.str()] = p;

                //visualize only the best recognitions
                if ((p.second > cnn_classifier_threshold) && (p.first.size() >= min_word_lenght))
                {
                    //rectangle(src, proposal_roi.tl(), proposal_roi.br(), Scalar(0,0,255));
                    cout << "(" << batch_node_indexes[ib] << ") " << proposal_roi
                         << std::fixed << std::setprecision(4) 
                         << dendrogram[batch_node_indexes[ib]].probability << " "
                         << p.second << " " 
                         << dendrogram[batch_node_indexes[ib]].nfa << " " 
                         << p.first << endl;
                    //imshow("",src_grey(proposal_roi));
                    //waitKey(-1);
                }
                //visualize good proposals
                if (VDEBUG)
                {
                  for (size_t vj=0; vj<gt_words.size(); vj++)
                  {
                    float I_area = (float)(proposal_roi & gt_rects[vj]).area();
                    float U_area = (float)(proposal_roi.area() + gt_rects[vj].area() - I_area);
                    float IoU = I_area / U_area;
                    if (IoU >=0.5)
                    {
                      rectangle(src_vis, proposal_roi.tl(), proposal_roi.br(), Scalar(0,255,255));
                      cout << "(" << batch_node_indexes[ib] << ") " << proposal_roi
                           << std::fixed << std::setprecision(4) 
                           << dendrogram[batch_node_indexes[ib]].probability << " "
                           << p.second << " " 
                           << dendrogram[batch_node_indexes[ib]].nfa << " " 
                           << p.first << endl;
                      break;
                    } 
                  }
                }
             }

          } // end while to eval all nodes in one dendrogram
          t_cnn += ((double)getTickCount() - t_cnn0) / getTickFrequency();

          // here apply a miximallity criteria to the dendrogram

          double t_sr0 = (double)getTickCount();
          StoppingRule sr;
          vector<int> maxIdxs;
          switch (lexicon)
          {
	    case 0:
              sr( dendrogram, maxIdxs, lex50, min_word_lenght, weak_classifier_threshold, cnn_classifier_threshold, true );
	      break;
	    case 1:
              sr( dendrogram, maxIdxs, full_lex, min_word_lenght, weak_classifier_threshold, cnn_classifier_threshold, true );
	      break;
	    default:
              sr( dendrogram, maxIdxs, empty_lex, min_word_lenght, weak_classifier_threshold, cnn_classifier_threshold, true );
          }
          t_sr += ((double)getTickCount() - t_sr0)/ getTickFrequency();


          //accumulate Max clusters in the global list max_clusters for nms ... Optionally visualize
          for (size_t ib=0; ib<maxIdxs.size(); ib++)
          {
                if (  (dendrogram[maxIdxs[ib]].cnn_probability > cnn_classifier_threshold) 
                   && (dendrogram[maxIdxs[ib]].cnn_recognition.size() >= min_word_lenght) )
                {
                    max_clusters.push_back(dendrogram[maxIdxs[ib]]);
                    Rect proposal_roi = dendrogram[maxIdxs[ib]].rect;
                    //rectangle(src, proposal_roi.tl(), proposal_roi.br(), Scalar(0,255,0));
                    cout << KBOLD << KRED << "    MAX (" << maxIdxs[ib] << ") " 
                         << proposal_roi 
                         << std::fixed << std::setprecision(4) << " "
                         << dendrogram[maxIdxs[ib]].cnn_probability << " " 
                         << dendrogram[maxIdxs[ib]].nfa << " " 
                         << dendrogram[maxIdxs[ib]].cnn_recognition << KRST << endl;
                    // TODO NOTICE that small groups still may be selected as maximal ... but these have usually less a much smaller number of elements.
                    //cout << Mat(dendrogram[maxIdxs[ib]].childs).t() << endl;
                    //imshow("",src_grey(proposal_roi));
                    //waitKey(-1);
                }
           }
          
        } // end for each similarity cue
        free(data);

    } // end for each channel


  // TODO here do non-maximal suppression of detections in the different dendrograms

  double t_nms0 = (double)getTickCount();
  std::sort (max_clusters.begin(), max_clusters.end(), nmsHClusterSort);
  for (size_t i=0; i<max_clusters.size(); i++)
  {
    for (size_t j=max_clusters.size()-1; j>i; j--)
    {
      float I_area = (float)(max_clusters[i].rect & max_clusters[j].rect).area();
      // case :: small boxes with low probability inside bigger boxes with better recognition
      // TODO here in img_200 you priorize "courthouse" ove ("court" and "house") !!
      // case :: small boxes with large intersection with bigger boxes with better recognition
      if ( (I_area > nms_I_threshold * max_clusters[j].rect.area()) &&
           (max_clusters[i].cnn_probability >= max_clusters[j].cnn_probability) )
      {
        max_clusters.erase(max_clusters.begin()+j);
        continue;
      }
      float U_area = (float)(max_clusters[i].rect.area() + max_clusters[j].rect.area() - I_area);
      float IoU = I_area / U_area;
      // case :: boxes with very large overlapping and same recognition string
      if (IoU > nms_IoU_threshold)
      {
        if (max_clusters[i].cnn_probability >= max_clusters[j].cnn_probability)
        {
          max_clusters.erase(max_clusters.begin()+j);
          continue;
        } 
        else
        {
          max_clusters.erase(max_clusters.begin()+i);
          i--;
          break;
        }
      }
    }
  }
  t_nms += ((double)getTickCount() - t_nms0)/ getTickFrequency();


  for (size_t i=0; i<max_clusters.size(); i++)
  {
    Rect proposal_roi = max_clusters[i].rect;
    //rectangle(src, proposal_roi.tl(), proposal_roi.br(), Scalar(0,255,0), 2);
    cout << KBOLD << KGRN << "    FINAL (" 
         << proposal_roi 
         << std::fixed << std::setprecision(4) << " "
         << max_clusters[i].cnn_probability << " " 
         << max_clusters[i].nfa << " " 
         << max_clusters[i].cnn_recognition << KRST << endl;
  }


  cout << " Total Nodes         " << nodes_total << endl;
  cout << " Nodes evaluated     " << nodes_evaluated << endl;
  cout << " Nodes inherited     " << nodes_rejected_by_inheritance << endl;
  cout << " Nodes filtered      " << nodes_rejected_by_weak_classifier << endl;
  cout << " Nodes hashed        " << nodes_rejected_by_hash << endl << endl;
  cout << " Time loading model      " << t_cnn_load << " s." << endl;
  cout << " Time full algorithm     " << ((double)getTickCount()-t_algorithm_0)/ getTickFrequency() << " s." << endl;
  cout << "      time mser          " << t_mser << " s." << endl;
  cout << "      time reg feat      " << t_feat << " s." << endl;
  cout << "      time clustering    " << t_clustering << " s." << endl;
  cout << "      time cnn           " << t_cnn  << " s." << endl;
  cout << "      time sr            " << t_sr   << " s." << endl;
  cout << "      time nms           " << t_nms  << " s." << endl;


///// END   loopable code (x image)


  string out_fname = string("./res/res_")+GTimagename_s.str()+string(".txt");
  ofstream im_outfile(out_fname.c_str(), ofstream::out | ofstream::trunc);
  
  for (size_t i=0; i<max_clusters.size(); i++)
  {
    Rect proposal_roi = max_clusters[i].rect;
    string recognition = max_clusters[i].cnn_recognition;
    std::transform(recognition.begin(), recognition.end(), recognition.begin(), ::toupper);
    im_outfile  << proposal_roi.x << "," << proposal_roi.y << ","
         << proposal_roi.x + proposal_roi.width-1 << "," << proposal_roi.y << ","
         << proposal_roi.x + proposal_roi.width-1 << "," << proposal_roi.y + proposal_roi.height-1 << ","
         << proposal_roi.x << "," << proposal_roi.y + proposal_roi.height-1 << ","
         << recognition << endl;
  }

  im_outfile.close();

  cout << "results saved in file " << out_fname << endl;



/*
        for (size_t i=0; i<max_clusters.size(); i++)
        {
          bool matched = false;
          for (size_t j=0; j<gt_words.size(); j++)
          {
            if (max_clusters[i].cnn_recognition != gt_words[j]) continue;
            float I_area = (float)(max_clusters[i].rect & gt_rects[j]).area();
            float U_area = (float)(max_clusters[i].rect.area() + gt_rects[j].area() - I_area);
            float IoU = I_area / U_area;
            if (IoU >=0.5)
            {
              im_correct++;
              gt_words.erase(gt_words.begin()+j);
              gt_rects.erase(gt_rects.begin()+j);
              matched = true;
              if (VDEBUG)
                rectangle(src, max_clusters[i].rect.tl(), max_clusters[i].rect.br(),
                          Scalar(0,255,0), 2);
              break;
            } 
          }
          if (!matched)
          { 
            for (size_t j=0; j<dc_words.size(); j++)
            {
              float I_area = (float)(max_clusters[i].rect & dc_rects[j]).area();
              float U_area = (float)(max_clusters[i].rect.area() + dc_rects[j].area() - I_area);
              float IoU = I_area / U_area;
              if (IoU >=0.5)
              {
                // do not penalize detection of don't care objects
                matched = true;
                if (VDEBUG)
                  rectangle(src, max_clusters[i].rect.tl(), max_clusters[i].rect.br(),
                            Scalar(0,125,255), 1);
                break;
              } 
            }
          }
          if (!matched)
          { 
            im_err++;
            if (VDEBUG)
              rectangle(src, max_clusters[i].rect.tl(), max_clusters[i].rect.br(),
                        Scalar(0,0,255), 2);
          }
        }
        im_misses  += gt_words.size(); //false negatives
        if (VDEBUG)
        {
          for (size_t i=0; i<gt_words.size(); i++)
            rectangle(src, gt_rects[i].tl(), gt_rects[i].br(), Scalar(255,0,0), 2);
        }
     
        correct += im_correct;
        err     += im_err;
        misses  += im_misses;
 
        cout << " RES True Positives  " << correct << endl;
        cout << " RES False Positives " << err << endl;
        cout << " RES False Negatives " << misses << endl;
        float p = (float)correct / (correct+err);
        float r = (float)correct / (correct+misses);
        cout << " RES precision = " << p << endl;
        cout << " RES recall    = " << r << endl;
        cout << " RES f-score   = " << (2*p*r) / (p+r) << endl;
        outfile << " RES True Positives  " << correct << endl;
        outfile << " RES False Positives " << err << endl;
        outfile << " RES False Negatives " << misses << endl;
        outfile << " RES precision = " << p << endl;
        outfile << " RES recall    = " << r << endl;
        outfile << " RES f-score   = " << (2*p*r) / (p+r) << endl;


        if (VDEBUG)
        {
          cout << endl << endl;
          cout << " IMAGE True Positives  " << im_correct << endl;
          cout << " IMAGE False Positives " << im_err << endl;
          cout << " IMAGE False Negatives " << im_misses << endl;
          if ( (src.cols > 1024) || (src.rows > 768) ) 
            resize(src,src,Size(src.cols/2,src.rows/2));
          imshow(GTimagename.c_str(),src);
          imshow("proposals",src_vis);
          waitKey(-1);
          destroyAllWindows();
        }
*/

    } //end foreach image

    outfile.close();
}
