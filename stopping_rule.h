#ifndef STOPPING_RULE_H
#define STOPPING_RULE_H

#include "agglomerative_clustering.h"

class StoppingRule
{
public:
	
	/// Constructor.
	StoppingRule();

	void operator()(vector<HCluster> &dendrogram, vector<int> &maxIdxs, vector<string> &lex, int min_word_lenght = 3,
                        float weak_classifier_threshold = 0.01, float cnn_classifier_threshold = 0.4, bool nms=false);

private:
    
};

#endif
