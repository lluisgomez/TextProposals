#include "stopping_rule.h"

StoppingRule::StoppingRule()
{
}

void StoppingRule::operator()(vector<HCluster> &dendrogram, vector<int> &maxIdxs, vector<string> &lex, int min_word_lenght,
                              float weak_classifier_threshold, float cnn_classifier_threshold, bool nms)
{

  //TODO the nms here is suboptimal... (only removes a few number of proposals and we do not gain much time by doing it here instead of in the global bbox nms) but its beautiful in the sense that exploits hierarchical organization

  for (size_t i=0; i<dendrogram.size(); i++)
  {

    //cout << i << "(" << dendrogram[i].node1 << "," << dendrogram[i].node2 << ")" 
    //   << dendrogram[i].cnn_recognition << " " << dendrogram[i].cnn_probability << endl;

    if ((dendrogram[i].node1<0)&&(dendrogram[i].node2<0)) // both nodes are individual regions, so this is the max in the branch
    {
       dendrogram[i].branch_max_probability = dendrogram[i].cnn_probability;
       if (  (dendrogram[i].probability >= weak_classifier_threshold)
          && (dendrogram[i].cnn_probability >= cnn_classifier_threshold)
          && (dendrogram[i].cnn_recognition.size() >= min_word_lenght)
          && ((lex.empty()) || (find (lex.begin(), lex.end(), dendrogram[i].cnn_recognition) != lex.end())) )
       {
         dendrogram[i].branch_maxIdx = i;
         //cout << "    MAX!" << endl;
       } else {
         dendrogram[i].branch_maxIdx = -1;
       }
       continue;
    }

    int node1 = dendrogram[i].node1;
    int node2 = dendrogram[i].node2;

    if ((dendrogram[i].inherit_cnn_probability == 1) && (node1>0))
    {
      dendrogram[i].cnn_probability = dendrogram[node1].cnn_probability;
      dendrogram[i].cnn_recognition = dendrogram[node1].cnn_recognition;
      dendrogram[i].rect = dendrogram[node1].rect;
    }
    else if ((dendrogram[i].inherit_cnn_probability == 2) && (node2>0))
    {
      dendrogram[i].cnn_probability = dendrogram[node2].cnn_probability;
      dendrogram[i].cnn_recognition = dendrogram[node2].cnn_recognition;
      dendrogram[i].rect = dendrogram[node2].rect;
    }



    if ((dendrogram[i].node1>=0)&&(dendrogram[i].node2<0)) // only look at child 1
    {
       if (  (dendrogram[i].cnn_probability*dendrogram[i].nfa >= dendrogram[node1].branch_max_probability)
          && (dendrogram[i].probability >= weak_classifier_threshold)
          && (dendrogram[i].cnn_probability >= cnn_classifier_threshold)
          && (dendrogram[i].cnn_recognition.size() >= min_word_lenght)
          && ((lex.empty()) || (find (lex.begin(), lex.end(), dendrogram[i].cnn_recognition) != lex.end())) )
       {
         for (size_t j=0; j<dendrogram[i].childs.size(); j++)
         {
           dendrogram[dendrogram[i].childs[j]].branch_max_probability = dendrogram[i].cnn_probability * dendrogram[i].nfa;
           dendrogram[dendrogram[i].childs[j]].branch_maxIdx = i;
         }
         //cout << "    MAX!" << endl;
         //cout << "    " << dendrogram[i].cnn_probability << " > " 
         //   << dendrogram[node1].branch_max_probability << endl;
         //cout << "    all childs set to no max " << Mat(dendrogram[i].childs).t() << endl;
       } else {
         dendrogram[i].branch_max_probability = dendrogram[node1].branch_max_probability;
         dendrogram[i].branch_maxIdx = dendrogram[node1].branch_maxIdx;
         //cout << "    no" << endl;
       }
       continue;
    }

    if ((dendrogram[i].node1<0)&&(dendrogram[i].node2>=0)) // only look at child 2
    {
       if (  (dendrogram[i].cnn_probability*dendrogram[i].nfa >= dendrogram[node2].branch_max_probability)
          && (dendrogram[i].probability >= weak_classifier_threshold)
          && (dendrogram[i].cnn_probability >= cnn_classifier_threshold)
          && (dendrogram[i].cnn_recognition.size() >= min_word_lenght)
          && ((lex.empty()) || (find (lex.begin(), lex.end(), dendrogram[i].cnn_recognition) != lex.end())) )
       {
         for (size_t j=0; j<dendrogram[i].childs.size(); j++)
         {
           dendrogram[dendrogram[i].childs[j]].branch_max_probability = dendrogram[i].cnn_probability*dendrogram[i].nfa;
           dendrogram[dendrogram[i].childs[j]].branch_maxIdx = i;
         }
         //cout << "    MAX!" << endl;
         //cout << "    " << dendrogram[i].cnn_probability << " > " 
         //   << dendrogram[node2].branch_max_probability << endl;
         //cout << "    all childs set to no max " << Mat(dendrogram[i].childs).t() << endl;
       } else {
         dendrogram[i].branch_max_probability = dendrogram[node2].branch_max_probability;
         dendrogram[i].branch_maxIdx = dendrogram[node2].branch_maxIdx;
         //cout << "    no" << endl;
       }
       continue;
    }

    // if here we must take a look at both childs
    if (   (dendrogram[i].cnn_probability*dendrogram[i].nfa >= dendrogram[node1].branch_max_probability) 
        && (dendrogram[i].cnn_probability*dendrogram[i].nfa >= dendrogram[node2].branch_max_probability)
        && (dendrogram[i].probability >= weak_classifier_threshold)
        && (dendrogram[i].cnn_probability >= cnn_classifier_threshold)
        && (dendrogram[i].cnn_recognition.size() >= min_word_lenght)
        && ((lex.empty()) || (find (lex.begin(), lex.end(), dendrogram[i].cnn_recognition) != lex.end())) )
    {
      for (size_t j=0; j<dendrogram[i].childs.size(); j++)
      {
        if (dendrogram[i].childs[j]<0) continue;
        dendrogram[dendrogram[i].childs[j]].branch_max_probability = dendrogram[i].cnn_probability*dendrogram[i].nfa;
        dendrogram[dendrogram[i].childs[j]].branch_maxIdx = i;
      }
         //cout << "    MAX!" << endl;
         //cout << "    " << dendrogram[i].cnn_probability << " > " 
         //   << dendrogram[node1].branch_max_probability << " > "
         //   << dendrogram[node2].branch_max_probability << endl;
         //cout << "    all childs set to no max " << Mat(dendrogram[i].childs).t() << endl;
    } else if (dendrogram[node1].branch_max_probability > dendrogram[node2].branch_max_probability){
      dendrogram[i].branch_max_probability = dendrogram[node1].branch_max_probability;
      dendrogram[i].branch_maxIdx = dendrogram[node1].branch_maxIdx;
      //cout << "    no , node1 is better " << dendrogram[node1].branch_max_probability << endl;
      if ((nms) && (dendrogram[node1].branch_maxIdx > 0) && (dendrogram[node2].branch_maxIdx >0))
      {
        if ((dendrogram[dendrogram[node1].branch_maxIdx].rect & dendrogram[dendrogram[node2].branch_maxIdx].rect) 
            == dendrogram[dendrogram[node2].branch_maxIdx].rect)
          dendrogram[node2].branch_maxIdx = dendrogram[node1].branch_maxIdx;
      }
    } else {
      dendrogram[i].branch_max_probability = dendrogram[node2].branch_max_probability;
      dendrogram[i].branch_maxIdx = dendrogram[node2].branch_maxIdx;
      //cout << "    no , node2 is better " << dendrogram[node2].branch_max_probability << endl;
      if ((nms) && (dendrogram[node1].branch_maxIdx > 0) && (dendrogram[node2].branch_maxIdx >0))
      {
        if ((dendrogram[dendrogram[node2].branch_maxIdx].rect & dendrogram[dendrogram[node1].branch_maxIdx].rect) 
            == dendrogram[dendrogram[node1].branch_maxIdx].rect)
          dendrogram[node1].branch_maxIdx = dendrogram[node2].branch_maxIdx;
      }
    }

  }

  for (size_t i=0; i<dendrogram.size(); i++)
  {
    if (dendrogram[i].branch_maxIdx == i)
       maxIdxs.push_back(i);
  }
}
