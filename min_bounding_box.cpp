
#include "min_bounding_box.h"
  
Minibox::Minibox()
{
	initialized = false;
}

void Minibox::check_in (vector<float> *p)
{
	if(!initialized) for (int i=0; i<p->size(); i++)
	{
		edge_begin.push_back(p->at(i));
		edge_end.push_back(p->at(i)+0.00000000000000001);
		initialized = true;
	}
	else for (int i=0; i<p->size(); i++)
	{
		edge_begin.at(i) = min(p->at(i),edge_begin.at(i));
		edge_end.at(i) = max(p->at(i),edge_end.at(i));
		//fprintf(stderr,"	edge_begin[%d] = %e\n",i,edge_begin[i]);
		//fprintf(stderr,"	edge_end[%d] = %e\n",i,edge_end[i]);
	}
}

long double Minibox::volume ()
{
	long double volume = 1;
	for (int i=0; i<edge_begin.size(); i++)
        {
                volume = volume * (edge_end.at(i) - edge_begin.at(i));
		//fprintf(stderr," partial volume = %Le \n",volume);
        }
	//fprintf(stderr," --------- final volume = %Le \n",volume);
	return (volume);
}
