
#ifndef MIN_BOUNDING_BOX
#define MIN_BOUNDING_BOX

// smallest enclosing box of a set of n points in dimension d
// Class Definitions
// =================

// Minibox 
// --------

#include <vector>

using namespace std;
    
class Minibox {
private:
  vector<float> edge_begin;
  vector<float> edge_end;
  bool   initialized;

public:
  // creates an empty box 
  Minibox(); 

  // copies p to the internal point set
  void        check_in (vector<float> *p);

  // returns the volume of the box
  long double      volume();
};

#endif
