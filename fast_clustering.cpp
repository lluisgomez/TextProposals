/*
  fastcluster: Fast hierarchical clustering routines for R and Python

  Copyright © 2011 Daniel Müllner
  <http://math.stanford.edu/~muellner>

  This library implements various fast algorithms for hierarchical, agglomerative
` clustering methods:

  (1) Algorithms for the "stored matrix approach": the input is the array of
      pairwise dissimilarities.

      MST_linkage_core: single linkage clustering with the "minimum spanning tree
      algorithm (Rohlfs)

      NN_chain_core: nearest-neighbor-chain algorithm, suitable for single,
      complete, average, weighted and Ward linkage (Murtagh)

      generic_linkage: generic algorithm, suitable for all distance update formulas
      (Müllner)

  (2) Algorithms for the "stored data approach": the input are points in a vector
      space.

      MST_linkage_core_vector: single linkage clustering for vector data

      generic_linkage_vector: generic algorithm for vector data, suitable for
      the Ward, centroid and median methods.

      generic_linkage_vector_alternative: alternative scheme for updating the
      nearest neighbors. This method seems faster than "generic_linkage_vector"
      for the centroid and median methods but slower for the Ward method.
*/

//#define __STDC_LIMIT_MACROS
//#include <stdint.h>

#include <limits> // for infinity()

#include <float.h>
#ifndef DBL_MANT_DIG
#error The constant DBL_MANT_DIG could not be defined.
#endif

//#include <cmath>
#include <algorithm>

#ifndef LONG_MAX
#include <limits.h>
#endif
#ifndef LONG_MAX
#error The constant LONG_MAX could not be defined.
#endif
#ifndef INT_MAX
#error The constant INT_MAX could not be defined.
#endif

#ifndef INT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#include <cmath>

typedef int_fast32_t t_index;
#ifndef INT32_MAX
#define MAX_INDEX 0x7fffffffL
#else
#define MAX_INDEX INT32_MAX
#endif
#if (LONG_MAX < MAX_INDEX)
#error The integer format "t_index" must not have a greater range than "long int".
#endif
#if (INT_MAX > MAX_INDEX)
#error The integer format "int" must not have a greater range than "t_index".
#endif
typedef double t_float;
#define T_FLOAT_MANT_DIG DBL_MANT_DIG

enum method_codes {
  // non-Euclidean methods
  METHOD_METR_SINGLE           = 0,
  METHOD_METR_COMPLETE         = 1,
  METHOD_METR_AVERAGE          = 2,
  METHOD_METR_WEIGHTED         = 3,
  METHOD_METR_WARD             = 4,
  METHOD_METR_CENTROID         = 5,
  METHOD_METR_MEDIAN           = 6
};

enum {
  // Euclidean methods
  METHOD_VECTOR_SINGLE         = 0,
  METHOD_VECTOR_WARD           = 1,
  METHOD_VECTOR_CENTROID       = 2,
  METHOD_VECTOR_MEDIAN         = 3
};

enum {
   // Return values
  RET_SUCCESS        = 0,
  RET_MEMORY_ERROR   = 1,
  RET_STL_ERROR      = 2,
  RET_UNKNOWN_ERROR  = 3
 };

// self-destructing array pointer
template <typename type>
class auto_array_ptr{
private:
  type * ptr;
public:
  auto_array_ptr() { ptr = NULL; }
  template <typename index>
  auto_array_ptr(index const size) { init(size); }
  template <typename index, typename value>
  auto_array_ptr(index const size, value const val) { init(size, val); }
  ~auto_array_ptr() {
    delete [] ptr; }
  void free() {
    delete [] ptr;
    ptr = NULL;
  }
  template <typename index>
  void init(index const size) {
    ptr = new type [size];
  }
  template <typename index, typename value>
  void init(index const size, value const val) {
    init(size);
    for (index i=0; i<size; i++) ptr[i] = val;
  }
  inline operator type *() const { return ptr; }
};

struct node {
  t_index node1, node2;
  t_float dist;

  /*
  inline bool operator< (const node a) const {
    return this->dist < a.dist;
  }
  */

  inline friend bool operator< (const node a, const node b) {
    // Numbers are always smaller than NaNs.
    return a.dist < b.dist || (a.dist==a.dist && b.dist!=b.dist);
  }
};

class cluster_result {
private:
  auto_array_ptr<node> Z;
  t_index pos;

public:
  cluster_result(const t_index size)
    : Z(size)
  {
    pos = 0;
  }

  void append(const t_index node1, const t_index node2, const t_float dist) {
    Z[pos].node1 = node1;
    Z[pos].node2 = node2;
    Z[pos].dist  = dist;
    pos++;
  }

  node * operator[] (const t_index idx) const { return Z + idx; }

  /* Define several methods to postprocess the distances. All these functions
     are monotone, so they do not change the sorted order of distances. */

  void sqrt() const {
    for (t_index i=0; i<pos; i++) {
      Z[i].dist = ::sqrt(Z[i].dist);
    }
  }

  void sqrt(const t_float) const { // ignore the argument
    sqrt();
  }

  void sqrtdouble(const t_float) const { // ignore the argument
    for (t_index i=0; i<pos; i++) {
      Z[i].dist = ::sqrt(2*Z[i].dist);
    }
  }

  #ifdef R_pow
  #define my_pow R_pow
  #else
  #define my_pow pow
  #endif

  void power(const t_float p) const {
    t_float const q = 1/p;
    for (t_index i=0; i<pos; i++) {
      Z[i].dist = my_pow(Z[i].dist,q);
    }
  }

  void plusone(const t_float) const { // ignore the argument
    for (t_index i=0; i<pos; i++) {
      Z[i].dist += 1;
    }
  }

  void divide(const t_float denom) const {
    for (t_index i=0; i<pos; i++) {
      Z[i].dist /= denom;
    }
  }
};

class doubly_linked_list {
  /*
    Class for a doubly linked list. Initially, the list is the integer range
    [0, size]. We provide a forward iterator and a method to delete an index
    from the list.

    Typical use: for (i=L.start; L<size; i=L.succ[I])
    or
    for (i=somevalue; L<size; i=L.succ[I])
  */
public:
  t_index start;
  auto_array_ptr<t_index> succ;

private:
  auto_array_ptr<t_index> pred;
  // Not necessarily private, we just do not need it in this instance.

public:
  doubly_linked_list(const t_index size)
    // Initialize to the given size.
    : succ(size+1), pred(size+1)
  {
    for (t_index i=0; i<size; i++) {
      pred[i+1] = i;
      succ[i] = i+1;
    }
    // pred[0] is never accessed!
    //succ[size] is never accessed!
    start = 0;
  }

  void remove(const t_index idx) {
    // Remove an index from the list.
    if (idx==start) {
      start = succ[idx];
    }
    else {
      succ[pred[idx]] = succ[idx];
      pred[succ[idx]] = pred[idx];
    }
    succ[idx] = 0; // Mark as inactive
  }

  bool is_inactive(t_index idx) const {
    return (succ[idx]==0);
  }
};

// Indexing functions
// D is the upper triangular part of a symmetric (NxN)-matrix
// We require r_ < c_ !
#define D_(r_,c_) ( D[(static_cast<std::ptrdiff_t>(2*N-3-(r_))*(r_)>>1)+(c_)-1] )
// Z is an ((N-1)x4)-array
#define Z_(_r, _c) (Z[(_r)*4 + (_c)])

/*
  Lookup function for a union-find data structure.

  The function finds the root of idx by going iteratively through all
  parent elements until a root is found. An element i is a root if
  nodes[i] is zero. To make subsequent searches faster, the entry for
  idx and all its parents is updated with the root element.
 */
class union_find {
private:
  auto_array_ptr<t_index> parent;
  t_index nextparent;

public:
  void init(const t_index size) {
    parent.init(2*size-1, 0);
    nextparent = size;
  }

  t_index Find (t_index idx) const {
    if (parent[idx] !=0 ) { // a → b
      t_index p = idx;
      idx = parent[idx];
      if (parent[idx] !=0 ) { // a → b → c
        do {
          idx = parent[idx];
        } while (parent[idx] != 0);
        do {
          t_index tmp = parent[p];
          parent[p] = idx;
          p = tmp;
        } while (parent[p] != idx);
      }
    }
    return idx;
  }

  void Union (const t_index node1, const t_index node2) {
    parent[node1] = parent[node2] = nextparent++;
  }
};

static void MST_linkage_core(const t_index N, const t_float * const D,
                             cluster_result & Z2) {
/*
    N: integer, number of data points
    D: condensed distance matrix N*(N-1)/2
    Z2: output data structure

    The basis of this algorithm is an algorithm by Rohlf:

    F. James Rohlf, Hierarchical clustering using the minimum spanning tree,
    The Computer Journal, vol. 16, 1973, p. 93–95.

    This implementation should handle Inf values correctly (designed to
    do so but not tested).

    This implementation avoids NaN if possible. It treats NaN as if it was
    greater than +Infinity, ie. whenever we find a non-NaN value, this is
    preferred in all the minimum-distance searches.
*/
  t_index i;
  t_index idx2;
  doubly_linked_list active_nodes(N);
  auto_array_ptr<t_float> d(N);

  t_index prev_node;
  t_float min;

  // first iteration
  idx2 = 1;
  min = d[1] = D[0];
  for (i=2; min!=min && i<N; i++) {  // eliminate NaNs if possible
    min = d[i] = D[i-1];
    idx2 = i;
  }
  for ( ; i<N; i++) {
    d[i] = D[i-1];
    if (d[i] < min) {
      min = d[i];
      idx2 = i;
    }
  }
  Z2.append(0, idx2, min);

  for (t_index j=1; j<N-1; j++) {
    prev_node = idx2;
    active_nodes.remove(prev_node);

    idx2 = active_nodes.succ[0];
    min = d[idx2];
    for (i=idx2; min!=min && i<prev_node; i=active_nodes.succ[i]) {
      min = d[i] = D_(i, prev_node);
      idx2 = i;
    }
    for ( ; i<prev_node; i=active_nodes.succ[i]) {
      if (d[i] > D_(i, prev_node))
        d[i] = D_(i, prev_node);
      if (d[i] < min) {
        min = d[i];
        idx2 = i;
      }
    }
    for (; min!=min && i<N; i=active_nodes.succ[i]) {
      min = d[i] = D_(prev_node, i);
      idx2 = i;
    }
    for (; i<N; i=active_nodes.succ[i]) {
      if (d[i] > D_(prev_node, i))
        d[i] = D_(prev_node, i);
      if (d[i] < min) {
        min = d[i];
        idx2 = i;
      }
    }
    Z2.append(prev_node, idx2, min);
  }
}

/* Functions for the update of the dissimilarity array */

inline static void f_single( t_float * const b, const t_float a ) {
  if (*b > a) *b = a;
}
inline static void f_complete( t_float * const b, const t_float a ) {
  if (*b < a) *b = a;
}
inline static void f_average( t_float * const b, const t_float a, const t_float s, const t_float t) {
  *b = s*a + t*(*b);
}
inline static void f_weighted( t_float * const b, const t_float a) {
  *b = (a+*b)/2;
}
inline static void f_ward( t_float * const b, const t_float a, const t_float c, const t_float s, const t_float t, const t_float v) {
  *b = ( (v+s)*a - v*c + (v+t)*(*b) ) / (s+t+v);
  //*b = a+(*b)-(t*a+s*(*b)+v*c)/(s+t+v);
}
inline static void f_centroid( t_float * const b, const t_float a, const t_float stc, const t_float s, const t_float t) {
  *b = s*a + t*(*b) - stc;
}
inline static void f_median( t_float * const b, const t_float a, const t_float c_4) {
  *b = (a+(*b))/2 - c_4;
}

template <const unsigned char method, typename t_members>
static void NN_chain_core(const t_index N, t_float * const D, t_members * const members, cluster_result & Z2) {
/*
    N: integer
    D: condensed distance matrix N*(N-1)/2
    Z2: output data structure

    This is the NN-chain algorithm, described on page 86 in the following book:

﻿   Fionn Murtagh, Multidimensional Clustering Algorithms,
    Vienna, Würzburg: Physica-Verlag, 1985.

    This implementation does not give defined results when NaN or Inf values
    are present in the array D.
*/
  t_index i;

  auto_array_ptr<t_index> NN_chain(N);
  t_index NN_chain_tip = 0;

  t_index idx1, idx2;

  t_float size1, size2;
  doubly_linked_list active_nodes(N);

  t_float min;

  for (t_index j=0; j<N-1; j++) {
    if (NN_chain_tip <= 3) {
      NN_chain[0] = idx1 = active_nodes.start;
      NN_chain_tip = 1;

      idx2 = active_nodes.succ[idx1];
      min = D_(idx1,idx2);
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i]) {
        if (D_(idx1,i) < min) {
          min = D_(idx1,i);
          idx2 = i;
        }
      }
    }  // a: idx1   b: idx2
    else {
      NN_chain_tip -= 3;
      idx1 = NN_chain[NN_chain_tip-1];
      idx2 = NN_chain[NN_chain_tip];
      min = idx1<idx2 ? D_(idx1,idx2) : D_(idx2,idx1);
    }  // a: idx1   b: idx2

    do {
      NN_chain[NN_chain_tip] = idx2;

      for (i=active_nodes.start; i<idx2; i=active_nodes.succ[i]) {
        if (D_(i,idx2) < min) {
          min = D_(i,idx2);
          idx1 = i;
        }
      }
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i]) {
        if (D_(idx2,i) < min) {
          min = D_(idx2,i);
          idx1 = i;
        }
      }

      idx2 = idx1;
      idx1 = NN_chain[NN_chain_tip++];

    } while (idx2 != NN_chain[NN_chain_tip-2]);

    Z2.append(idx1, idx2, min);

    if (idx1>idx2) {
      t_index tmp = idx1;
      idx1 = idx2;
      idx2 = tmp;
    }

    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD) {
      size1 = static_cast<t_float>(members[idx1]);
      size2 = static_cast<t_float>(members[idx2]);
      members[idx2] += members[idx1];
    }

    // Remove the smaller index from the valid indices (active_nodes).
    active_nodes.remove(idx1);

    switch (method) {
    case METHOD_METR_SINGLE:
      /*
      Single linkage.

      Characteristic: new distances are never longer than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
        f_single(&D_(i, idx2), D_(i, idx1) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=active_nodes.succ[i])
        f_single(&D_(i, idx2), D_(idx1, i) );
      // Update the distance matrix in the range (idx2, N).
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
        f_single(&D_(idx2, i), D_(idx1, i) );
      break;

    case METHOD_METR_COMPLETE:
      /*
      Complete linkage.

      Characteristic: new distances are never shorter than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
        f_complete(&D_(i, idx2), D_(i, idx1) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=active_nodes.succ[i])
        f_complete(&D_(i, idx2), D_(idx1, i) );
      // Update the distance matrix in the range (idx2, N).
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
        f_complete(&D_(idx2, i), D_(idx1, i) );
      break;

    case METHOD_METR_AVERAGE: {
      /*
      Average linkage.

      Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      t_float s = size1/(size1+size2);
      t_float t = size2/(size1+size2);
      for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
        f_average(&D_(i, idx2), D_(i, idx1), s, t );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=active_nodes.succ[i])
        f_average(&D_(i, idx2), D_(idx1, i), s, t );
      // Update the distance matrix in the range (idx2, N).
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
        f_average(&D_(idx2, i), D_(idx1, i), s, t );
      break;
    }

    case METHOD_METR_WEIGHTED:
      /*
      Weighted linkage.

      Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
        f_weighted(&D_(i, idx2), D_(i, idx1) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=active_nodes.succ[i])
        f_weighted(&D_(i, idx2), D_(idx1, i) );
      // Update the distance matrix in the range (idx2, N).
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
        f_weighted(&D_(idx2, i), D_(idx1, i) );
      break;

    case METHOD_METR_WARD:
      /*
      Ward linkage.

      Shorter and longer distances can occur, not smaller than min(d1,d2)
      but maybe bigger than max(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      //t_float v = static_cast<t_float>(members[i]);
      for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
        f_ward(&D_(i, idx2), D_(i, idx1), min,
               size1, size2, static_cast<t_float>(members[i]) );
      // Update the distance matrix in the range (idx1, idx2).
      for (; i<idx2; i=active_nodes.succ[i])
        f_ward(&D_(i, idx2), D_(idx1, i), min,
               size1, size2, static_cast<t_float>(members[i]) );
      // Update the distance matrix in the range (idx2, N).
      for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
        f_ward(&D_(idx2, i), D_(idx1, i), min,
               size1, size2, static_cast<t_float>(members[i]) );
      break;
    }
  }
}

class binary_min_heap {
  /*
  Class for a binary min-heap. The data resides in an array A. The elements of A
  are not changed but two lists I and R of indices are generated which point to
  elements of A and backwards.

  The heap tree structure is

     H[2*i+1]     H[2*i+2]
         \            /
          \          /
           ≤        ≤
            \      /
             \    /
              H[i]

  where the children must be less or equal than their parent. Thus, H[0] contains
  the minimum. The lists I and R are made such that H[i] = A[I[i]] and R[I[i]] = i.

  This implementation avoids NaN if possible. It treats NaN as if it was
  greater than +Infinity, ie. whenever we find a non-NaN value, this is
  preferred in all comparisons.
  */
private:
  t_float * A;
  t_index size;
  auto_array_ptr<t_index> I;
  auto_array_ptr<t_index> R;

public:
  binary_min_heap(const t_index size)
    : I(size), R(size)
  { // Allocate memory and initialize the lists I and R to the identity. This does
    // not make it a heap. Call heapify afterwards!
    this->size = size;
    for (t_index i=0; i<size; i++)
      R[i] = I[i] = i;
  }

  binary_min_heap(const t_index size1, const t_index size2, const t_index start)
    : I(size1), R(size2)
  { // Allocate memory and initialize the lists I and R to the identity. This does
    // not make it a heap. Call heapify afterwards!
    this->size = size1;
    for (t_index i=0; i<size; i++) {
      R[i+start] = i;
      I[i] = i + start;
    }
  }

  void heapify(t_float * const A) {
    // Arrange the indices I and R so that H[i] := A[I[i]] satisfies the heap
    // condition H[i] < H[2*i+1] and H[i] < H[2*i+2] for each i.
    //
    // Complexity: Θ(size)
    // Reference: Cormen, Leiserson, Rivest, Stein, Introduction to Algorithms,
    // 3rd ed., 2009, Section 6.3 “Building a heap”
    t_index idx;
    this->A = A;
    for (idx=(size>>1); idx>0; ) {
      idx--;
      update_geq_(idx);
    }
  }

  inline t_index argmin() const {
    // Return the minimal element.
    return I[0];
  }

  void heap_pop() {
    // Remove the minimal element from the heap.
    size--;
    I[0] = I[size];
    R[I[0]] = 0;
    update_geq_(0);
  }

  void remove(t_index idx) {
    // Remove an element from the heap.
    size--;
    R[I[size]] = R[idx];
    I[R[idx]] = I[size];
    if ( H(size)<=A[idx] || A[idx]!=A[idx] ) {
      update_leq_(R[idx]);
    }
    else {
      update_geq_(R[idx]);
    }
  }

  void replace ( const t_index idxold, const t_index idxnew, const t_float val) {
    R[idxnew] = R[idxold];
    I[R[idxnew]] = idxnew;
    if (val<=A[idxold] || A[idxold]!=A[idxold]) // avoid NaN! ????????????????????
      update_leq(idxnew, val);
    else
      update_geq(idxnew, val);
  }

  void update ( const t_index idx, const t_float val ) const {
    // Update the element A[i] with val and re-arrange the indices the preserve the
    // heap condition.
    if (val<=A[idx] || A[idx]!=A[idx]) // avoid NaN! ????????????????????
      update_leq(idx, val);
    else
      update_geq(idx, val);
  }

  void update_leq ( const t_index idx, const t_float val ) const {
    // Use this when the new value is not more than the old value.
    A[idx] = val;
    update_leq_(R[idx]);
  }

  void update_geq ( const t_index idx, const t_float val ) const {
    // Use this when the new value is not less than the old value.
    A[idx] = val;
    update_geq_(R[idx]);
  }

private:
  void update_leq_ (t_index i) const {
    t_index j;
    for ( ; (i>0) && ( H(i)<H(j=(i-1)>>1) || H(j)!=H(j) ); i=j)
      // avoid NaN!
      heap_swap(i,j);
  }

  void update_geq_ (t_index i) const {
    t_index j;
    for ( ; (j=2*i+1)<size; i=j) {
      if ( H(j)>=H(i) || H(j)!=H(j) ) {  // avoid Nan!
        j++;
        if ( j>=size || H(j)>=H(i) || H(j)!=H(j) ) break; // avoid NaN!
      }
      else if ( j+1<size && H(j+1)<H(j) ) j++;
      heap_swap(i, j);
    }
  }

  void heap_swap(const t_index i, const t_index j) const {
    // Swap two indices.
    t_index tmp = I[i];
    I[i] = I[j];
    I[j] = tmp;
    R[I[i]] = i;
    R[I[j]] = j;
  }

  inline t_float H(const t_index i) const {
    return A[I[i]];
  }

};

template <const unsigned char method, typename t_members>
static void generic_linkage(const t_index N, t_float * const D, t_members * const members, cluster_result & Z2) {
  /*
    N: integer, number of data points
    D: condensed distance matrix N*(N-1)/2
    Z2: output data structure

    This implementation does not give defined results when NaN or Inf values
    are present in the array D.
  */

  const t_index N_1 = N-1;
  t_index i, j; // loop variables
  t_index idx1, idx2; // row and column indices

  auto_array_ptr<t_index> n_nghbr(N_1); // array of nearest neighbors
  auto_array_ptr<t_float> mindist(N_1); // distances to the nearest neighbors
  auto_array_ptr<t_index> row_repr(N); // row_repr[i]: node number that the i-th row
                                       // represents
  doubly_linked_list active_nodes(N);
  binary_min_heap nn_distances(N_1); // minimum heap structure for the distance
                                     // to the nearest neighbor of each point

  t_index node1, node2;     // node numbers in the output
  t_float size1, size2;     // and their cardinalities

  t_float min; // minimum and row index for nearest-neighbor search
  t_index idx;

  for (i=0; i<N; i++)
    // Build a list of row ↔ node label assignments.
    // Initially i ↦ i
    row_repr[i] = i;

  // Initialize the minimal distances:
  // Find the nearest neighbor of each point.
  // n_nghbr[i] = argmin_{j>i} D(i,j) for i in range(N-1)
  t_float * DD = D;
  for (i=0; i<N_1; i++) {
    min = *(DD++);
    idx = j = i+1;
    while (j<N_1) {
      j++;
      if (*DD<min) {
        min = *DD;
        idx = j;
      }
      DD++;
    }
    mindist[i] = min;
    n_nghbr[i] = idx;
  }
  // Put the minimal distances into a heap structure to make the repeated global
  // minimum searches fast.
  nn_distances.heapify(mindist);

  // Main loop: We have N-1 merging steps.
  for (i=0; i<N_1; i++) {
    /*
      Here is a special feature that allows fast bookkeeping and updates of the
      minimal distances.

      mindist[i] stores a lower bound on the minimum distance of the point i to
      all points of higher index:

          mindist[i] ≥ min_{j>i} D(i,j)

      Normally, we have equality. However, this minimum may become invalid due to
      the updates in the distance matrix. The rules are:

      1) If mindist[i] is equal to D(i, n_nghbr[i]), this is the correct minimum
         and n_nghbr[i] is a nearest neighbor.

      2) If mindist[i] is smaller than D(i, n_nghbr[i]), this might not be the
         correct minimum. The minimum needs to be recomputed.

      3) mindist[i] is never bigger than the true minimum. Hence, we never miss the
         true minimum if we take the smallest mindist entry, re-compute the value if
         necessary (thus maybe increasing it) and looking for the now smallest
         mindist entry until a valid minimal entry is found. This step is done in the
         lines below.

      The update process for D below takes care that these rules are fulfilled. This
      makes sure that the minima in the rows D(i,i+1:)of D are re-calculated when
      necessary but re-calculation is avoided whenever possible.

      The re-calculation of the minima makes the worst-case runtime of this algorithm
      cubic in N. We avoid this whenever possible, and in most cases the runtime
      appears to be quadratic.
    */
    idx1 = nn_distances.argmin();
    if (method != METHOD_METR_SINGLE) {
      while ( D_(idx1, n_nghbr[idx1]) > mindist[idx1] ) {
        // Recompute the minimum mindist[idx1] and n_nghbr[idx1].
        n_nghbr[idx1] = j = active_nodes.succ[idx1]; // exists, maximally N-1
        min = D_(idx1,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          if (D_(idx1,j)<min) {
            min = D_(idx1,j);
            n_nghbr[idx1] = j;
          }
        }
        /* Update the heap with the new true minimum and search for the (possibly
           different) minimal entry. */
        nn_distances.update_geq(idx1, min);
        idx1 = nn_distances.argmin();
      }
    }

    nn_distances.heap_pop(); // Remove the current minimum from the heap.
    idx2 = n_nghbr[idx1];

    // Write the newly found minimal pair of nodes to the output array.
    node1 = row_repr[idx1];
    node2 = row_repr[idx2];

    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      size1 = static_cast<t_float>(members[idx1]);
      size2 = static_cast<t_float>(members[idx2]);
      members[idx2] += members[idx1];
    }
    Z2.append(node1, node2, mindist[idx1]);

    // Remove idx1 from the list of active indices (active_nodes).
    active_nodes.remove(idx1);
    // Index idx2 now represents the new (merged) node with label N+i.
    row_repr[idx2] = N+i;

    // Update the distance matrix
    switch (method) {
    case METHOD_METR_SINGLE:
      /*
        Single linkage.

        Characteristic: new distances are never longer than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_single(&D_(j, idx2), D_(j, idx1));
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j]) {
        f_single(&D_(j, idx2), D_(idx1, j));
        // If the new value is below the old minimum in a row, update
        // the mindist and n_nghbr arrays.
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      // Recompute the minimum mindist[idx2] and n_nghbr[idx2].
      if (idx2<N_1) {
        min = mindist[idx2];
        for (j=active_nodes.succ[idx2]; j<N; j=active_nodes.succ[j]) {
          f_single(&D_(idx2, j), D_(idx1, j) );
          if (D_(idx2, j) < min) {
            n_nghbr[idx2] = j;
            min = D_(idx2, j);
          }
        }
        nn_distances.update_leq(idx2, min);
      }
      break;

    case METHOD_METR_COMPLETE:
      /*
        Complete linkage.

        Characteristic: new distances are never shorter than the old distances.
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_complete(&D_(j, idx2), D_(j, idx1) );
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j])
        f_complete(&D_(j, idx2), D_(idx1, j) );
      // Update the distance matrix in the range (idx2, N).
      for (j=active_nodes.succ[idx2]; j<N; j=active_nodes.succ[j])
        f_complete(&D_(idx2, j), D_(idx1, j) );
      break;

    case METHOD_METR_AVERAGE: {
      /*
        Average linkage.

        Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      t_float s = size1/(size1+size2);
      t_float t = size2/(size1+size2);
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_average(&D_(j, idx2), D_(j, idx1), s, t);
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j]) {
        f_average(&D_(j, idx2), D_(idx1, j), s, t);
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        f_average(&D_(idx2, j), D_(idx1, j), s, t);
        min = D_(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          f_average(&D_(idx2, j), D_(idx1, j), s, t);
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
      break;
    }

    case METHOD_METR_WEIGHTED:
      /*
        Weighted linkage.

        Shorter and longer distances can occur.
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_weighted(&D_(j, idx2), D_(j, idx1) );
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j]) {
        f_weighted(&D_(j, idx2), D_(idx1, j) );
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        f_weighted(&D_(idx2, j), D_(idx1, j) );
        min = D_(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          f_weighted(&D_(idx2, j), D_(idx1, j) );
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
      break;

    case METHOD_METR_WARD:
      /*
        Ward linkage.

        Shorter and longer distances can occur, not smaller than min(d1,d2)
        but maybe bigger than max(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_ward(&D_(j, idx2), D_(j, idx1), mindist[idx1],
               size1, size2, static_cast<t_float>(members[j]) );
        if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j]) {
        f_ward(&D_(j, idx2), D_(idx1, j), mindist[idx1], size1, size2,
               static_cast<t_float>(members[j]) );
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        f_ward(&D_(idx2, j), D_(idx1, j), mindist[idx1],
               size1, size2, static_cast<t_float>(members[j]) );
        min = D_(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          f_ward(&D_(idx2, j), D_(idx1, j), mindist[idx1],
                 size1, size2, static_cast<t_float>(members[j]) );
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
      break;

    case METHOD_METR_CENTROID: {
      /*
        Centroid linkage.

        Shorter and longer distances can occur, not bigger than max(d1,d2)
        but maybe smaller than min(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      t_float s = size1/(size1+size2);
      t_float t = size2/(size1+size2);
      t_float stc = s*t*mindist[idx1];
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_centroid(&D_(j, idx2), D_(j, idx1), stc, s, t);
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
        else if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j]) {
        f_centroid(&D_(j, idx2), D_(idx1, j), stc, s, t);
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        f_centroid(&D_(idx2, j), D_(idx1, j), stc, s, t);
        min = D_(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          f_centroid(&D_(idx2, j), D_(idx1, j), stc, s, t);
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
      break;
    }

    case METHOD_METR_MEDIAN:
      /*
        Median linkage.

        Shorter and longer distances can occur, not bigger than max(d1,d2)
        but maybe smaller than min(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      t_float c_4 = mindist[idx1]/4;
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        f_median(&D_(j, idx2), D_(j, idx1), c_4 );
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
        else if (n_nghbr[j] == idx1)
          n_nghbr[j] = idx2;
      }
      // Update the distance matrix in the range (idx1, idx2).
      for (; j<idx2; j=active_nodes.succ[j]) {
        f_median(&D_(j, idx2), D_(idx1, j), c_4 );
        if (D_(j, idx2)<mindist[j]) {
          nn_distances.update_leq(j, D_(j, idx2));
          n_nghbr[j] = idx2;
        }
      }
      // Update the distance matrix in the range (idx2, N).
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        f_median(&D_(idx2, j), D_(idx1, j), c_4 );
        min = D_(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          f_median(&D_(idx2, j), D_(idx1, j), c_4 );
          if (D_(idx2,j)<min) {
            min = D_(idx2,j);
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
      break;
    }
  }
}

/*
  Clustering methods for vector data
*/

template <typename t_dissimilarity>
static void MST_linkage_core_vector(const t_index N,
                                    t_dissimilarity & dist,
                                    cluster_result & Z2) {
/*
    N: integer, number of data points
    dist: function pointer to the metric
    Z2: output data structure

    The basis of this algorithm is an algorithm by Rohlf:

    F. James Rohlf, Hierarchical clustering using the minimum spanning tree,
    The Computer Journal, vol. 16, 1973, p. 93–95.

    This implementation should handle Inf values correctly (designed to
    do so but not tested).

    This implementation avoids NaN if possible. It treats NaN as if it was
    greater than +Infinity, ie. whenever we find a non-NaN value, this is
    preferred in all the minimum-distance searches.
*/
  t_index i;
  t_index idx2;
  doubly_linked_list active_nodes(N);
  auto_array_ptr<t_float> d(N);

  t_index prev_node;
  t_float min;

  // first iteration
  idx2 = 1;
  min = d[1] = dist(0,1);
  for (i=2; min!=min && i<N; i++) { // eliminate NaNs if possible
    min = d[i] = dist(0,i);
    idx2 = i;
  }

  for ( ; i<N; i++) {
    d[i] = dist(0,i);
    if (d[i] < min) {
      min = d[i];
      idx2 = i;
    }
  }

  Z2.append(0, idx2, min);

  for (t_index j=1; j<N-1; j++) {
    prev_node = idx2;
    active_nodes.remove(prev_node);

    idx2 = active_nodes.succ[0];
    min = d[idx2];

    for (i=idx2; min!=min && i<N; i=active_nodes.succ[i]) { // eliminate NaNs if possible
      min = d[i] = dist(i, prev_node);
      idx2 = i;
    }

    for ( ; i<N; i=active_nodes.succ[i]) {
      t_float tmp = dist(i, prev_node);
      if (d[i] > tmp)
        d[i] = tmp;
      if (d[i] < min) {
        min = d[i];
        idx2 = i;
      }
    }
    Z2.append(prev_node, idx2, min);
  }
}

template <const unsigned char method, typename t_dissimilarity>
static void generic_linkage_vector(const t_index N,
                                   t_dissimilarity & dist,
                                   cluster_result & Z2) {
  /*
    N: integer, number of data points
    dist: function pointer to the metric
    Z2: output data structure

    This algorithm is valid for the distance update methods
    "Ward", "centroid" and "median" only!

    This implementation does not give defined results when NaN or Inf values
    are returned by the distance function.
  */
  const t_index N_1 = N-1;
  t_index i, j; // loop variables
  t_index idx1, idx2; // row and column indices

  auto_array_ptr<t_index> n_nghbr(N_1); // array of nearest neighbors
  auto_array_ptr<t_float> mindist(N_1); // distances to the nearest neighbors
  auto_array_ptr<t_index> row_repr(N); // row_repr[i]: node number that the i-th
                                       // row represents
  doubly_linked_list active_nodes(N);
  binary_min_heap nn_distances(N_1); // minimum heap structure for the distance
                                     // to the nearest neighbor of each point

  t_index node1, node2;     // node numbers in the output
  t_float min; // minimum and row index for nearest-neighbor search

  for (i=0; i<N; i++)
    // Build a list of row ↔ node label assignments.
    // Initially i ↦ i
    row_repr[i] = i;

  // Initialize the minimal distances:
  // Find the nearest neighbor of each point.
  // n_nghbr[i] = argmin_{j>i} D(i,j) for i in range(N-1)
  for (i=0; i<N_1; i++) {
    t_index idx = j = i+1;
    switch (method) {
    case METHOD_METR_WARD:
      min = dist.ward_initial(i,j);
      break;
    default:
      min = dist.sqeuclidean(i,j);
    }
    for(j++; min!=min && j<N; j++) { // eliminate NaN if possible
      switch (method) {
      case METHOD_METR_WARD:
        min = dist.ward_initial(i,j);
        break;
      default:
        min = dist.sqeuclidean(i,j);
      }
      idx = j;
    }
    for( ; j<N; j++) {
      t_float tmp;
      switch (method) {
      case METHOD_METR_WARD:
        tmp = dist.ward_initial(i,j);
        break;
      default:
        tmp = dist.sqeuclidean(i,j);
      }
      if (tmp<min) {
        min = tmp;
        idx = j;
      }
    }
    switch (method) {
    case METHOD_METR_WARD:
      mindist[i] = t_dissimilarity::ward_initial_conversion(min);
      break;
    default:
      mindist[i] = min;
    }
    n_nghbr[i] = idx;
  }

  // Put the minimal distances into a heap structure to make the repeated global
  // minimum searches fast.
  nn_distances.heapify(mindist);

  // Main loop: We have N-1 merging steps.
  for (i=0; i<N_1; i++) {
    idx1 = nn_distances.argmin();

    while ( active_nodes.is_inactive(n_nghbr[idx1]) ) {
      // Recompute the minimum mindist[idx1] and n_nghbr[idx1].
      n_nghbr[idx1] = j = active_nodes.succ[idx1]; // exists, maximally N-1
      switch (method) {
      case METHOD_METR_WARD:
        min = dist.ward(idx1,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          t_float const tmp = dist.ward(idx1,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx1] = j;
          }
        }
        break;
      default:
        min = dist.sqeuclidean(idx1,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          t_float const tmp = dist.sqeuclidean(idx1,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx1] = j;
          }
        }
      }
      /* Update the heap with the new true minimum and search for the (possibly
         different) minimal entry. */
      nn_distances.update_geq(idx1, min);
      idx1 = nn_distances.argmin();
    }

    nn_distances.heap_pop(); // Remove the current minimum from the heap.
    idx2 = n_nghbr[idx1];

    // Write the newly found minimal pair of nodes to the output array.
    node1 = row_repr[idx1];
    node2 = row_repr[idx2];

    Z2.append(node1, node2, mindist[idx1]);

    switch (method) {
    case METHOD_METR_WARD:
    case METHOD_METR_CENTROID:
      dist.merge_inplace(idx1, idx2);
      break;
    case METHOD_METR_MEDIAN:
      dist.merge_inplace_weighted(idx1, idx2);
      break;
    }

    // Index idx2 now represents the new (merged) node with label N+i.
    row_repr[idx2] = N+i;
    // Remove idx1 from the list of active indices (active_nodes).
    active_nodes.remove(idx1);  // TBD later!!!

    // Update the distance matrix
    switch (method) {
    case METHOD_METR_WARD:
      /*
        Ward linkage.

        Shorter and longer distances can occur, not smaller than min(d1,d2)
        but maybe bigger than max(d1,d2).
      */
      // Update the distance matrix in the range [start, idx1).
      for (j=active_nodes.start; j<idx1; j=active_nodes.succ[j]) {
        if (n_nghbr[j] == idx2) {
          n_nghbr[j] = idx1; // invalidate
        }
      }
      // Update the distance matrix in the range (idx1, idx2).
      for ( ; j<idx2; j=active_nodes.succ[j]) {
        t_float const tmp = dist.ward(j, idx2);
        if (tmp<mindist[j]) {
          nn_distances.update_leq(j, tmp);
          n_nghbr[j] = idx2;
        }
        else if (n_nghbr[j]==idx2) {
          n_nghbr[j] = idx1; // invalidate
        }
      }
      // Find the nearest neighbor for idx2.
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        min = dist.ward(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          t_float const tmp = dist.ward(idx2,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
      break;

    default:
      /*
        Centroid and median linkage.

        Shorter and longer distances can occur, not bigger than max(d1,d2)
        but maybe smaller than min(d1,d2).
      */
      for (j=active_nodes.start; j<idx2; j=active_nodes.succ[j]) {
        t_float const tmp = dist.sqeuclidean(j, idx2);
        if (tmp<mindist[j]) {
          nn_distances.update_leq(j, tmp);
          n_nghbr[j] = idx2;
        }
        else if (n_nghbr[j] == idx2)
          n_nghbr[j] = idx1; // invalidate
      }
      // Find the nearest neighbor for idx2.
      if (idx2<N_1) {
        n_nghbr[idx2] = j = active_nodes.succ[idx2]; // exists, maximally N-1
        min = dist.sqeuclidean(idx2,j);
        for (j=active_nodes.succ[j]; j<N; j=active_nodes.succ[j]) {
          t_float const tmp = dist.sqeuclidean(idx2, j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx2] = j;
          }
        }
        nn_distances.update(idx2, min);
      }
    }
  }
}

template <const unsigned char method, typename t_dissimilarity>
static void generic_linkage_vector_alternative(const t_index N,
                                               t_dissimilarity & dist,
                                               cluster_result & Z2) {
  /*
    N: integer, number of data points
    dist: function pointer to the metric
    Z2: output data structure

    This algorithm is valid for the distance update methods
    "Ward", "centroid" and "median" only!

    This implementation does not give defined results when NaN or Inf values
    are returned by the distance function.
  */
  const t_index N_1 = N-1;
  t_index i, j=0; // loop variables
  t_index idx1, idx2; // row and column indices

  auto_array_ptr<t_index> n_nghbr(2*N-2); // array of nearest neighbors
  auto_array_ptr<t_float> mindist(2*N-2); // distances to the nearest neighbors

  doubly_linked_list active_nodes(N+N_1);
  binary_min_heap nn_distances(N_1, 2*N-2, 1); // minimum heap structure for the
                               // distance to the nearest neighbor of each point

  t_float min; // minimum for nearest-neighbor searches

  // Initialize the minimal distances:
  // Find the nearest neighbor of each point.
  // n_nghbr[i] = argmin_{j<i} D(i,j) for i in range(N-1)
  for (i=1; i<N; i++) {
    t_index idx = j = 0;
    switch (method) {
    case METHOD_METR_WARD:
      min = dist.ward_initial(i,j);
      break;
    default:
      min = dist.sqeuclidean(i,j);
    }
    for(j++; min!=min && j<i; j++) { // eliminate NaN if possible
      switch (method) {
      case METHOD_METR_WARD:
        min = dist.ward_initial(i,j);
        break;
      default:
        min = dist.sqeuclidean(i,j);
      }
      idx = j;
    }
    for( ; j<i; j++) {
      t_float tmp;
      switch (method) {
      case METHOD_METR_WARD:
        tmp = dist.ward_initial(i,j);
        break;
      default:
        tmp = dist.sqeuclidean(i,j);
      }
      if (tmp<min) {
        min = tmp;
        idx = j;
      }
    }
    switch (method) {
    case METHOD_METR_WARD:
      mindist[i] = t_dissimilarity::ward_initial_conversion(min);
      break;
    default:
      mindist[i] = min;
    }
    n_nghbr[i] = idx;
  }

  // Put the minimal distances into a heap structure to make the repeated global
  // minimum searches fast.
  nn_distances.heapify(mindist);

  // Main loop: We have N-1 merging steps.
  for (i=N; i<N+N_1; i++) {
    /*
      The bookkeeping is different from the "stored matrix approach" algorithm
      generic_linkage.

      mindist[i] stores a lower bound on the minimum distance of the point i to
      all points of *lower* index:

          mindist[i] ≥ min_{j<i} D(i,j)

      Moreover, new nodes do not re-use one of the old indices, but they are given
      a new, unique index (SciPy convention: initial nodes are 0,…,N−1, new
      nodes are N,…,2N−2).

      Invalid nearest neighbors are not recognized by the fact that the stored
      distance is smaller than the actual distance, but the list active_nodes
      maintains a flag whether a node is inactive. If n_nghbr[i] points to an
      active node, the entries nn_distances[i] and n_nghbr[i] are valid, otherwise
      they must be recomputed.
    */
    idx1 = nn_distances.argmin();
    while ( active_nodes.is_inactive(n_nghbr[idx1]) ) {
      // Recompute the minimum mindist[idx1] and n_nghbr[idx1].
      n_nghbr[idx1] = j = active_nodes.start;
      switch (method) {
      case METHOD_METR_WARD:
        min = dist.ward_extended(idx1,j);
        for (j=active_nodes.succ[j]; j<idx1; j=active_nodes.succ[j]) {
          t_float tmp = dist.ward_extended(idx1,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx1] = j;
          }
        }
        break;
      default:
        min = dist.sqeuclidean_extended(idx1,j);
        for (j=active_nodes.succ[j]; j<idx1; j=active_nodes.succ[j]) {
          t_float const tmp = dist.sqeuclidean_extended(idx1,j);
          if (tmp<min) {
            min = tmp;
            n_nghbr[idx1] = j;
          }
        }
      }
      /* Update the heap with the new true minimum and search for the (possibly
         different) minimal entry. */
      nn_distances.update_geq(idx1, min);
      idx1 = nn_distances.argmin();
    }

    idx2 = n_nghbr[idx1];
    active_nodes.remove(idx1);
    active_nodes.remove(idx2);

    Z2.append(idx1, idx2, mindist[idx1]);

    if (i<2*N_1) {
      switch (method) {
      case METHOD_METR_WARD:
      case METHOD_METR_CENTROID:
        dist.merge(idx1, idx2, i);
        break;

      case METHOD_METR_MEDIAN:
        dist.merge_weighted(idx1, idx2, i);
        break;
      }

      n_nghbr[i] = active_nodes.start;
      if (method==METHOD_METR_WARD) {
        /*
          Ward linkage.

          Shorter and longer distances can occur, not smaller than min(d1,d2)
          but maybe bigger than max(d1,d2).
        */
        min = dist.ward_extended(active_nodes.start, i);
        // TBD: avoid NaN
        for (j=active_nodes.succ[active_nodes.start]; j<i; j=active_nodes.succ[j]) {
          t_float tmp = dist.ward_extended(j, i);
          if (tmp<min) {
            min = tmp;
            n_nghbr[i] = j;
          }
        }
      }
      else {
        /*
          Centroid and median linkage.

          Shorter and longer distances can occur, not bigger than max(d1,d2)
          but maybe smaller than min(d1,d2).
        */
        min = dist.sqeuclidean_extended(active_nodes.start, i);
        // TBD: avoid NaN
        for (j=active_nodes.succ[active_nodes.start]; j<i; j=active_nodes.succ[j]) {
          t_float tmp = dist.sqeuclidean_extended(j, i);
          if (tmp<min) {
            min = tmp;
            n_nghbr[i] = j;
          }
        }
      }
      if (idx2<active_nodes.start)  {
        nn_distances.remove(active_nodes.start);
      } else {
        nn_distances.remove(idx2);
      }
      nn_distances.replace(idx1, i, min);
    }
  }
}




class linkage_output {
private:
  t_float * Z;
  t_index pos;

public:
  linkage_output(t_float * const Z) {
    this->Z = Z;
    pos = 0;
  }

  void append(const t_index node1, const t_index node2, const t_float dist, const t_float size) {
    if (node1<node2) {
      Z[pos++] = static_cast<t_float>(node1);
      Z[pos++] = static_cast<t_float>(node2);
    }
    else {
      Z[pos++] = static_cast<t_float>(node2);
      Z[pos++] = static_cast<t_float>(node1);
    }
    Z[pos++] = dist;
    Z[pos++] = size;
  }
};

/*
  Generate the specific output format for a dendrogram from the
  clustering output.

  The list of merging steps can be sorted or unsorted.
*/

// The size of a node is either 1 (a single point) or is looked up from
// one of the clusters.
#define size_fc_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

template <bool sorted>
static void generate_dendrogram(t_float * const Z, cluster_result & Z2, const t_index N) {
   //fprintf(stderr, "  entering generate_dendrogram\n");

  // The array "nodes" is a union-find data structure for the cluster
  // identites (only needed for unsorted cluster_result input).
  union_find nodes;
  if (!sorted) {
    std::stable_sort(Z2[0], Z2[N-1]);
    nodes.init(N);
  }

  linkage_output output(Z);
  t_index node1, node2;

  for (t_index i=0; i<N-1; i++) {
    // Get two data points whose clusters are merged in step i.
    if (sorted) {
      node1 = Z2[i]->node1;
      node2 = Z2[i]->node2;
    }
    else {
      // Find the cluster identifiers for these points.
      node1 = nodes.Find(Z2[i]->node1);
      node2 = nodes.Find(Z2[i]->node2);
      // Merge the nodes in the union-find data structure by making them
      // children of a new node.
      nodes.Union(node1, node2);
    }
   //fprintf(stderr, "	node1 = %d , node2 = %d , Z2[i]->dist = %f , size_fc_(node1)+size_fc_(node2) = %f", node1, node2, Z2[i]->dist, size_fc_(node1)+size_fc_(node2));
    output.append(node1, node2, Z2[i]->dist, size_fc_(node1)+size_fc_(node2));
  }
}

/*
   Clustering on vector data
*/

enum {
  // metrics
  METRIC_EUCLIDEAN       =  0,
  METRIC_MINKOWSKI       =  1,
  METRIC_CITYBLOCK       =  2,
  METRIC_SEUCLIDEAN      =  3,
  METRIC_SQEUCLIDEAN     =  4,
  METRIC_COSINE          =  5,
  METRIC_HAMMING         =  6,
  METRIC_JACCARD         =  7,
  METRIC_CHEBYCHEV       =  8,
  METRIC_CANBERRA        =  9,
  METRIC_BRAYCURTIS      = 10,
  METRIC_MAHALANOBIS     = 11,
  METRIC_YULE            = 12,
  METRIC_MATCHING        = 13,
  METRIC_DICE            = 14,
  METRIC_ROGERSTANIMOTO  = 15,
  METRIC_RUSSELLRAO      = 16,
  METRIC_SOKALSNEATH     = 17,
  METRIC_KULSINSKI       = 18,
  METRIC_USER            = 19,
  METRIC_INVALID         = 20, // sentinel
  METRIC_JACCARD_BOOL    = 21 // separate function for Jaccard metric on Boolean
};                             // input data

/*
  This class handles all the information about the dissimilarity
  computation.
*/

class dissimilarity {
private:
  t_float * Xa;
  auto_array_ptr<t_float> Xnew;
  std::ptrdiff_t dim; // size_t saves many statis_cast<> in products
  t_index N;
  t_index * members;
  void (cluster_result::*postprocessfn) (const t_float) const;
  t_float postprocessarg;

  t_float (dissimilarity::*distfn) (const t_index, const t_index) const;

  auto_array_ptr<t_float> precomputed;
  t_float * precomputed2;

  t_float * V;
  const t_float * V_data;

public:
  dissimilarity (t_float * const Xa, int N, int dim,
                        t_index * const members,
                        const unsigned char method,
                        const unsigned char metric,
                        bool temp_point_array)
    : Xa(Xa),
      dim(dim),
      N(N),
      members(members),
      postprocessfn(NULL),
      V(NULL)
  {
  //fprintf(stderr, " constructing dissimilarity\n");
	//for (int i=0; i<8; i++)
		//fprintf(stderr, " my vector %f \n", Xa[i]);
    switch (method) {
    case METHOD_METR_SINGLE:
      postprocessfn = NULL; // default
      switch (metric) {
      case METRIC_EUCLIDEAN:
        set_euclidean();
        break;
      case METRIC_SEUCLIDEAN:
        /*if (extraarg==NULL) {
          PyErr_SetString(PyExc_TypeError,
                          "The 'seuclidean' metric needs a variance parameter.");
          throw pythonerror();
        }
        V  = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(extraarg,
                                               PyArray_DescrFromType(NPY_DOUBLE),
                                               1, 1,
                                               NPY_ARRAY_CARRAY_RO,
                                               NULL));
        if (PyErr_Occurred()) {
          throw pythonerror();
        }
        if (PyArray_DIM(V, 0)!=dim) {
          PyErr_SetString(PyExc_ValueError,
          "The variance vector must have the same dimensionality as the data.");
          throw pythonerror();
        }
        V_data = reinterpret_cast<t_float *>(PyArray_DATA(V));
        distfn = &dissimilarity::seuclidean;
        postprocessfn = &cluster_result::sqrt;
        break;*/
      case METRIC_SQEUCLIDEAN:
        distfn = &dissimilarity::sqeuclidean;
        break;
      case METRIC_CITYBLOCK:
        set_cityblock();
        break;
      case METRIC_CHEBYCHEV:
        set_chebychev();
        break;
      case METRIC_MINKOWSKI:
        //set_minkowski(extraarg);
        break;
      case METRIC_COSINE:
        distfn = &dissimilarity::cosine;
        postprocessfn = &cluster_result::plusone;
        // precompute norms
        precomputed.init(N);
        for (t_index i=0; i<N; i++) {
          t_float sum=0;
          for (t_index k=0; k<dim; k++) {
            sum += X(i,k)*X(i,k);
          }
          precomputed[i] = 1/sqrt(sum);
        }
        break;
      case METRIC_HAMMING:
        distfn = &dissimilarity::hamming;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_JACCARD:
        distfn = &dissimilarity::jaccard;
        break;
      case METRIC_CANBERRA:
        distfn = &dissimilarity::canberra;
        break;
      case METRIC_BRAYCURTIS:
        distfn = &dissimilarity::braycurtis;
        break;
      case METRIC_MAHALANOBIS:
        /*if (extraarg==NULL) {
          PyErr_SetString(PyExc_TypeError,
            "The 'mahalanobis' metric needs a parameter for the inverse covariance.");
          throw pythonerror();
        }
        V = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(extraarg,
              PyArray_DescrFromType(NPY_DOUBLE),
              2, 2,
              NPY_ARRAY_CARRAY_RO,
              NULL));
        if (PyErr_Occurred()) {
          throw pythonerror();
        }
        if (PyArray_DIM(V, 0)!=N || PyArray_DIM(V, 1)!=dim) {
          PyErr_SetString(PyExc_ValueError,
            "The inverse covariance matrix has the wrong size.");
          throw pythonerror();
        }
        V_data = reinterpret_cast<t_float *>(PyArray_DATA(V));
        distfn = &dissimilarity::mahalanobis;
        postprocessfn = &cluster_result::sqrt;
        break;*/
      case METRIC_YULE:
        distfn = &dissimilarity::yule;
        break;
      case METRIC_MATCHING:
        distfn = &dissimilarity::matching;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_DICE:
        distfn = &dissimilarity::dice;
        break;
      case METRIC_ROGERSTANIMOTO:
        distfn = &dissimilarity::rogerstanimoto;
        break;
      case METRIC_RUSSELLRAO:
        distfn = &dissimilarity::russellrao;
        postprocessfn = &cluster_result::divide;
        postprocessarg = static_cast<t_float>(dim);
        break;
      case METRIC_SOKALSNEATH:
        distfn = &dissimilarity::sokalsneath;
        break;
      case METRIC_KULSINSKI:
        distfn = &dissimilarity::kulsinski;
        postprocessfn = &cluster_result::plusone;
        precomputed.init(N);
        for (t_index i=0; i<N; i++) {
          t_index sum=0;
          for (t_index k=0; k<dim; k++) {
            sum += Xb(i,k);
          }
          precomputed[i] = -.5/static_cast<t_float>(sum);
        }
        break;
      default: // case METRIC_JACCARD_BOOL:
        distfn = &dissimilarity::jaccard_bool;
      }
      break;

    case METHOD_METR_WARD:
      postprocessfn = &cluster_result::sqrtdouble;
      break;

    default:
      postprocessfn = &cluster_result::sqrt;
    }

    if (temp_point_array) {
      Xnew.init((N-1)*dim);
    }
    //fprintf(stderr, " first distance %f \n", (this->*distfn)(0,1));
  }

  ~dissimilarity() {
    free(V);
  }

  inline t_float operator () (const t_index i, const t_index j) const {
    return (this->*distfn)(i,j);
  }

  inline t_float X (const t_index i, const t_index j) const {
    return Xa[i*dim+j];
  }

  inline bool Xb (const t_index i, const t_index j) const {
    return  reinterpret_cast<bool *>(Xa)[i*dim+j];
  }

  inline t_float * Xptr(const t_index i, const t_index j) const {
    return Xa+i*dim+j;
  }

  void merge(const t_index i, const t_index j, const t_index newnode) const {
    t_float const * const Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim;
    t_float const * const Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for(t_index k=0; k<dim; k++) {
      Xnew[(newnode-N)*dim+k] = (Pi[k]*static_cast<t_float>(members[i]) +
                                 Pj[k]*static_cast<t_float>(members[j])) /
        static_cast<t_float>(members[i]+members[j]);
    }
    members[newnode] = members[i]+members[j];
  }

  void merge_weighted(const t_index i, const t_index j, const t_index newnode) const {
    t_float const * const Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim;
    t_float const * const Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for(t_index k=0; k<dim; k++) {
      Xnew[(newnode-N)*dim+k] = (Pi[k]+Pj[k])*.5;
    }
  }

  void merge_inplace(const t_index i, const t_index j) const {
    t_float const * const Pi = Xa+i*dim;
    t_float * const Pj = Xa+j*dim;
    for(t_index k=0; k<dim; k++) {
      Pj[k] = (Pi[k]*static_cast<t_float>(members[i]) +
               Pj[k]*static_cast<t_float>(members[j])) /
        static_cast<t_float>(members[i]+members[j]);
    }
    members[j] += members[i];
  }

  void merge_inplace_weighted(const t_index i, const t_index j) const {
    t_float const * const Pi = Xa+i*dim;
    t_float * const Pj = Xa+j*dim;
    for(t_index k=0; k<dim; k++) {
      Pj[k] = (Pi[k]+Pj[k])*.5;
    }
  }

  void postprocess(cluster_result & Z2) const {
    if (postprocessfn!=NULL) {
        (Z2.*postprocessfn)(postprocessarg);
    }
  }

  inline t_float ward(const t_index i, const t_index j) const {
    t_float mi = static_cast<t_float>(members[i]);
    t_float mj = static_cast<t_float>(members[j]);
    return sqeuclidean(i,j)*mi*mj/(mi+mj);
  }

  inline t_float ward_initial(const t_index i, const t_index j) const {
    // alias for sqeuclidean
    // Factor 2!!!
    return sqeuclidean(i,j);
  }

  inline static t_float ward_initial_conversion(const t_float min) {
    return min*.5;
  }

  inline t_float ward_extended(const t_index i, const t_index j) const {
    t_float mi = static_cast<t_float>(members[i]);
    t_float mj = static_cast<t_float>(members[j]);
    return sqeuclidean_extended(i,j)*mi*mj/(mi+mj);
  }

  t_float sqeuclidean(const t_index i, const t_index j) const {
    t_float sum = 0;
    //fprintf(stderr, " 	entering sqeuclidean\n");
    /*
    for (t_index k=0; k<dim; k++) {
        t_float diff = X(i,k) - X(j,k);
        sum += diff*diff;
    }
    */
    // faster
    t_float const * Pi = Xa+i*dim;
    t_float const * Pj = Xa+j*dim;
    for (t_index k=0; k<dim; k++) {
      t_float diff = Pi[k] - Pj[k];
      //fprintf(stderr, "   %f - %f = %f\n", Pi[k] , Pj[k] , diff);
      
      sum += diff*diff;
    }
    //fprintf(stderr, "   sum = %f",sum);
    //fprintf(stderr, "	calculated distance : %f\n", sum);
    return sum;
  }

  t_float sqeuclidean_extended(const t_index i, const t_index j) const {
    t_float sum = 0;
    t_float const * Pi = i<N ? Xa+i*dim : Xnew+(i-N)*dim; // TBD
    t_float const * Pj = j<N ? Xa+j*dim : Xnew+(j-N)*dim;
    for (t_index k=0; k<dim; k++) {
      t_float diff = Pi[k] - Pj[k];
      sum += diff*diff;
    }
    return sum;
  }

private:

  void set_euclidean() {
    distfn = &dissimilarity::sqeuclidean;
    postprocessfn = &cluster_result::sqrt;
  }

  void set_cityblock() {
    distfn = &dissimilarity::cityblock;
  }

  void set_chebychev() {
    distfn = &dissimilarity::chebychev;
  }

  t_float seuclidean(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      t_float diff = X(i,k)-X(j,k);
      sum += diff*diff/V_data[k];
    }
    return sum;
  }

  t_float cityblock(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += fabs(X(i,k)-X(j,k));
    }
    return sum;
  }

  t_float minkowski(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += pow(fabs(X(i,k)-X(j,k)),postprocessarg);
    }
    return sum;
  }

  t_float chebychev(const t_index i, const t_index j) const {
    t_float max = 0;
    for (t_index k=0; k<dim; k++) {
      t_float diff = fabs(X(i,k)-X(j,k));
      if (diff>max) {
        max = diff;
      }
    }
    return max;
  }

  t_float cosine(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum -= X(i,k)*X(j,k);
    }
    return sum*precomputed[i]*precomputed[j];
  }

  t_float hamming(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += (X(i,k)!=X(j,k));
    }
    return sum;
  }

  // Differs from scipy.spatial.distance: equal vectors correctly
  // return distance 0.
  t_float jaccard(const t_index i, const t_index j) const {
    t_index sum1 = 0;
    t_index sum2 = 0;
    for (t_index k=0; k<dim; k++) {
      sum1 += (X(i,k)!=X(j,k));
      sum2 += ((X(i,k)!=0) || (X(j,k)!=0));
    }
    return sum1==0 ? 0 : static_cast<t_float>(sum1) / static_cast<t_float>(sum2);
  }

  t_float canberra(const t_index i, const t_index j) const {
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      t_float numerator = fabs(X(i,k)-X(j,k));
      sum += numerator==0 ? 0 : numerator / (fabs(X(i,k)) + fabs(X(j,k)));
    }
    return sum;
  }


  t_float braycurtis(const t_index i, const t_index j) const {
    t_float sum1 = 0;
    t_float sum2 = 0;
    for (t_index k=0; k<dim; k++) {
      sum1 += fabs(X(i,k)-X(j,k));
      sum2 += fabs(X(i,k)+X(j,k));
    }
    return sum1/sum2;
  }

  t_float mahalanobis(const t_index i, const t_index j) const {
    // V_data contains the product X*VI
    t_float sum = 0;
    for (t_index k=0; k<dim; k++) {
      sum += (V_data[i*dim+k]-V_data[j*dim+k])*(X(i,k)-X(j,k));
    }
    return sum;
  }

  t_index mutable NTT; // 'local' variables
  t_index mutable NXO;
  t_index mutable NTF;
  #define NTFFT NTF
  #define NFFTT NTT

  void nbool_correspond(const t_index i, const t_index j) const {
    NTT = 0;
    NXO = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
    }
  }

  void nbool_correspond_tfft(const t_index i, const t_index j) const {
    NTT = 0;
    NXO = 0;
    NTF = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
      NTF += (Xb(i,k) & ~Xb(j,k)) ;
    }
    NTF *= (NXO-NTF); // NTFFT
    NTT *= (dim-NTT-NXO); // NFFTT
  }

  void nbool_correspond_xo(const t_index i, const t_index j) const {
    NXO = 0;
    for (t_index k=0; k<dim; k++) {
      NXO += (Xb(i,k) ^  Xb(j,k)) ;
    }
  }

  void nbool_correspond_tt(const t_index i, const t_index j) const {
    NTT = 0;
    for (t_index k=0; k<dim; k++) {
      NTT += (Xb(i,k) &  Xb(j,k)) ;
    }
  }

  // Caution: zero denominators can happen here!
  t_float yule(const t_index i, const t_index j) const {
    nbool_correspond_tfft(i, j);
    return static_cast<t_float>(2*NTFFT) / static_cast<t_float>(NTFFT + NFFTT);
  }

  // Prevent a zero denominator for equal vectors.
  t_float dice(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(NXO) / static_cast<t_float>(NXO+2*NTT);
  }

  t_float rogerstanimoto(const t_index i, const t_index j) const {
    nbool_correspond_xo(i, j);
    return static_cast<t_float>(2*NXO) / static_cast<t_float>(NXO+dim);
  }

  t_float russellrao(const t_index i, const t_index j) const {
    nbool_correspond_tt(i, j);
    return static_cast<t_float>(dim-NTT);
  }

  // Prevent a zero denominator for equal vectors.
  t_float sokalsneath(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(2*NXO) / static_cast<t_float>(NTT+2*NXO);
  }

  t_float kulsinski(const t_index i, const t_index j) const {
    nbool_correspond_tt(i, j);
    return static_cast<t_float>(NTT) * (precomputed[i] + precomputed[j]);
  }

  // 'matching' distance = Hamming distance
  t_float matching(const t_index i, const t_index j) const {
    nbool_correspond_xo(i, j);
    return static_cast<t_float>(NXO);
  }

  // Prevent a zero denominator for equal vectors.
  t_float jaccard_bool(const t_index i, const t_index j) const {
    nbool_correspond(i, j);
    return (NXO==0) ? 0 :
      static_cast<t_float>(NXO) / static_cast<t_float>(NXO+NTT);
  }
};


/*Clustering for the "stored matrix approach": the input is the array of pairwise dissimilarities*/
static int linkage(t_float *D, int N, t_float * Z, unsigned char method)
{

  try{

    if (N < 1 ) {
      // N must be at least 1.
      //fprintf(stderr,"At least one element is needed for clustering.");
      return -1;
    }

    // (1)
    // The biggest index used below is 4*(N-2)+3, as an index to Z. This must fit
    // into the data type used for indices.
    // (2)
    // The largest representable integer, without loss of precision, by a floating
    // point number of type t_float is 2^T_FLOAT_MANT_DIG. Here, we make sure that
    // all cluster labels from 0 to 2N-2 in the output can be accurately represented
    // by a floating point number.
    //if (N > MAX_INDEX/4 || (N-1)>>(T_FLOAT_MANT_DIG-1) > 0) {
        //fprintf(stderr,"Data is too big, index overflow.");
    //  return -1;
    //}

    if (method>METHOD_METR_MEDIAN) {
	//fprintf(stderr,"Invalid method index.");
      return -1;
    }


    cluster_result Z2(N-1);
    auto_array_ptr<t_index> members;
    // For these methods, the distance update formula needs the number of
    // data points in a cluster.
    if (method==METHOD_METR_AVERAGE ||
        method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      members.init(N, 1);
    }
    // Operate on squared distances for these methods.
    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      for (std::ptrdiff_t i=0; i < static_cast<std::ptrdiff_t>(N)*(N-1)/2; i++)
        D[i] *= D[i];
    }

    switch (method) {
    case METHOD_METR_SINGLE:
      MST_linkage_core(N, D, Z2);
      break;
    case METHOD_METR_COMPLETE:
      NN_chain_core<METHOD_METR_COMPLETE, t_index>(N, D, NULL, Z2);
      break;
    case METHOD_METR_AVERAGE:
      NN_chain_core<METHOD_METR_AVERAGE, t_index>(N, D, members, Z2);
      break;
    case METHOD_METR_WEIGHTED:
      NN_chain_core<METHOD_METR_WEIGHTED, t_index>(N, D, NULL, Z2);
      break;
    case METHOD_METR_WARD:
      NN_chain_core<METHOD_METR_WARD, t_index>(N, D, members, Z2);
      break;
    case METHOD_METR_CENTROID:
      generic_linkage<METHOD_METR_CENTROID, t_index>(N, D, members, Z2);
      break;
    default: // case METHOD_METR_MEDIAN
      generic_linkage<METHOD_METR_MEDIAN, t_index>(N, D, NULL, Z2);
    }

    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID ||
        method==METHOD_METR_MEDIAN) {
      Z2.sqrt();
    }

    if (method==METHOD_METR_CENTROID ||method==METHOD_METR_MEDIAN) {
      generate_dendrogram<true>(Z, Z2, N);
    }
    else {
      generate_dendrogram<false>(Z, Z2, N);
    }

  } // try
  catch (const std::bad_alloc&) {
    //fprintf(stderr, "Not enough Memory");
    return -1;
  }
  catch(const std::exception& e){
    //fprintf(stderr, "Uncaught exception");
    return -1;
  }
  catch(...){
    //fprintf(stderr, "C++ exception (unknown reason). Please send a bug report.");
    return -1;
  }
  return 0;

}
/*Clustering for the "stored data approach": the input are points in a vector space.*/
static int linkage_vector(t_float *X, int N, int dim, t_float * Z, unsigned char method, unsigned char metric) {

  //fprintf(stderr, "entering linkage_vector\n");
	//for (int ii=0; ii<8; ii++)
		//fprintf(stderr, " my vector %f \n", X[ii]);

  try{

    if (N < 1 ) {
      // N must be at least 1.
      //fprintf(stderr,"At least one element is needed for clustering.");
      return -1;
    }

    if (dim < 1 ) {
      //fprintf(stderr,"Invalid dimension of the data set.");
      return -1;
    }

    // (1)
    // The biggest index used below is 4*(N-2)+3, as an index to Z. This must fit
    // into the data type used for indices.
    // (2)
    // The largest representable integer, without loss of precision, by a floating
    // point number of type t_float is 2^T_FLOAT_MANT_DIG. Here, we make sure that
    // all cluster labels from 0 to 2N-2 in the output can be accurately represented
    // by a floating point number.
    //if (N > MAX_INDEX/4 || (N-1)>>(T_FLOAT_MANT_DIG-1) > 0) {
        //fprintf(stderr,"Data is too big, index overflow.");
    //  return -1;
    //}

    cluster_result Z2(N-1);

    auto_array_ptr<t_index> members;
    if (method==METHOD_METR_WARD || method==METHOD_METR_CENTROID) {
      members.init(2*N-1, 1);
    }

    if ((method!=METHOD_METR_SINGLE && metric!=METRIC_EUCLIDEAN) ||
        metric>=METRIC_INVALID) {
      //fprintf(stderr, "Invalid metric index.");
      return -1;
    }

    /*if (PyArray_ISBOOL(X)) {
      if (metric==METRIC_HAMMING) {
        metric = METRIC_MATCHING; // Alias
      }
      if (metric==METRIC_JACCARD) {
        metric = METRIC_JACCARD_BOOL;
      }
    }*/

    /* temp_point_array must be true if the alternative algorithm
       is used below (currently for the centroid and median methods). */
    bool temp_point_array = (method==METHOD_METR_CENTROID ||
                             method==METHOD_METR_MEDIAN);

    dissimilarity dist(X, N, dim, members, method, metric, temp_point_array);

    // TODO lluis: just convert the dist into a sparse matrix like you do with D (for the co_occurrence mat) <t_float *>DistMatrix, and then you can call :
    // 		   NN_chain_core<METHOD_METR_COMPLETE, t_index>(N, DistMatrix, NULL, Z2); (or whatever)

    if (method!=METHOD_METR_SINGLE &&
        method!=METHOD_METR_WARD &&
        method!=METHOD_METR_CENTROID &&
        method!=METHOD_METR_MEDIAN) {
      //fprintf(stderr, "Invalid method index.");
      return -1;
    }

    switch (method) {
    case METHOD_METR_SINGLE:
      //fprintf(stderr, " calling MST_linkage_core_vector %d \n", N);
      MST_linkage_core_vector(N, dist, Z2);
      break;
    case METHOD_METR_WARD:
      generic_linkage_vector<METHOD_METR_WARD>(N, dist, Z2);
      break;
    case METHOD_METR_CENTROID:
      generic_linkage_vector_alternative<METHOD_METR_CENTROID>(N, dist, Z2);
      break;
    default: // case METHOD_METR_MEDIAN:
      generic_linkage_vector_alternative<METHOD_METR_MEDIAN>(N, dist, Z2);
    }

    if (method==METHOD_METR_WARD ||
        method==METHOD_METR_CENTROID) {
      members.free();
    }

    dist.postprocess(Z2);

  //fprintf(stderr, " generating dendogram\n");
    if (method!=METHOD_METR_SINGLE) {
      generate_dendrogram<true>(Z, Z2, N);
    }
    else {
      generate_dendrogram<false>(Z, Z2, N);
    }

  } // try
  catch (const std::bad_alloc&) {
    //fprintf(stderr, "Not enough Memory");
    return -1;
  }
  catch(const std::exception& e){
    //fprintf(stderr, "Uncaught exception");
    return -1;
  }
  catch(...){
    //fprintf(stderr, "C++ exception (unknown reason). Please send a bug report.");
    return -1;
  }
  return 0;
}



// Just a main test function to test fastcluter lib. it compiles with:
// g++ -O3 -Wall -pedantic -ansi -Wconversion -Wsign-conversion -Wextra clustering.cpp -o clustering
/*
int  main()
{
	unsigned int N = 1113;
	t_float *X = (t_float*)malloc(2*N * sizeof(t_float));
	
	for (unsigned int i=0; i<N*2; i++)
		X[i] = rand() % 10 ;
		//fprintf(stderr, " my vector %f \n", X[i]);

	t_float *Z = (t_float*)malloc(((N-1)*4) * sizeof(t_float)); // we need 4 floats foreach obs.

	linkage_vector(X, (int)N, 2, Z, METHOD_METR_SINGLE, METRIC_EUCLIDEAN);
	//linkage_vector(X, (int)N, 2, Z, METHOD_METR_WARD, METRIC_EUCLIDEAN);
	//linkage_vector(X, (int)N, 2, Z, METHOD_METR_MEDIAN, METRIC_EUCLIDEAN);
	//linkage_vector(X, (int)N, 2, Z, METHOD_METR_CENTROID, METRIC_EUCLIDEAN);

	//fprintf(stderr, " \n\n");
	//for (unsigned int i=0; i<((N-1)*4); i++)
		//fprintf(stderr, " my dendogram %f \n", Z[i]);
	return 0;
}*/
