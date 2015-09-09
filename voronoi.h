/*!
  \file        voronoi.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/9/11

________________________________________________________________________________

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
________________________________________________________________________________

The morphological skeleton of an image is the set of its non-zero pixels
which are equidistant to its boundaries.
More info: http://en.wikipedia.org/wiki/Topological_skeleton

Thinning an image consits in reducing its non-zero pixels to their
morphological skeleton.
More info: http://en.wikipedia.org/wiki/Thinning_(morphology)

\class VoronoiSkeleton is a C++ class
made for the fast computing of Voronoi diagrams of monochrome images.
It contains different implementations of thinning algorithms:

* Zhang - Suen
 explained in
 "A fast parallel algorithm for thinning digital patterns" by T.Y. Zhang and C.Y. Suen.
 and based on implentation by
 \link http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/

* Guo - Hall:
 explained in
 "Parallel thinning with two sub-iteration algorithms" by Zicheng Guo and Richard Hall.
 and based on implentation by
 \link http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/

* a morphological one, based on the erode() and dilate() operators.
  Coming from:
  \link http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

A special care has been given to optimize the 2 first ones.
Instead of re-examining the whole image at each iteration,
only the pixels of the current contour are considered.

This leads to a speedup by almost 100 times on experimental tests.

 */

#ifndef VORONOI_H
#define VORONOI_H

#include <deque>
#include <opencv2/imgproc/imgproc.hpp>
#include "image_contour.h"

#define IMPL_MORPH                "morph"
#define IMPL_ZHANG_SUEN           "zhang_suen"
#define IMPL_ZHANG_SUEN_ORIGINAL  "zhang_suen_original"
#define IMPL_ZHANG_SUEN_FAST      "zhang_suen_fast"
#define IMPL_GUO_HALL             "guo_hall"
#define IMPL_GUO_HALL_ORIGINAL    "guo_hall_original"
#define IMPL_GUO_HALL_FAST        "guo_hall_fast"

class VoronoiSkeleton {
public:
  //! a constant not limiting the number of iterations of an implementation
  static const int NOLIMIT = INT_MAX;

  //! default construtor
  VoronoiSkeleton() {
    _has_converged = false;
    element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * thin a given image,
   * \param img
   *  A monochrome image. All pixels > 0 are considered as part of the shape.
   * \param implementation_name
   *  One of the supported implementations.
   *  \see all_implementations()
   * \param crop_img_before
   *  true to crop the image to its bounding box before thinning.
   *  Leads to a considerable speedup if the non-zero content of the image
   *  represents only a part of the image.
   * \param max_iters
   *    limit the number of iterations on the image,
   *    NOLIMIT to let the implementation converge
   * \return
   *    true if success
   *    false if \a implementation_name is not a supported implementation
   */
  inline bool thin(const cv::Mat1b & img,
                   const std::string & implementation_name,
                   bool crop_img_before = true,
                   int max_iters = NOLIMIT) {
    if (implementation_name == IMPL_MORPH)
      return thin_morph(img, crop_img_before, max_iters);
    else if (implementation_name == IMPL_ZHANG_SUEN_ORIGINAL)
      return thin_zhang_suen_original(img, crop_img_before, max_iters);
    else if (implementation_name == IMPL_ZHANG_SUEN)
      return thin_zhang_suen(img, crop_img_before, max_iters);
    else if (implementation_name == IMPL_ZHANG_SUEN_FAST)
      return thin_zhang_suen_fast(img, crop_img_before, max_iters);
    else if (implementation_name == IMPL_GUO_HALL_ORIGINAL)
      return thin_guo_hall_original(img, crop_img_before, max_iters);
    else if (implementation_name == IMPL_GUO_HALL)
      return thin_guo_hall(img, crop_img_before, max_iters);
    else if (implementation_name == IMPL_GUO_HALL_FAST)
      return thin_guo_hall_fast(img, crop_img_before, max_iters);
    else {
      printf("Unknow implementation '%s', supported implementations: [%s]\n",
             implementation_name.c_str(), all_implementations_as_string().c_str());
      return false;
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! \return true if last thin() stopped because the algo converged,
   * and not because of the max_iters param.
   */
  inline bool has_converged() const { return _has_converged; }

  //////////////////////////////////////////////////////////////////////////////

  /*! \return the current skeleton
   * Call thin() before accessing it.
   * All non zero pixels correspond to the morphological skeleton of the image
   */
  inline const cv::Mat1b & get_skeleton() const {
    return skel;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! \return the bounding box used during thin().
   * Call thin() before accessing it.

   * Equals to the bounding box of the image if crop_img_before was true,
   * or the full image if crop_img_before was false.
   */
  inline cv::Rect get_bbox() const {
    return bbox;
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return true if \arg implementation is among the existing ones
  static inline bool is_implementation_valid(const std::string & implementation) {
    std::vector<std::string> impls = all_implementations();
    return (std::find(impls.begin(), impls.end(), implementation)
            != impls.end());
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return the list of all supported implementations
  static inline std::vector<std::string> all_implementations() {
    std::vector<std::string> out;
    out.push_back(IMPL_MORPH);
    out.push_back(IMPL_GUO_HALL);
    out.push_back(IMPL_GUO_HALL_ORIGINAL);
    out.push_back(IMPL_GUO_HALL_FAST);
    out.push_back(IMPL_ZHANG_SUEN);
    out.push_back(IMPL_ZHANG_SUEN_ORIGINAL);
    out.push_back(IMPL_ZHANG_SUEN_FAST);
    return out;
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return a comma seperated string of the list of all supported implementations
  static inline std::string all_implementations_as_string() {
    std::vector<std::string> impls = all_implementations();
    std::ostringstream out;
    for (unsigned int i = 0; i < impls.size(); ++i)
      out << impls[i] << (i == impls.size() - 1 ? "" : ", ");
    return out.str();
  }

  //////////////////////////////////////////////////////////////////////////////

  //! get the bounding box of the non-zero pixels of an image + a border of one pixel
  static inline cv::Rect bounding_box_plusone(const cv::Mat1b& img) {
    cv::Rect bbox = boundingBox(img);
    // printf("bbox:(%i, %i)+(%i, %i)\n", bbox.x, bbox.y, bbox.width, bbox.height);
    // the top and left boundary of the rectangle are inclusive,
    // while the right and bottom boundaries are not
    if (bbox.x <= 0 || bbox.x + bbox.width >= img.cols
        || bbox.y <= 0 || bbox.y + bbox.height >= img.rows) {
      printf("img:(%i, %i), bbox:(%i, %i)+(%i, %i) "
             " does not have a border of 1 pixel, using whole pic.\n",
             img.cols, img.rows, bbox.x, bbox.y, bbox.width, bbox.height);
      return bounding_box_full_img(img);
    }
    // add a border of one pixel
    bbox.x --;
    bbox.y --;
    bbox.width += 2;
    bbox.height += 2;
    return bbox;
  } // end bounding_box_plusone()

  //////////////////////////////////////////////////////////////////////////////

  //! copy the non zero content of \a img to \a out
  static inline cv::Rect copy_bounding_box_plusone(const cv::Mat1b& img,
                                                   cv::Mat1b& out) {
    cv::Rect bbox = bounding_box_plusone(img);
    // printf("bbox:(%i, %i)+(%i, %i)\n", bbox.x, bbox.y, bbox.width, bbox.height);
    img(bbox).copyTo(out);
    return bbox;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

protected:
  //////////////////////////////////////////////////////////////////////////////

  /*! From content_processing.h
 *\brief   get the bounding box of the non null points of an image
 *\param   img a monochrome image
 *\return the bounding box of the non null points,
 *        cv::Rect(-1, -1, -1, -1) if the image is empty
 */
  template<class _T>
  static inline cv::Rect boundingBox(const cv::Mat_<_T> & img) {
    assert(img.isContinuous());
    int xMin = 0, yMin = 0, xMax = 0, yMax = 0;
    bool was_init = false;
    const _T* img_it = img.ptr(0);
    int nrows = img.rows, ncols = img.cols;
    for (int y = 0; y < nrows; ++y) {
      for (int x = 0; x < ncols; ++x) {
        if (*img_it++) {
          if (!was_init) {
            xMin = xMax = x;
            yMin = yMax = y;
            was_init = true;
            continue;
          }
          if (x < xMin)
            xMin = x;
          else if (x > xMax)
            xMax = x;

          if (y < yMin)
            yMin = y;
          else if (y > yMax)
            yMax = y;
        }
      } // end loop x
    } // end loop y

    if (!was_init) // no white point found
      return cv::Rect(-1, -1, -1, -1);
    // from http://docs.opencv.org/java/org/opencv/core/Rect.html
    // OpenCV typically assumes that the top and left boundary of the rectangle
    // are inclusive, while the right and bottom boundaries are not.
    // For example, the method Rect_.contains returns true if
    // x <= pt.x < x+width,   y <= pt.y < y+height
    return cv::Rect(xMin, yMin, 1 + xMax - xMin,  1 + yMax - yMin);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return a rectangle correspondign to the size of the whole image
  static inline cv::Rect bounding_box_full_img(const cv::Mat1b& img) {
    return cv::Rect(0, 0, img.cols, img.rows);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! from \link http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
  bool thin_morph(const cv::Mat1b & img,
                  bool crop_img_before = true,
                  int max_iters = NOLIMIT) {
    if (crop_img_before) {
      copy_bounding_box_plusone(img, img_copy);
    }
    else {
      bbox = bounding_box_full_img(img);
      img.copyTo(img_copy);
    }
    skel.create(img_copy.size());
    skel.setTo(0);
    temp.create(img_copy.size());

    bool done = false;
    int niters = 0;
    while(!done) {
      cv::erode(img_copy, eroded, element);
      cv::dilate(eroded, dilated, element); // dilated = open(img)
      cv::subtract(img_copy, dilated, temp);
      cv::bitwise_or(skel, temp, skel);
      eroded.copyTo(img_copy);
      done = (cv::countNonZero(img_copy) == 0);
      // cv::imshow("skel", skel); cv::waitKey(0);
      if ((niters++) >= max_iters) // must be at the end of the loop
        break;
    }
    //printf("niters:%i\n", niters);
    _has_converged = done;
    return true;
  } // end from_img();

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Function for thinning the given binary image
   * From \link http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
   *
   * \param  im  Binary image with range = 0-255
   */
  bool thin_zhang_suen_original(const cv::Mat& img,
                                bool crop_img_before = true,
                                int max_iters = NOLIMIT) {
    // im /= 255;
    if (crop_img_before) {
      cv::threshold(img, temp, 10, 1, CV_THRESH_BINARY);
      bbox  = copy_bounding_box_plusone(temp, skel);
    }
    else {
      bbox = bounding_box_full_img(img);
      cv::threshold(img, skel, 10, 1, CV_THRESH_BINARY);
    }

    cv::Mat prev = cv::Mat::zeros(skel.size(), CV_8UC1);
    cv::Mat diff;

    int niters = 0;
    do {
      thin_zhang_suen_original_iter(skel, 0);
      thin_zhang_suen_original_iter(skel, 1);
      cv::absdiff(skel, prev, diff);
      skel.copyTo(prev);
      if ((niters++) >= max_iters) // must be at the end of the loop
        break;
    }
    while (cv::countNonZero(diff) > 0);

    skel *= 255;
    _has_converged = (niters < max_iters);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Function for thinning the given binary image
   * From \link http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
   *
   * \param  im  Binary image with range = 0-255
   */
  bool thin_zhang_suen(const cv::Mat1b& img,
                       bool crop_img_before = true,
                       int max_iters = NOLIMIT) {
    //im /= 255;
    // marker values need to be 0 or 1 for multiplications of values to make sense
    if (crop_img_before) {
      cv::threshold(img, temp, 10, 1, CV_THRESH_BINARY);
      bbox  = copy_bounding_box_plusone(temp, skel);
    }
    else {
      bbox = bounding_box_full_img(img);
      cv::threshold(img, skel, 10, 1, CV_THRESH_BINARY);
    }

    int niters = 0;
    while (true) {
      bool haschanged1 = thin_zhang_suen_iter(skel, 0);
      //printf("0\n"); skel *= 255; cv::imshow("skel", skel); cv::waitKey(0); skel /= 255;
      bool haschanged2 = thin_zhang_suen_iter(skel, 1);
      //printf("1\n"); skel *= 255; cv::imshow("skel", skel); cv::waitKey(0); skel /= 255;
      if (!haschanged1 && !haschanged2)
        break;
      if ((niters++) >= max_iters) // must be at the end of the loop
        break;
    }
    skel *= 255;
    _has_converged = (niters < max_iters);
    return true;
  } // end thin_zhang_suen();

  //////////////////////////////////////////////////////////////////////////////

  inline bool thin_zhang_suen_fast(const cv::Mat1b& img,
                                   bool crop_img_before = true,
                                   int max_iters = NOLIMIT) {
    return thin_fast_custom_voronoi_fn(img, need_set_zhang_suen, crop_img_before, max_iters);
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * From \link http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
   * Function for thinning the given binary image
   *
   * \param  im  Binary image with range = 0-255
   */
  bool thin_guo_hall_original(const cv::Mat1b& img,
                              bool crop_img_before = true,
                              int max_iters = NOLIMIT) {
    // skel /= 255;
    if (crop_img_before) {
      cv::threshold(img, temp, 10, 1, CV_THRESH_BINARY);
      bbox  = copy_bounding_box_plusone(temp, skel);
    }
    else {
      bbox = bounding_box_full_img(img);
      cv::threshold(img, skel, 10, 1, CV_THRESH_BINARY);
    }

    cv::Mat prev = cv::Mat::zeros(skel.size(), CV_8UC1);
    cv::Mat diff;

    int niters = 0;
    do {
      thin_guo_hall_original_iter(skel, 0);
      thin_guo_hall_original_iter(skel, 1);
      cv::absdiff(skel, prev, diff);
      skel.copyTo(prev);
      if ((niters++) >= max_iters) // must be at the end of the loop
        break;
    }
    while (cv::countNonZero(diff) > 0);

    skel *= 255;
    _has_converged = (niters < max_iters);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Function for thinning the given binary image
   * From \link http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
   *
   * \param  im  Binary image with range = 0-255
   */
  bool thin_guo_hall(const cv::Mat1b& img,
                     bool crop_img_before = true,
                     int max_iters = NOLIMIT) {
    //im /= 255;
    if (crop_img_before) {
      cv::threshold(img, temp, 10, 1, CV_THRESH_BINARY);
      bbox  = copy_bounding_box_plusone(temp, skel);
      // std::cout << "temp" << ImageContour::to_string(temp) << std::endl;
      // std::cout << "skel" << ImageContour::to_string(skel) << std::endl;
    }
    else {
      bbox = bounding_box_full_img(img);
      cv::threshold(img, skel, 10, 1, CV_THRESH_BINARY);
    }

    int niters = 0;
    while (true) {
      bool haschanged1 = thin_guo_hall_iter(skel, 0);
      // printf("0\n"); skel *= 255; cv::imshow("skel", skel); cv::waitKey(0); skel /= 255;
      // std::cout << "iter0: skel:" << ImageContour::to_string(skel) << std::endl;

      bool haschanged2 = thin_guo_hall_iter(skel, 1);
      // printf("1\n"); skel *= 255; cv::imshow("skel", skel); cv::waitKey(0); skel /= 255;
      // std::cout << "iter1: skel:" << ImageContour::to_string(skel) << std::endl;
      if (!haschanged1 && !haschanged2)
        break;
      if ((niters++) >= max_iters) // must be at the end of the loop
        break;
    }
    skel *= 255;
    _has_converged = (niters < max_iters);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline bool thin_guo_hall_fast(const cv::Mat1b& img,
                                 bool crop_img_before = true,
                                 int max_iters = NOLIMIT) {
    return thin_fast_custom_voronoi_fn(img, need_set_guo_hall, crop_img_before, max_iters);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return true if skel needs to be set to 0
  typedef bool (*VoronoiFn)(uchar*  skeldata, int iter, int col, int row, int cols);

  bool thin_fast_custom_voronoi_fn(const cv::Mat1b& img,
                                   VoronoiFn voronoi_fn,
                                   bool crop_img_before = true,
                                   int max_iters = NOLIMIT) {
    //  printf("thin_fast_custom_voronoi_fn(crop_img_before:%i, max_iters:%i)\n",
    //         crop_img_before, max_iters);
    // marker values need to be 0 or 1 for multiplications of values to make sense
    if (crop_img_before) {
      //      cv::threshold(img, temp, 10, 1, CV_THRESH_BINARY);
      bbox  = copy_bounding_box_plusone(img, skel);
      skelcontour.from_image_C4(skel);
    }
    else {
      bbox = bounding_box_full_img(img);
      //      cv::threshold(img, skel, 10, 1, CV_THRESH_BINARY);
      skelcontour.from_image_C4(img);
    }
    //printf("skelcontour:'%s'\n", skelcontour.to_string().c_str());
    int cols = skelcontour.cols, rows = skelcontour.rows;

    // clear queues
    uchar * skelcontour_data = skelcontour.data;

    int niters = 0;
    bool change_made = true;
    while (change_made && niters < max_iters) {
      //printf("loop\n");
      change_made = false;
      for (unsigned short iter = 0; iter < 2; ++iter) {
        //printf("loop iter\n");
        uchar *skelcontour_ptr = skelcontour_data;
        rows_to_set.clear();
        cols_to_set.clear();
        // for each point in skelcontour, check if it needs to be changed
        for (int row = 0; row < rows; ++row) {
          for (int col = 0; col < cols; ++col) {
            //printf("Checking (%i, %i)...\n", col, row);
            if (*skelcontour_ptr++ == ImageContour::CONTOUR &&
                voronoi_fn(skelcontour_data, iter, col, row, cols)) {
              //printf("(%i, %i) is to be removed\n", col, row);
              cols_to_set.push_back(col);
              rows_to_set.push_back(row);
            }
          } // end for (col)
        } // end for (row)

        // set all points in rows_to_set (of skel)
        unsigned int rows_to_set_size = rows_to_set.size();
        for (unsigned int pt_idx = 0; pt_idx < rows_to_set_size; ++pt_idx) {
          if (!change_made)
            change_made = (skelcontour(rows_to_set[pt_idx], cols_to_set[pt_idx]));
          skelcontour.set_point_empty_C4(rows_to_set[pt_idx], cols_to_set[pt_idx]);
        } // end for (pt_idx)

#if 0 // debug info
        //std::cout << "skel:" << std::endl << skel << std::endl;
        printf("iter:%i, rows_to_set.size():%i\n", iter, rows_to_set.size());
        printf("iter:%i, skelcontour.contour_size():%i\n", iter, skelcontour.contour_size());
        printf("iter:%i, skelcontour:%s\n", iter, skelcontour.to_string().c_str());
        cv::imshow("skelcontour", skelcontour);
        cv::waitKey(0);
#endif
        if ((niters++) >= max_iters) // must be at the end of the loop
          break;
      } // end for (iter)
    } // end while (true)

    skel = (skelcontour != ImageContour::EMPTY);
    _has_converged = !change_made;
    return true;
  } // end thin_fast_custom_voronoi_fn();

  //////////////////////////////////////////////////////////////////////////////

  static bool inline need_set_zhang_suen(uchar*  skeldata, int iter, int col, int row, int cols) {
    bool p2 = skeldata[(row-1) * cols + col];
    bool p3 = skeldata[(row-1) * cols + col+1];
    bool p4 = skeldata[row     * cols + col+1];
    bool p5 = skeldata[(row+1) * cols + col+1];
    bool p6 = skeldata[(row+1) * cols + col];
    bool p7 = skeldata[(row+1) * cols + col-1];
    bool p8 = skeldata[row     * cols + col-1];
    bool p9 = skeldata[(row-1) * cols + col-1];
    int A  = (!p2 && p3) + (!p3 && p4) +
             (!p4 && p5) + (!p5 && p6) +
             (!p6 && p7) + (!p7 && p8) +
             (!p8 && p9) + (!p9 && p2);
    int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
    int m1 = (iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8));
    int m2 = (iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8));
    return (A == 1 && (B >= 2 && B <= 6) && !m1 && !m2);
  }

  //////////////////////////////////////////////////////////////////////////////

  static bool inline need_set_guo_hall(uchar*  skeldata, int iter, int col, int row, int cols) {
    //uchar
    bool
        p2 = skeldata[(row-1) * cols + col],
        p3 = skeldata[(row-1) * cols + col+1],
        p4 = skeldata[row     * cols + col+1],
        p5 = skeldata[(row+1) * cols + col+1],
        p6 = skeldata[(row+1) * cols + col],
        p7 = skeldata[(row+1) * cols + col-1],
        p8 = skeldata[row     * cols + col-1],
        p9 = skeldata[(row-1) * cols + col-1];

    int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
             (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
    int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
    int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
    int N  = N1 < N2 ? N1 : N2;
    int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

    return (C == 1 && (N >= 2 && N <= 3) && m == 0);
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Perform one thinning iteration.
   * Normally you wouldn't call this function directly from your code.
   *
   * \param  im    Binary image with range = 0-1
   * \param  iter  0=even, 1=odd
   */
  void thin_zhang_suen_original_iter(cv::Mat& im, int iter)
  {
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
      for (int j = 1; j < im.cols-1; j++)
      {
        uchar p2 = im.at<uchar>(i-1, j);
        uchar p3 = im.at<uchar>(i-1, j+1);
        uchar p4 = im.at<uchar>(i, j+1);
        uchar p5 = im.at<uchar>(i+1, j+1);
        uchar p6 = im.at<uchar>(i+1, j);
        uchar p7 = im.at<uchar>(i+1, j-1);
        uchar p8 = im.at<uchar>(i, j-1);
        uchar p9 = im.at<uchar>(i-1, j-1);

        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                 (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
        int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
        int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
          marker.at<uchar>(i,j) = 1;
      }
    }

    im &= ~marker;
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Perform one thinning iteration.
   * Normally you wouldn't call this function directly from your code.
   *
   * \param  im    Binary image with range = 0-1
   * \param  iter  0=even, 1=odd
   */
  bool thin_zhang_suen_iter(cv::Mat1b& im, int iter) {
    bool haschanged = false;
    assert(im.isContinuous());
    uchar*  imdata = im.data;
    im.copyTo(temp);
    assert(temp.isContinuous());
    uchar*  tempdata = temp.data;
    unsigned int cols = im.cols, colmax = im.cols -1, rowmax = im.rows - 1;
    for (unsigned int row = 1; row < rowmax; row++) {
      for (unsigned int col = 1; col < colmax; col++) {
#if 0
        bool p2 = imdata[(row-1) * cols + col];
        bool p3 = imdata[(row-1) * cols + col+1];
        bool p4 = imdata[row     * cols + col+1];
        bool p5 = imdata[(row+1) * cols + col+1];
        bool p6 = imdata[(row+1) * cols + col];
        bool p7 = imdata[(row+1) * cols + col-1];
        bool p8 = imdata[row     * cols + col-1];
        bool p9 = imdata[(row-1) * cols + col-1];
        int A  = (!p2 && p3) + (!p3 && p4) +
                 (!p4 && p5) + (!p5 && p6) +
                 (!p6 && p7) + (!p7 && p8) +
                 (!p8 && p9) + (!p9 && p2);
        int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        int m1 = (iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8));
        int m2 = (iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8));

        if (A == 1 && (B >= 2 && B <= 6) && !m1 && !m2) {
#else
        if (need_set_zhang_suen(imdata, iter, col, row, cols)) {
#endif
          if (!haschanged)
            haschanged = (tempdata[row * cols +col] != 0);
          tempdata[row * cols +col] = 0;
        }
      } // end loop col
    } // end loop row

    std::swap(im, temp);
    return haschanged;
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Perform one thinning iteration.
   * Normally you wouldn't call this function directly from your code.
   *
   * \param  im    Binary image with range = 0-1
   * \param  iter  0=even, 1=odd
   */
  bool thin_guo_hall_iter(cv::Mat1b& im, int iter) {
    bool haschanged = false;
    assert(im.isContinuous());
    uchar*  imdata = im.data;
    im.copyTo(temp);
    assert(temp.isContinuous());
    uchar*  tempdata = temp.data;
    unsigned int cols = im.cols, colmax = im.cols -1, rowmax = im.rows - 1;
    for (unsigned int row = 1; row < rowmax; row++) {
      for (unsigned int col = 1; col < colmax; col++) {
#if 0
        uchar p2 = imdata[(row-1) * cols + col];
        uchar p3 = imdata[(row-1) * cols + col+1];
        uchar p4 = imdata[row     * cols + col+1];
        uchar p5 = imdata[(row+1) * cols + col+1];
        uchar p6 = imdata[(row+1) * cols + col];
        uchar p7 = imdata[(row+1) * cols + col-1];
        uchar p8 = imdata[row     * cols + col-1];
        uchar p9 = imdata[(row-1) * cols + col-1];

        int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                 (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
        int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
        int N  = N1 < N2 ? N1 : N2;
        int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

        if (C == 1 && (N >= 2 && N <= 3) && m == 0) {
#else
        if (need_set_guo_hall(imdata, iter, col, row, cols)) {
#endif
          if (!haschanged)
            haschanged = (tempdata[row * cols +col] != 0);
          tempdata[row * cols +col] = 0;
        }
      } // end loop col
    } // end loop row

    std::swap(im, temp);
    return haschanged;
  }

  //////////////////////////////////////////////////////////////////////////////

  /**
   * From \link http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
   * Perform one thinning iteration.
   * Normally you wouldn't call this function directly from your code.
   *
   * \param  im    Binary image with range = 0-1
   * \param  iter  0=even, 1=odd
   */
  void thin_guo_hall_original_iter(cv::Mat& im, int iter) {
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows; i++)
    {
      for (int j = 1; j < im.cols; j++)
      {
        uchar p2 = im.at<uchar>(i-1, j);
        uchar p3 = im.at<uchar>(i-1, j+1);
        uchar p4 = im.at<uchar>(i, j+1);
        uchar p5 = im.at<uchar>(i+1, j+1);
        uchar p6 = im.at<uchar>(i+1, j);
        uchar p7 = im.at<uchar>(i+1, j-1);
        uchar p8 = im.at<uchar>(i, j-1);
        uchar p9 = im.at<uchar>(i-1, j-1);

        int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                 (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
        int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
        int N  = N1 < N2 ? N1 : N2;
        int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

        if (C == 1 && (N >= 2 && N <= 3) && m == 0)
          marker.at<uchar>(i,j) = 1;
      }
    }

    im &= ~marker;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  cv::Rect bbox;
  cv::Mat1b skel;
  // geometric algo
  cv::Mat1b img_copy;
  cv::Mat1b temp;
  cv::Mat eroded;
  cv::Mat dilated;
  cv::Mat element;
  bool _has_converged;
  // Zhang-Suen
  // Guo Hall
  //cv::Mat1b marker;
  std::deque<int> marker_;
  // Zhang-Suen fast
  ImageContour skelcontour;
  //! list of keys to set to 0 at the end of the iteration
  std::deque<int> cols_to_set;
  std::deque<int> rows_to_set;
}; // end class VoronoiSkeleton

#endif // VORONOI_H
