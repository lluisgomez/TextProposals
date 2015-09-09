# TextProposals

Implementation of the method proposed in the paper "Object Proposals for Text Extraction in the Wild" (Gomez & Karatzas), International Conference on Document Analysis and Recognition, ICDAR2015.

This code reproduce the results published on the paper for the SVT and ICDAR2013 datasets.

If you make use of this code, we appreciate it if you cite our paper:
```
@inproceedings{GomezICDAR15object,
  author    = {Lluis Gomez and Dimosthenis Karatzas},
  title     = {Object Proposals for Text Extraction in the Wild},
  booktitle = {ICDAR},
  year      = {2015}
}
```



For any questions please write us: ({lgomez,dimos}@cvc.uab.es). Thanks!


Includes the following third party code:

  - fast_clustering.cpp Copyright (c) 2011 Daniel Müllner, under the BSD license. http://math.stanford.edu/~muellner/fastcluster.html
  - voronoi.h voronoi_skeleton Copyright (c) 2013 by Arnaud Ramey, under the LGPL license. https://github.com/arnaud-ramey/voronoi
  - binomial coefficient approximations are due to Rafael Grompone von Gioi. http://www.ipol.im/pub/art/2012/gjmr-lsd/

## Compilation

Requires OpenCV 2.4.x (will not work with 3.0.x)

```
cmake .
make
```

## Run

  ``./img2hierarchy <img_filename> <mser_delta>``

writes to stdout a list of proposals, one per line, with the format: x,y,w,h,c.
where x,y,w,h define a bounding box, and c is a confidence value used to rank the proposals.

The value of "mse_delta" argument must be an integer value for the delta parameter of the MSER algorithm. In our experiments we use delta=13. Smaller values may generate better recall rates at the expenses of a lager number of proposals.

## Evaluation

The following command lines generate a txt file with proposals for each image in the SVT and ICDAR2013 datasets respectively.

  ``for i in `cat /path/to/datasets/SVT/svt1/test.xml | grep imageName | cut -d '>' -f 2 | cut -d '<' -f 1 | cut -d '/' -f 2 | cut -d '.' -f 1 `; do echo $i; ./img2hierarchy /path/to/datasets/SVT/svt1/img/$i.jpg 13 > data/$i; done;``

  ``for i in `cat /path/to/datasets/ICDAR2013/test_locations.xml | grep imageName | cut -d '>' -f 2 | cut -d '<' -f 1 | cut -d '_' -f 2 | cut -d '.' -f 1`; do echo $i; ./img2hierarchy /path/to/datasets/ICDAR2013/test/img_$i.jpg 13 > data/$i; done``

once the files are generated you may want to run the matlab code in the evaluation/ folder to get the IoU scores and plots.
