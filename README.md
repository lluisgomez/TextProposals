# TextProposals

Implementation of the method proposed in the papers:

* "TextProposals: a Text-specific Selective Search Algorithm for Word Spotting in the Wild" (Gomez and Karatzas), arXiv:1604.02619 2016.

* "Object Proposals for Text Extraction in the Wild" (Gomez & Karatzas), International Conference on Document Analysis and Recognition, ICDAR2015.

This code reproduces the results published on the papers for the SVT, ICDAR2013, ICDAR2015 datasets.

If you make use of this code, we appreciate it if you cite our papers:
```
@article{gomez2016,
  title     = {TextProposals: a Text-specific Selective Search Algorithm for Word Spotting in the Wild},
  author    = {Lluis Gomez and Dimosthenis Karatzas},
  journal   = {arXiv preprint arXiv:1604.02619},
  year      = {2016}
}
```

```
@inproceedings{GomezICDAR15object,
  title     = {Object Proposals for Text Extraction in the Wild},
  author    = {Lluis Gomez and Dimosthenis Karatzas},
  booktitle = {ICDAR},
  year      = {2015}
}
```

For any questions please write us: ({lgomez,dimos}@cvc.uab.es). Thanks!

Includes the following third party code:

  - fast_clustering.cpp Copyright (c) 2011 Daniel Müllner, under the BSD license. http://math.stanford.edu/~muellner/fastcluster.html
  - binomial coefficient approximations are due to Rafael Grompone von Gioi. http://www.ipol.im/pub/art/2012/gjmr-lsd/

## CNN models 

The end-to-end evaluation require the DictNet_VGG model to be placed in the project root directory.
DictNet_VGG Caffe model and prototxt are available here https://goo.gl/sNn5Xt

## Compilation

Requires: OpenCV (3.0.x), Caffe (tested with d21772c), tinyXML

```
cmake .
make
```

(NOTE: you may need to change the include and lib paths to your Caffe and cuda installations in CMakeLists.txt file)

## Run

  ``./img2hierarchy <img_filename>``

writes to stdout a list of proposals, one per line, with the format: x,y,w,h,c.
where x,y,w,h define a bounding box, and c is a confidence value used to rank the proposals.


  ``./img2hierarchy_cnn <img_filename>``

same as before but for end-to-end recognition using the DictNet_VGG CNN model.

## End-to-end Evaluation

The following commands reproduce end-to-end results in our paper:

  ``./eval_IC03 data/ICDAR2003/SceneTrialTest/words.xml <LEX_SIZE>``

  ``./eval_SVT data/SVT/test.xml <LEX_SIZE>``

  ``./eval_IC15 <LEX_SIZE>``

The value of LEX_SIZE parameter indicates the size of the lexicon to be used: 0 (for small lexicons), 1 (for Full lexicon), or 2 (for no lexicon, i.e. the 90k word vocabulary of the DictNet model).

Ground truth data for each dataset must be downloaded and placed in their respective folders in ./data/ directory.

In the case of ICDAR2015, since test ground truth is not available, the program save the results in res/ directory. These results files can be uploaded to the ICDAR Robust Reading Competition site for evaluation.

## Object Proposla Evaluation

The following command lines generate a txt file with proposals for each image in the SVT and ICDAR2013 datasets respectively.

  ``for i in `cat /path/to/datasets/SVT/svt1/test.xml | grep imageName | cut -d '>' -f 2 | cut -d '<' -f 1 | cut -d '/' -f 2 | cut -d '.' -f 1 `; do echo $i; ./img2hierarchy /path/to/datasets/SVT/svt1/img/$i.jpg 13 > data/$i; done;``

  ``for i in `cat /path/to/datasets/ICDAR2013/test_locations.xml | grep imageName | cut -d '>' -f 2 | cut -d '<' -f 1 | cut -d '_' -f 2 | cut -d '.' -f 1`; do echo $i; ./img2hierarchy /path/to/datasets/ICDAR2013/test/img_$i.jpg 13 > data/$i; done``

once the files are generated you may want to run the matlab code in the evaluation/ folder to get the IoU scores and plots.

Notice that the MATLAB evaluation script performs deduplicatioin of the bounding boxes proposals. Thus, if tou use another evauation framework you must deduplicate proposals same way.
