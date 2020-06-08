# SAU-Net
This is the source code for the paper: [*SAU-Net: A Universal Deep Network for Cell Counting*](https://dl.acm.org/citation.cfm?id=3342153). 

Our U-Net implementation is based on https://github.com/jakeret/tf_unet and the training pipeline is based on https://github.com/erictzeng/adda.

### Dependencies
- tensorflow (1.13)
- numpy 
- click
- tqdm


### Data
All the four datasets used in the paper are provided for convenience in 
https://drive.google.com/drive/folders/1fD19kAhQi2IoGZNkDdB02dJThHilXVdr?usp=sharing

Download and put it in the root folder, like this:
```
sau-net
  |-adda
  |  |-data
  |  |-models
  |-scripts
  |-tools
  |-VGG
  |-MBM
  ...
```

The dot annotations are processed using `scipy.ndimage.gaussian_filter`.

Original Datasets:
- [VGG](http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip)
- [MBM & ADI](https://github.com/ieee8023/countception)
- [DCC](https://github.com/markmarsden/DublinCellDataset)


### Run

From the root folder, run
```
./scripts/run.sh [dataset] [iteration] [run] 
```
For example, the following code will run on `vgg` dataset with 350 iterations for 3 times. Each time the training and test set will be randomly split.
```
./scripts/run.sh vgg 350 3
```
The corresponding model weights and the perdition output can be found in `snapshot/`. The prediction output file `result_test_count` can be read using `shelve`. 

If you find this code useful in your research, please cite our paper:
```
@inproceedings{Guo:2019:SUD:3307339.3342153,
 author = {Guo, Yue and Stein, Jason and Wu, Guorong and Krishnamurthy, Ashok},
 title = {SAU-Net: A Universal Deep Network for Cell Counting},
 booktitle = {Proceedings of the 10th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},
 series = {BCB '19},
 year = {2019},
 isbn = {978-1-4503-6666-3},
 location = {Niagara Falls, NY, USA},
 pages = {299--306},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/3307339.3342153},
 doi = {10.1145/3307339.3342153},
 acmid = {3342153},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {cell counting, data augmentation, neural networks},
} 
```
