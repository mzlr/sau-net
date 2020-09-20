# SAU-Net
This is the source code for the paper, *SAU-Net: A Universal Network for Cell Counting in 2D and 3D Microscopy Images* (under review) and this paper is an extended version of our prior work [*SAU-Net: A Universal Deep Network for Cell Counting*](https://dl.acm.org/citation.cfm?id=3342153). 

Our 2D U-Net implementation is based on https://github.com/jakeret/tf_unet.

## Dependencies
- python 2.7
- tensorflow (1.15.2) 

## Data
All the five datasets used in the paper are provided for convenience in 
https://drive.google.com/drive/folders/1Ap91365akA1FkuWLv9k_DHt_EtrFlrbY?usp=sharing

Download the `data` folder and put it in the root folder, like this:
```
sau-net
  |-data
  |  |-VGG
  |  |-MBM
  ...
```
The dot annotations are processed using `scipy.ndimage.gaussian_filter`.

Original Datasets:
- [VGG](http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip)
- [MBM & ADI](https://github.com/ieee8023/countception)
- [DCC](https://github.com/markmarsden/DublinCellDataset)
- [MBC](https://github.com/nestorsaiz/saiz-et-al_2016)


## Run

From the root folder, run
```
bash run.sh [2D_dataset] [SELF_ATTN_FLAG] [GPU_ID] 
```
or 
```
bash run_3d.sh [3D_dataset] [SELF_ATTN_FLAG] [GPU_ID] 
```
For example, the following code will run on `vgg` dataset with Self-attention module using GPU $$0$$ (the default ID If only one GPU is available). 
```
bash run.sh vgg 1 0
```
Each time the training and test set will be randomly split by a random seed appended in the output folder. The corresponding model weights and the predictions can be found in `outputs/`. If you run into memory issues, consider using a smaller batch size, which can be found in the scripts, `run.sh` or `run_3d.sh`.

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
