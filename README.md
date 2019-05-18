# SAU-Net
This is the source for the paper: *SAU-Net: A Universal Deep Network for Cell Counting*.

Our U-Net implementation is based on https://github.com/jakeret/tf_unet and the training pipeline is based on https://github.com/erictzeng/adda.

### Dependencies
- tensorflow (1.13)
- numpy 
- click
- tqdm


### Data
All the four datasets used in the paper are provided for convenience in 
https://drive.google.com/drive/folders/1fD19kAhQi2IoGZNkDdB02dJThHilXVdr?usp=sharing

Download and put it in the root folder. The dot annotations are processed using `scipy.ndimage.gaussian_filter`.

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
The corresponding model weights and the perdition output can be found in `snapshot/`. The perdition output file `result_test_count` can be read using `shelve`. 