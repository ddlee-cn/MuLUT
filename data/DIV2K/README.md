
Please download DIV2K from their [official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

Then, put DIV2K dataset here in the following structure. In the first run, the dataset will be cached into numpy arrays. About 8GB GPU shared memory is needed to load all images.

```
dataset/DIV2K/
            /HR/*.png
            /LR/X2/*.png
            /LR/X3/*.png
            /LR/X4/*.png

```