Please download the SR benchmark datasets following the instruction of [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets).

Then, put the downloaded SR benchmark datasets here as the follwing structure. `[testset]` can be `['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']`.

```
dataset/SRBenchmark/
                   /[testset]/HR/*.png
                             /LR_bicubic/X2/*.png
                   /...
```