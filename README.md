## [ECCV 2022] MuLUT: Cooperating Mulitple Look-Up Tables for Efficient Image Super-Resolution
## [T-PAMI 2024] Toward DNN of LUTs: Learning Efficient Image Restoration with Multiple Look-Up Tables

[Jiacheng Li](http://ddlee-cn.github.io), Chang Chen, Zhen Cheng, and [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong)


[ECCV Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234.pdf) | [ECCV Paper Supp.](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234-supp.pdf) | [T-PAMI Paper](https://ieeexplore.ieee.org/document/10530442/) | [Poster](https://mulut.pages.dev/static/MuLUT-Poster-ECCV22.pdf) | [Project Page](https://mulut.pages.dev) | [Intro Video](https://youtu.be/xmvQYW7dtaE)

## News


2025.06 The extended version of LeRF is accepted by [IEEE T-PAMI](https://ieeexplore.ieee.org/document/11027639).

2025.06 The refactored ANDROID code of MuLUT is [open sourced](https://github.com/ddlee-cn/MuLUT-Android).

2025.05 My Ph.D. advisor, Prof. Zhiwei Xiong, gave a [talk (in Chinese)](https://ccig.csig.org.cn/2025/6874/list.html#:~:text=%E6%8A%A5%E5%91%8A%E9%A2%98%E7%9B%AE%EF%BC%9A%20%E5%9F%BA%E4%BA%8E%E5%8F%AF%E5%AD%A6%E4%B9%A0%E6%9F%A5%E6%89%BE%E8%A1%A8%E7%9A%84%E9%AB%98%E6%95%88%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86) on learned LUTs at Chinese Congress on Image and Graphics (CCIG) 2025.

2024.12 Our new work, [In-Loop Filtering via Trained Look-Up Tables(ILF-LUT)](https://ieeexplore.ieee.org/abstract/document/10849824) has been accepted by VCIP 2024. ILF-LUT extends learned LUTs to video coding, offering a way to integrate learned components into the video codec pipeline.

2024.06 Our new work, [Diagonal-First Compression for LUT(DFC)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Look-Up_Table_Compression_for_Efficient_Image_Restoration_CVPR_2024_paper.html) has been presented as a highlight paper at CVPR 2024. DFC reduce the storage requirement of restoration LUTs significantly (up to 10%) while preserving their performance.

2024.05 The extended version of MuLUT, DNN-of-LUT, is accepted by [IEEE T-PAMI](https://ieeexplore.ieee.org/document/10530442/).

2023.03 Extended version of MuLUT is available on [arxiv](https://arxiv.org/abs/2303.14506).

2023.02 Our new work, [Learning Resampling Function(LeRF)](https://lerf.pages.dev), is accepted by CVPR 2023. LeRF makes up for the regrets of MuLUT on replacing interpolation methods via achiving continuous resampling.

2022.10 MuLUT is published at ECCV 2022 and open sourced here.

## At A Glance

![MuLUT-ECCV-Github](./docs/MuLUT-At-A-Glance.png)

Please learn more at our [project page](https://mulut.pages.dev).

## Usage

### Code overview

In the `sr` directory, we provide the code of training MuLUT networks, transferring MuLUT network into LUts, finetuning LUTs, and testing LUTs, taking the task of single image super-resolution as an example. 

In the `common/network.py` file, we provide a universal implementation of MuLUT blocks, which can be constructed into MuLUT networks in a lego-like way.

### Dataset

Please following the instructions of [training](./data/DIV2K/README.md) and [testing](./data/SRBenchmark/README.md).

### Step 0: Installation

Clone this repo.

```
git clone https://github.com/ddlee-cn/MuLUT
```

Install requirements: `torch>=1.5.0`, `opencv-python`, `scipy`


### Step 1: Training MuLUT network

First, let us train a MuLUT network.

```
cd sr
python 1_train_model.py --stages 2 --modes sdy -e ../models/sr_x2sdy \
                        --trainDir ../data/DIV2K --valDir ../data/SRBenchmark
```
Our trained model and log are available under the `models/sr_x2sdy` directory.

### Step 2: Transfer MuLUT blocks into LUTs

Now, we are ready to cache the MuLUT network into multiple LUTs.

```
python 2_transfer_to_lut.py --stages 2 --modes sdy -e ../models/sr/x2sdy
```


### Step 3: Fine-tuning LUTs

```
python 3_finetune_lut.py --stages 2 --modes sdy -e ../models/sr_x2sdy \
                        --batchSize 256 --totalIter 2000 \ # when finetuning, the batch size can be larger, and the total training steps can be smaller 
                        --trainDir ../data/DIV2K --valDir ../data/SRBenchmark
```

After fine-tuning, LUTs are saved into `.npy` files and can be deployed on other devices. Our fine-tuned LUTs and log are available under the `models/sr_x2sdy` directory.


### Step 4: Test LUTs

Finally, we provide the following script to execute the LUT retrieval.

```
python 4_test_lut.py --stages 2 --modes sdy -e ../models/sr_x2sdy
```

The reference results for the `Set5` dataset are provided under the `results/sr_x2sdy` directory.


## Contact
If you have any questions, feel free to contact me by e-mail `jclee [at] mail.ustc.edu.cn`.


## Citation
If you find our work helpful, please cite the following papers.

```
@InProceedings{Li_2022_MuLUT,
      author    = {Li, Jiacheng and Chen, Chang and Cheng, Zhen and Xiong, Zhiwei},
      title     = {{MuLUT}: Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution},
      booktitle = {ECCV},
      year      = {2022},
  }
  
@ARTICLE{10530442,
  author={Li, Jiacheng and Chen, Chang and Cheng, Zhen and Xiong, Zhiwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Toward DNN of LUTs: Learning Efficient Image Restoration With Multiple Look-Up Tables}, 
  year={2024},
  volume={46},
  number={12},
  pages={8284-8301},
  doi={10.1109/TPAMI.2024.3401048}}


@InProceedings{Li_2023_LeRF,
      author    = {Li, Jiacheng and Chen, Chang and Huang, Wei and Lang, Zhiqiang and Song, Fenglong and Yan, Youliang and Xiong, Zhiwei},
      title     = {Learning Steerable Function for Efficient Image Resampling},
      booktitle = {CVPR},
      year      = {2023},
  }

@ARTICLE{11027639,
  author={Li, Jiacheng and Chen, Chang and Song, Fenglong and Yan, Youliang and Xiong, Zhiwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={LeRF: Learning Resampling Function for Adaptive and Efficient Image Interpolation}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2025.3577227}}


```


## License
MIT


## Acknowledgement

Our code is build upon [SR-LUT](https://github.com/yhjo09/SR-LUT).
