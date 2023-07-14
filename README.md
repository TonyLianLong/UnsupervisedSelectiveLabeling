# Unsupervised Selective Labeling for More Effective Semi-Supervised Learning
by [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Long Lian*](https://tonylian.com/), and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley/ICSI. (*: co-first authors)

[Arxiv Paper](https://arxiv.org/abs/2110.03006) | [ECCV Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900423.pdf) | [**Poster**](https://people.eecs.berkeley.edu/~longlian/usl_poster.pdf) | [**Video**](https://people.eecs.berkeley.edu/~longlian/usl_video.html) | [Citation](#citation)

*European Conference on Computer Vision* (ECCV), 2022.

This work is also presented in [CV in the Wild workshop](https://computer-vision-in-the-wild.github.io/eccv-2022/) in ECCV 2022.

![Teaser](assets/teaser.png)

This repository contains the code for USL on CIFAR. Other implementations are coming soon.

For further information regarding the paper, please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu). For information regarding the code and implementation, please contact [Long Lian](mailto:longlian@berkeley.edu).

## News
* **USL-T is fully supported** in this implementation. Pretraining weights and trained models are available.
* Selected sample indices on USL-T are added for reference (note that USL-T is training-based and thus gives different results on different runs)
* **Poster** and **video** are added (see above)
* ImageNet scripts, intermediate results, final results, and FixMatch checkpoints are added
* Provided CLD pretrained model and reference selections
* Initial Implementation

## Supported Methods
- [x] USL
- [x] USL-T

## Supported SSL Methods
- [x] FixMatch
- [x] SimCLRv2
- [x] SimCLRv2-CLD

## Supported Datasets
- [x] CIFAR-10
- [x] CIFAR-100
- [x] ImageNet100
- [x] ImageNet

## Preparation
Install the required packages:
```
pip install -r requirements.txt
```

You also need to install `clip` if you want to use `clip` models:
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

For ImageNet, you need to change the data path to your data path in the config. For CIFAR, it will download the data automatically.

## Run USL
### CIFAR-10/100
#### Download our CLD pretrained model on CIFAR for USL
```shell
mkdir selective_labeling/pretrained
cd selective_labeling/pretrained
# CLD checkpoint on CIFAR-10
wget https://people.eecs.berkeley.edu/~longlian/files/cifar10_ckpt_epoch_200.pth
# CLD checkpoint on CIFAR-100
wget https://people.eecs.berkeley.edu/~longlian/files/cifar100_ckpt_epoch_200.pth
```

#### Perform USL on CIFAR-10/100
```shell
cd selective_labeling
# CIFAR-10
python usl-cifar.py --cfg configs/cifar10_usl.yaml
# CIFAR-100
python usl-cifar.py --cfg configs/cifar100_usl.yaml
```

#### Evaluate USL on CIFAR-10/100 with SimCLRv2-CLD
You can also find the config for SimCLR in `semisup-simclrv2-cld/configs`. The implementation for FixMatch is in `semisup-fixmatch-cifar`.
```shell
cd semisup-simclrv2-cld
# CIFAR-10
python fine_tune.py --cfg semisup-simclrv2-cld/configs/cifar10_usl-t_finetune.yaml
# CIFAR-100
python fine_tune.py --cfg semisup-simclrv2-cld/configs/cifar100_usl-t_finetune.yaml
```

### ImageNet
#### Download our CLD pretrained model on ImageNet for USL (USL-MoCo only)
```
mkdir selective_labeling/pretrained
cd selective_labeling/pretrained
# MoCov2 checkpoint on ImageNet (with EMAN as normalization)
wget https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_800ep.pth.tar
```

CLIP models will be downloaded at the first run with USL-CLIP config.

#### Use pre-computed intermediate results (Recommended)
This step is optional but recommended to refrain from recomputing the feature from the dataset. Furthermore, it relieves the non-deterministic behavior from obtaining feature, kNN, and clustering, since GPU ops lead to non-deterministic behavior (even though seed is set), which is more prominent for large datasets where more compute is used.

We provide intermediate results after obtaining feature, kNN, k-Means, and the final selected indices in numpy and csv format. Intermediate results can be obtained by:
<details>
<summary>USL-MoCo experiments</summary>

```
mkdir -p selective_labeling/saved/imagenet_usl_moco_0.2
cd selective_labeling/saved/imagenet_usl_moco_0.2
wget https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet_moco_0.2_intermediate.zip
unzip usl_imagenet_moco_0.2_intermediate.zip
cd ../../..
```

Please also download the precomputed MoCov2 feature [here](https://drive.google.com/file/d/1r8hJ_tuQ7Eta2eVTmZQT1W59FzLkvDZ3/view?usp=share_link) and unzip `memory_feats_list.npy` into `selective_labeling/saved/imagenet_usl_moco_0.2`.

</details>

<details>
<summary>USL-CLIP experiments</summary>

```
mkdir -p selective_labeling/saved/imagenet_usl_clip_0.2
cd selective_labeling/saved/imagenet_usl_clip_0.2
wget https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet_clip_0.2_intermediate.zip
unzip usl_imagenet_clip_0.2_intermediate.zip
cd ../../..
```

Please also download the precomputed CLIP feature [here](https://drive.google.com/file/d/1V47BFvWs9uQYO_sOGDqj3RrslMwjaEsO/view?usp=share_link) and unzip `memory_feats_list.npy` into `selective_labeling/saved/imagenet_usl_clip_0.2`.

</details>

You can also obtain the intermediate and final results [here](https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet.html).

If this step is skipped (i.e., compute everything from scratch), `RECOMPUTE_ALL` and `RECOMPUTE_NUM_DEP` need to be set to True in the config.

#### Perform USL on ImageNet
##### USL-MoCo experiments
```
cd selective_labeling
python usl-imagenet.py
```

##### USL-CLIP experiments
```
cd selective_labeling
python usl-imagenet.py --cfg configs/ImageNet_usl_clip_0.2.yaml
```

#### Evaluate USL on ImageNet with FixMatch
Use the script [here](https://github.com/amazon-science/exponential-moving-average-normalization). The output `csv` file is compatible with the split of labeled dataset.

## Run USL-T
To make our results more comparable, we release our selected indices and trained models, see [model zoo](#model-zoo) below.
### CIFAR-10/100
#### Download models trained with unsupervised learning methods
```
mkdir -p usl-t_pretraining/pretrained
wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_cifar10.tgz
tar zxvf pretrained_kNN_cifar10.tgz

wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_cifar100.tgz
tar zxvf pretrained_kNN_cifar100.tgz
```
#### Perform USL-T (training and sample selection)
* USL-T Training (also called pretraining because it is prior to training in semi-supervised learning):
```
cd usl-t_pretraining
# CIFAR-10
python usl-t-cifar-pretrain.py --cfg configs/cifar10_usl-t_pretrain.yaml
# CIFAR-100
python usl-t-cifar-pretrain.py --cfg configs/cifar100_usl-t_pretrain.yaml
cd ..
```
* Sample selection:
```
cd ../selective_labeling
# CIFAR-10
python usl-t-cifar.py --cfg configs/cifar10_usl-t.yaml
# CIFAR-100
python usl-t-cifar.py --cfg configs/cifar100_usl-t.yaml
cd ..
```
#### Evaluate USL-T on CIFAR-10/100 on SimCLR-CLD
```
# CIFAR-10
python fine_tune.py --cfg configs/cifar10_usl-t_finetune.yaml
# CIFAR-100
python fine_tune.py --cfg configs/cifar100_usl-t_finetune.yaml
```
### ImageNet
#### Download models trained with unsupervised learning methods
```
mkdir -p usl-t_pretraining/pretrained
wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_imagenet.tgz
tar zxvf pretrained_kNN_imagenet.tgz

wget https://people.eecs.berkeley.edu/~longlian/files/pretrained_kNN_imagenet100.tgz
tar zxvf pretrained_kNN_imagenet100.tgz
```
#### Perform USL-T (training and sample selection)
* Training:
```
cd usl-t_pretraining
# ImageNet100
python usl-t-imagenet-pretrain.py --cfg configs/ImageNet100_usl-t_pretrain.yaml
# ImageNet
python usl-t-imagenet-pretrain.py --cfg configs/ImageNet_usl-t_pretrain.yaml
cd ..
```
* Sample selection:
```
cd ../selective_labeling
# CIFAR-10
python usl-t-imagenet.py --cfg ImageNet100_usl-t_0.3.yaml
# CIFAR-100
python usl-t-imagenet.py --cfg ImageNet_usl-t_0.2.yaml
cd ..
```
#### Evaluate USL-T on ImageNet100/ImageNet with SimCLR
* SimCLR weights `r50_1x_sk0.pth` can be downloaded at [here](https://people.eecs.berkeley.edu/~longlian/files/r50_1x_sk0.pth.tgz) and should be put under pretrained.
```
cd semisup-simclrv2
# ImageNet-100
python fine_tune.py --lr 0.16 --seed 0 --name-to-save "usl-t_0.3p_imagenet100" --trainindex_x "[path to train_0.3p_gen_imagenet100_usl-t_0.3_index.csv]" -j 16 --batch-size 256 --eval-freq 48 --epochs 48 --warmup-epochs 0 --weight-decay 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 --pretrain pretrained/r50_1x_sk0.pth --roll-data 20 --amp-opt-level O1 --roll-data 40 --num-classes 100 --freeze-backbone [Path to ImageNet 100]

# ImageNet
python fine_tune.py --lr 0.16 --seed 0 --name-to-save "usl-t_0.2p_imagenet" --trainindex_x "[path to train_0.2p_gen_imagenet_usl-t_0.2_index.csv]" -j 16 --batch-size 512 --eval-freq 6 --epochs 12 --warmup-epochs 0 --weight-decay 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrain pretrained/r50_1x_sk0.pth --multi-epochs-data-loader --amp-opt-level O1 --no-lr-decay [Path to ImageNet]
```
## Samples selected by USL and USL-T
### CIFAR-10
Note that both USL and USL-T have some randomness due to non-deterministic computations in GPU and could vary between each run or server, despite setting the seed. Therefore, we release samples selected by USL run on our end.

These are the instance indices by the torch CIFAR-10 dataset.
<details>
<summary>Random indices on CIFAR-10</summary>
Seed 1 (class distribution [1, 6, 5, 3, 1, 3, 5, 4, 6, 6]):

```
[26247, 35067, 34590, 16668, 12196,  2600,  9047,  2206, 25607,
11606,  3624, 43027, 15190, 25816, 26370,  1281, 29433, 36256,
34217, 39950,  6756, 26652,  3991, 40312, 24580,  4949, 18783,
39205, 23784, 39508, 19062, 48140, 11314,   766, 39319, 15299,
10298, 25573, 18750, 19298]
```

Seed 2 (class distribution [4, 2, 6, 5, 7, 1, 5, 2, 4, 4]):

```
[23656, 27442, 40162,  8459,  8051, 42404,    89,  1461, 13519,
42536, 20817, 45479,  3121, 36502, 40119, 35971,  8784, 14084,
4063, 18730, 17763, 29366, 43841, 10741,  3986, 40475,  8470,
35621, 30892, 27652, 35359, 24435, 47853,  8835,  6572, 36456,
8750, 21067,  4337, 24908]
```

Seed 5 (class distribution [6, 6, 2, 3, 5, 3, 5, 2, 2, 6]):

```
[24166, 42699, 15927,  7473,  5070, 33926, 21409,  9495, 16235,
35747, 46288, 13560, 29644, 28992, 35350, 43077, 35757, 24106,
26555, 22478,  1951, 29145, 33373, 10043, 21988, 37116, 15760,
48939, 29761,  3702,  3273,  4175, 30998, 31012,  8754, 33953,
22206, 28896, 31669, 19275]
```

Seed 3 and 4 are not selected because seed 3 and seed 4 do not lead to instances of 10 classes for **random selection** and thus the comparison would not bring us much insights. Note that seed 3 and 4 lead to instances of 10 classes for **our selection**.

Note that these can be obtained by `selective_labeling/random-cifar.py`.
</details>

<details>
<summary>USL indices on CIFAR-10</summary>
Seed 1 (class distribution [5, 4, 5, 2, 2, 5, 5, 4, 3, 5]):

```
[ 3301, 37673, 33436, 28722, 10113,  5286, 21957, 13485,   445,
48678, 43647, 27879, 39987, 14374, 32536, 14741, 38215, 22102,
23082, 16734,  7409,   881, 10912, 37632, 39363,  7119,  6203,
28474, 25040, 43960, 24780, 45742, 49642, 25728,  9297, 21092,
4689,  4712, 48444, 30117]
```

Seed 2 (class distribution [4, 4, 4, 3, 3, 5, 4, 5, 3, 5]):

```
[19957, 40843, 45218,   881,  4557,  6203, 11400, 14374, 27595,
21092, 41009, 38215, 35471, 49642, 25728, 28722, 17094, 48678,
43960, 39363, 43647,  3907, 16734, 48023,  3301, 22102, 37632,
21130,  3646, 14741,  7127,  9297, 11961, 39987,  4712, 45568,
39908, 23505, 48421, 33436]
```

Seed 5 (class distribution [4, 5, 4, 3, 3, 4, 4, 4, 4, 5]):

```
[38215, 43213, 39363, 27965,   445, 16734, 14374,   914, 17063,
45918,  3301,  5286, 32457, 19867, 48678, 10455, 43647, 10912,
28722,  4712, 29946,  1221,  3907, 10110, 20670, 13410,  4689,
49642, 10018, 41210, 43755, 46227, 11961, 15682, 45742, 21092,
9692, 48023, 14741,  2703]
```

Seed 3 and 4 are not selected because seed 3 and seed 4 do not lead to instances of 10 classes for **random selection** and thus the comparison would not bring us much insights. Note that seed 3 and 4 lead to instances of 10 classes for **our selection**.

Note that these can be obtained by `selective_labeling/usl-cifar.py`.
</details>

<details>
<summary>USL-T indices on CIFAR-10</summary>
Class distribution [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]:

```
[7998, 45774, 27115, 8389, 28558, 8454, 12390, 42528, 28249, 
12885, 25101, 39912, 19571, 7904, 43637, 3267, 6935, 21794, 
24489, 13999, 24554, 19979, 1573, 36597, 5403, 44836, 29500, 
16935, 9408, 47504, 35673, 20778, 44636, 37123, 49130, 8086, 
39994, 8499, 48597, 7753]
```

</details>

### ImageNet
The random selection and USL selected samples on ImageNet (in `csv` format) could be obtained [here](https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet.html).
**Update**: USL-T selected samples on ImageNet is available at [here](https://people.eecs.berkeley.edu/~longlian/files/usl_t_imagenet_0.2_selected_indices.zip). 

## Model Zoo

### USL
Note that USL does not require training, so the weights are just pretraining weights (e.g., with MoCo/SimCLR/CLD).

| Dataset   | Models for Sample Selection |
|-----------|-----------------------------|
| CIFAR-10  | [Download (CLD)](https://people.eecs.berkeley.edu/~longlian/files/cifar10_ckpt_epoch_200.pth)              |
| CIFAR-100 | [Download (CLD)](https://people.eecs.berkeley.edu/~longlian/files/cifar100_ckpt_epoch_200.pth)             |
| ImageNet  | [Download (MoCov2-EMAN)](https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_800ep.pth.tar)            |

These weights can be directly parsed by `selective_labeling/usl-{cifar,imagenet}.py` to get selections. The selection should match with the selections released (besides slight variations due to indeterministic operations in GPUs).

#### FixMatch SSL Trained Model for USL
You can obtain the EMAN-FixMatch trained model for baselines/Stratified/USL-MoCo/USL-CLIP on ImageNet [here](https://people.eecs.berkeley.edu/~longlian/files/usl_imagenet.html).

### USL-T
| Dataset   | Models for Sample Selection | Results of this re-implementation | Results in the work |
|-----------|------------------------------------------------------------------------------------------------------------|-----------|-----------|
| CIFAR-10  | [Download](https://people.eecs.berkeley.edu/~longlian/files/cifar10_usl_t_3_heads.pth)                     | SimCLR-CLD: 76.1 | SimCLR-CLD: 76.1 |
| CIFAR-100 | [Download](https://people.eecs.berkeley.edu/~longlian/files/cifar100_usl_t_3_heads.pth)                    | SimCLR-CLD: 36.1 | SimCLR-CLD: 36.9 |
| ImageNet  | [Download](https://people.eecs.berkeley.edu/~longlian/files/imagenet_usl_t_3_heads.pth)                    | SimCLR: 42.0 | SimCLR: 41.3 |

These weights can be directly parsed by `selective_labeling/usl-t-{cifar,imagenet}.py` to get selections. The selection should mostly match with the selections released (besides slight variations due to indeterministic operations in GPUs).

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@inproceedings{wang2022unsupervised,
  title={Unsupervised Selective Labeling for More Effective Semi-Supervised Learning},
  author={Wang, Xudong and Lian, Long and Yu, Stella X},
  booktitle={European Conference on Computer Vision},
  pages={427--445},
  year={2022},
  organization={Springer}
}
```

## How to get support from us?
This is a re-implementation of our original codebase. If you have any questions about the implementation in this repo, feel free to email us at `longlian at berkeley.edu`.

I'm also happy to provide additional checkpoints for baselines and selected labels.

## License
This project is licensed under the MIT license. See [LICENSE](LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This project uses code fragments from many projects. See credits in comments for the code fragments and adhere to their own LICENSE. The code that is written by the authors of this project is licensed under the MIT license.

We thank the authors of the following projects that we referenced in our implementation:
1. [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) for the overall configuration framework. 
2. [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification) for augmentations, auxiliary dataloader wrappers, and many utility functions in USL-T.
3. [PAWS](https://github.com/facebookresearch/suncet) for the use of sharpening function.
4. [MoCov2](https://github.com/facebookresearch/moco) for augmentations.
5. [pykeops](https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html) for kNN and k-Means.
6. [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) for EMA.
7. [SimCLRv2-pytorch](https://github.com/Separius/SimCLRv2-Pytorch) for extracting and using SimCLRv2 weights
8. Other functions listed with their own credits.
