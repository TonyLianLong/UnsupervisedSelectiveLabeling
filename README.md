# Unsupervised Selective Labeling for More Effective Semi-Supervised Learning
by [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Long Lian*](https://tonylian.com/), and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley/ICSI. (*: co-first authors)

[Arxiv Paper](https://arxiv.org/abs/2110.03006) | [ECCV Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900423.pdf) | [Citation](#citation)

*European Conference on Computer Vision*, 2022.

This work is also presented in [CV in the Wild workshop](https://computer-vision-in-the-wild.github.io/eccv-2022/) in ECCV 2022.

![Teaser](assets/teaser.png)

This repository contains the code for USL on CIFAR. Other implementations are coming soon.

For further information regarding the paper, please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu). For information regarding the code and implementation, please contact [Long Lian](mailto:longlian@berkeley.edu).

## News
* 11/17/22 Provided CLD pretrained model and reference selections
* 07/20/22 Initial Implementation

## Supported Methods
- [x] USL

## Supported SSL Methods
- [x] FixMatch
- [x] SimCLRv2
- [x] SimCLRv2-CLD

## Run USL
### CIFAR-10
#### Download our CLD pretrained model on CIFAR for USL
```
mkdir selective_labeling/pretrained
cd selective_labeling/pretrained
# CLD checkpoint on CIFAR-10
wget https://people.eecs.berkeley.edu/~longlian/files/cifar10_ckpt_epoch_200.pth
```

#### Perform USL on CIFAR-10
```
cd selective_labeling
python usl-cifar.py
```

#### Evaluate USL on CIFAR-10 with SimCLRv2-CLD
```
cd semisup-simclrv2-cld
python fine_tune.py
```

## Samples selected by USL
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
If you have any general questions about this implementation, feel free to email us at `longlian at berkeley.edu` and `xdwang at eecs.berkeley.edu`.

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
