# Unsupervised Selective Labeling for More Effective Semi-Supervised Learning
by [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Long Lian*](https://tonylian.com/), and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley/ICSI. (*: co-first authors)

[Preprint](https://arxiv.org/abs/2110.03006v3) | [Citation](#citation)

*European Conference on Computer Vision*, 2022.

![Teaser](assets/teaser.png)

This repository contains the code for USL on CIFAR-10. The ImageNet versions are coming soon.

For further information regarding the paper, please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu). For information regarding the code and implementation, please contact [Long Lian](mailto:longlian@berkeley.edu).


## Example Training Commands

Train USL on CIFAR-10
```
cd selective_labeling
python usl-cifar.py
```

Evaluate USL on CIFAR-10 with SimCLRv2-CLD
```
cd semisup-simclrv2-cld
python fine_tune.py
```

## Model Zoo
Coming soon.

## FAQ
Coming soon.

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@article{wang2021unsupervised,
  title={Unsupervised Selective Labeling for More Effective Semi-Supervised Learning},
  author={Wang, Xudong and Lian, Long and Yu, Stella X},
  journal={arXiv preprint arXiv:2110.03006},
  year={2021}
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
2. [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification) for augmentations, auxiliary dataloaders, and many utility functions in USL-T
3. [PAWS](https://github.com/facebookresearch/suncet) for the use of sharpening function.
4. [MoCov2](https://github.com/facebookresearch/moco) for augmentations.
5. [pykeops](https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html) for kNN and k-Means.
6. [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) for EMA.
7. Other functions listed with their own credits.
