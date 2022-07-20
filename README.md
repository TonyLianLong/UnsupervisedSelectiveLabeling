# Unsupervised Selective Labeling for More Label-Efficient Semi-Supervised Learning
by [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Long Lian*](https://tonylian.com/), and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley/ICSI. (*: co-first authors)

To appear in *ECCV*, 2022.

![Teaser](assets/teaser.png)

This repository contains the code for USL on CIFAR-10. The ImageNet versions are coming soon.

Further information please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu) (regarding the paper) and [Long Lian](mailto:longlian@berkeley.edu) (regarding this repo and code implementation).

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

## How to get support from us?
If you have any general questions, feel free to email us at `longlian at berkeley.edu` and `xdwang at eecs.berkeley.edu`.

## License
This project is licensed under the MIT license. See [LICENSE](LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This project uses code fragments from many projects. The code fragments are credited with comments and adhere to their own LICENSE. The code that is written by the authors of this project is licensed under the MIT license.

We thank the authors of the following projects that we referenced in our implementation:
1. [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) for the overall configuration framework. 
2. [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification) for augmentations, auxiliary dataloaders, and many utility functions in USL-T
3. [PAWS](https://github.com/facebookresearch/suncet) for the use of sharpening function.
4. [MoCov2](https://github.com/facebookresearch/moco) for augmentations.
5. [pykeops](https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html) for kNN and k-Means.
6. [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) for EMA.
7. Other small functions listed with their own credits.
