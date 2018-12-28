# Fully Convolutional Network with Multi-Step Reinforcement Learning for Image Processing
This is the official implementation of the [paper](https://www.hal.t.u-tokyo.ac.jp/~furuta/pub/fcn_rl/fcn_rl.html) in AAAI2019.
We provide the sample codes for training and testing and pretrained models on Gaussian denoising.

## Requirements
- Python 3.5+
- Chainer 5.0+
- ChainerRL 0.5+
- Cupy 5.0+
- OpenCV 3.4+

You can install the required libraries by the command `pip install -r requirements.txt`.
We checked this code on cuda-10.0 and cudnn-7.3.1.

## Folders
The folder `denoise` contains the training and test codes and pretrained models without convGRU and reward map convolution (please see Table 2 in [our paper](https://arxiv.org/abs/1811.04323)).
`denoise_with_convGRU` contains the ones with convGRU, and `denoise_with_convGRU_and_RMC` contains the ones with both convGRU and reward map convolution.

## Usage

### Training
If you want to train the model without convGRU and reward map convolution, please go to `denoise` and run `train.py`.
```
cd denoise
python train.py
```

### Test with pretrained models
If you want to test the pretrained model without convGRU and reward map convolution,
```
cd denoise
python test.py
```

## Note
Although we used BSD68 training set and [Waterloo exploration database](https://ece.uwaterloo.ca/~k29ma/exploration/) for training in our paper, this sample code contains only BSD68 training set (428 images in `BSD68/gray/train`).
Therefore, to reproduce our results (Table 2 in [our paper](https://arxiv.org/abs/1811.04323)) by running `train.py`, please download [Waterloo exploration database](https://ece.uwaterloo.ca/~k29ma/exploration/) and add it into training set by yourself.

The pretraind models were trained on both BSD68 training set and [Waterloo exploration database](https://ece.uwaterloo.ca/~k29ma/exploration/), so `test.py` can reproduce our results (Table 2 in [our paper](https://arxiv.org/abs/1811.04323)).

## References
We used the publicly avaliable models of [[Zhang+, CVPR17]](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html) as the initial weights of our model except for the convGRU and the last layers. We obtained the weights from [here](https://github.com/cszn/DnCNN) and converted them from MatConvNet format to caffemodel in order to read them with Chainer.

We obtained the BSD68 dataset from
- [https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html),
and converted the images to gray scale by using the cvtColor function in OpenCV.

Our implementation is based on [a3c.py](https://github.com/chainer/chainerrl/blob/master/chainerrl/agents/a3c.py) in ChainerRL library and the following articles. We would like to thank them.
- [http://seiya-kumada.blogspot.com/2016/03/fully-convolutional-networks-chainer.html](http://seiya-kumada.blogspot.com/2016/03/fully-convolutional-networks-chainer.html)
- [https://www.procrasist.com/entry/chainerrl-memo](https://www.procrasist.com/entry/chainerrl-memo)

## Citation
If you use our code in your research, please cite our paper.
```
@inproceedings{aaai_furuta_2019,
    author={Ryosuke Furuta and Naoto Inoue and Toshihiko Yamasaki},
    title={Fully Convolutional Network with Multi-Step Reinforcement Learning for Image Processing},
    booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
    year={2019}
}
```
