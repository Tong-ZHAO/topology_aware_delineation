# Topology Aware Delineation

Final Project for M2 MVA course: Deformable models and geodesic methods for image analysis. In this project, we implement and analyse the method proposed in [Beyond the Pixel-Wise Loss for Topology-Aware Delineation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mosinska_Beyond_the_Pixel-Wise_CVPR_2018_paper.pdf) using PyTorch. It aims to improve the performance for the problem of delineation of curvilinear structures using deep learning methods.


 <figure>
  <img src="/img/pipeline.png" width="900"/>
  <figcaption>Figure 1: Training Pipeline Reprinted from Acticle</figcaption>
</figure> 

## Requirements

* PyTorch 1.0
* \>= Python 3
* Visdom

## Dataset

We test our algorithm on [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/). A Python script is provided in the Data folder to download the whole dataset. We then resize all images to 256 in order to reduce memory usage. 


## Run

### Training

First we train a UNet-based model with topology-aware loss using K=1.

```
python -m visdom.server 
python train.py
```

Then the model is finetuned using K=3 on the same dataset.

```
python train.py --K 3 --model ../log/pretrained/model.t7
```

<figure>
  <img src="/img/screenshot.png" width="900"/>
</figure> 


### Testing

```
python test.py --model path_to_model --image path_to_image --label path_to_groundtruth
```

## Results

We provide three pretrained models in the log folder, namely the UNet model without topology-aware loss (pure UNet), the UNet model with topology-aware loss using K=1 (Topological UNet), and the finetuned UNet model with topology-aware loss using K=3 (Iterative UNet).

### Qualitative Results

Here we show the results for two images in test dataset.

<p float="left">
  <img src="/img/input.png" width="260" />
  <img src="/img/label.png" width="260" /> 
  <figcaption>Figure 2: Input and Groundtruth for the First Example</figcaption>
</p>

<p float="left">
  <img src="/img/unet_iter1.png" width="260" />
  <img src="/img/k1_iter1.png" width="260" /> 
  <img src="/img/finetune_iter3.png" width="260" /> 
  <figcaption>Figure 3: pure UNet, topological UNet and iterative UNet (from left to right)</figcaption>
</p>

<p float="left">
  <img src="/img/input_207.png" width="260" />
  <img src="/img/label_207.png" width="260" /> 
  <figcaption>Figure 4: Input and Groundtruth for the Second Example</figcaption>
</p>

<p float="left">
  <img src="/img/unet_iter1_207.png" width="260" />
  <img src="/img/k1_iter1_207.png" width="260" /> 
  <img src="/img/finetune_iter3_207.png" width="260" /> 
  <figcaption>Figure 5: pure UNet, topological UNet and iterative UNet (from left to right)</figcaption>
</p>

### Quantitative Results

We use three criteria: Completeness, Correctness and Quality. The definitions can be found [here](https://pdfs.semanticscholar.org/5bfd/6cea8caef4a44ac67835c95f9906d61da894.pdf).

The results on test dataset are shown as below:

| Model | Completeness | Correctness | Quality |
|:-----:|:------------:|:-----------:|:-------:|
|  U-Net  |  0.67584426 | 0.59423022 | 0.43436453|
|  K=1     | 0.58282891 | 0.7042716  | 0.44924965 |
|  K=3     |  0.61157189 | 0.70223687 | 0.46712003 |
