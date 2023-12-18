![header](imgs/header.png)
# Vision Transformers applied To Fundus Images

This repository contains my personal experiment regarding the training of Vision Transformers and customized derivatives on fundus images

This work is *in progress* and definetly **not production ready**.

## Adapting RETFound

Our code uses pretrained weights, either on ImageNet (coming from the great [timm library](https://timm.fast.ai/)) or from the Retinal Foundation Model [RETFound](https://github.com/rmaphoh/RETFound_MAE/tree/main). Please cite them accordingly if you use this code.

This repository adapts these models to fit in the pytorch-lightning and weights and biases framework for ease of experimentation.

## Dynamic Tokens Resampling

We explore the dynamic resampling of the tokens sequence within the Transformer. The idea is to proceed with multiple forward passes at progressively increasing resolutions. At each *scale*, we resample the sequence to only keep the most relevant tokens based on the Attention activations of the previous scales. The **max_tokens** argument in the [config file](configs/config.yaml) indicates the maximum length of the sequence.

This idea is heavily inspired of our [Focused Attention](https://www.sciencedirect.com/science/article/pii/S1361841522002377) paper, but adapted for training. We can track the selected tokens over the differents scales to get some insights on what the model is using in the input image to predict the disease.

|Image | Refined Attention|
:---------------:|:---------------:
![](figures/_tmp_images/batch_0.png) | ![](figures/_tmp_attn/batch_0.png)


## Running the code

```
git clone
cd RetinalViT
pip install .
train
```

You will need to adjust the path to the data (EyePACS and APTOS dataset) in the file [config file](configs/config.yaml)

To finetune the RetFound, you will also need to download [their weights](https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing)
