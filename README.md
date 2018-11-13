# 2018 Futurelab AI Contest

## In brief

The model is based on MobileNetV2 and NasNet. It is the ensemble of these two models with concatenation of these two feature maps.<br>
Parameters：
>lr:0.001(every epoch decay 0.98)<br>
>weight-decay:0.001<br>
>momentum:0.99<br>
>batch_size:64<br>
>epoch:100<br>

The MobileNetV2 network structure is given by [this](https://github.com/tonylins/pytorch-mobilenet-v2), and the pretrained model can be downloaded by [this](https://drive.google.com/file/d/1nFZhtKQcw_PeMg8ZZDLdWBcnzqx67hY9/view). We remove the parameters of the final fully connected layer and remove the prefix `module` in the `state_dict`.

The pretrained NasNet model, which can be downloaded by [this](https://github.com/veronikayurchuk/pretrained-models.pytorch/releases/download/v1.0/nasnetmobile-7e03cead.pth.tar), is processed in the same way as MobileNetV2.


## Software requirement

- Ubuntu 16.04
- CUDA 8
- Cudnn v6+
- python 3.5
    - pytorch 0.4.0
    - torchvision 0.2.0
    - numpy 1.13.3
    - Pillow 5.1.0
    - tensorboard_logger 0.1.0（optional）

## Deploy
- Train

    Dataset processing: split the sample dataset into training set and validation set with the default ratio 10:1：

    ```Bash
    python3 datautil.py <sample csv file>
    ```

    training：

    ```Bash
    python3 baseline.py [--val <val csv file>] <train csv file>
    ```

    **Important: the path of the data folder and the path of the csv file should be in the same folder**

- Test

    ```Bash
    python3 test.py  --test_model <model to be tested> --data <data folder>
    ```

