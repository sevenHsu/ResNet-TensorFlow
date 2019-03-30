# ResNet implementation with TensorFlow
## project files
>Project file or dir description
 
| file/dir         | git ignore | description                            |
|:-----------------|:-----------|:---------------------------------------|
| data             | N          | contain cifar-10 and test_images       |
| data/cifar-10    | Y          | cifar-10 dataset                       |
| data/test_images | N          | contain test images                    |
| models           | N          | for saving trained models              |
| summary          | N          | for saving training log                |
| config.py        | N          | training & base configuration          |
| data_loader.py   | N          | load & process data script             |
| evaluate.py      | N          | contain evaluate functions             |
| resnet.py        | N          | ResNet model script                    |
| run.py           | N          | start script for training/test/predict |

## Environment
- python3+
- tensorflow1.1+
- you need to install third packages required in this project with
   ```python
   pip/pip3/conda install -r requirements.txt
   ```
## Dataset
- I trained ResNet with [**CIFAR-10**](http://www.cs.toronto.edu/~kriz/cifar.html),size of images is 32*32.
- you can set size of input images to 224*224 and train this model with your own dataset 
## Usage
- python run.py -op {train,test,predict} -img img_path(only required when -op predict)
- you can reset super parameters in script named 'config.py'
## Experiment
> I trained cifar-10 dataset for 5 iterations about 10 minutes with GPU(2080Ti).I did not using randomly crop for training,
>so the accuracy of the trained ResNet not good.

![airplane](/predict_images/airplane.png)

![bird](/predict_images/bird.png)

![car](/predict_images/car.png)

![deer](/predict_images/deer.png)

![dog](/predict_images/dog.png)

you can see more predictions in the directory [predict_images](/predict_images) 
## Reference
- [ResNet](https://arxiv.org/abs/1512.03385v1)
- [My Blog](https://sevenhsu.github.io/2019/03/22/2019_03_19_10/)