ResNet implementation with TensorFlow
# project files
>project file or dir description
 
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

# environment
- python3+
- tensorflow1.1+
- you need to install third packages required in this project with
   ```python
   pip/pip3/conda install -r requirements.txt
   ```
# usage
- python run.py -op {train,test,predict} -img img_path(only required when -op predict)
- you can reset super parameters in script named 'config.py'