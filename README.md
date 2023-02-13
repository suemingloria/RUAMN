# Recurrent Unit Augmented Memory Network for Video Summarization
A PyTorch implementation of our paper "Recurrent Unit Augmented Memory Network for Video Summarization".

## Architecture Overview
![image](https://github.com/suemingloria/RUAMN/blob/main/imgs/framework.png)

## Installation
The development and evaluation was done on the following configuration:
### System configuration
- Platform :   Windows-10-10.0.19041-SP0
- GPU: NVIDIA GeForce GTX 1080 Ti
- CUDA:  9.2
- CUDNN: 7201

### Python packages
- Python: 3.6.13
- PyTorch:  1.5.1
- NumPy: 1.19.5 
- json: 2.0.9
- h5py: 3.1.0
- ortools: 6.9.5824


## Datasets and pretrained models
Preprocessed datasets [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), 
[YouTube](https://sites.google.com/site/vsummsite/download) and [OVP](https://sites.google.com/site/vsummsite/download) 
as well as RUAMN pretrained models can be downloaded by [preprocessed datasets link](https://pan.baidu.com/s/1eqyPdkHhv3bzCmk1mkr8Aw) with extraction code "ndba" and [pretrained datasets link](https://pan.baidu.com/s/1WNWd4tFj4TEcYEEH0Jj9MQ) with extraction code "za2x", respectively.

Datasets will be stored in ```./datasets``` 
directory and models in ```./data/models```.

Original version of the datasets can be downloaded from 
[http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz](http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz) 
or
[https://www.dropbox.com/s/ynl4jsa2mxohs16/data.zip?dl=0](https://www.dropbox.com/s/ynl4jsa2mxohs16/data.zip?dl=0).

## Training
To train the RUAMN on all split files in the ```./splits``` directory run this command:
```
python3 main.py --train
```

Results, including a copy of the split and python files, will be stored in ```./data``` directory. 

The final results will be recorded in ```./data/results.txt``` with corresponding models in 
the ```./data/models``` directory.    


## Evaluation
To evaluate all splits in ```./data/splits``` with corresponding trained models in ```./data/models``` 
run the following: 
```
python3 main.py
```


## Acknowledgement

We would like to thank to [K. Zhou et al.](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) and [K Zhang et al.](http://www-scf.usc.edu/~zhan355/ke_eccv2016.pdf) for making the preprocessed datasets publicly available and also [Jiri Fajtl et al.]() for the most of the VASNet code which I copied from https://github.com/ok1zjf/VASNet and modified according to the new Network architecture.


## References
```
@misc{fajtl2018summarizing,
    title={Summarizing Videos with Attention},
    author={Jiri Fajtl and Hajar Sadeghi Sokeh and Vasileios Argyriou and Dorothy Monekosso and Paolo Remagnino},
    year={2018},
    eprint={1812.01969},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

