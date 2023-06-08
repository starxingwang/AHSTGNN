# Adaptive Hybrid Spatial-Temporal Graph Neural Network for Cellular Traffic Prediction

This is the original pytorch implementation of AHSTGNN in the following paper: [Adaptive Hybrid Spatial-Temporal Graph Neural Network for Cellular Traffic Prediction, ICC 2023](https://arxiv.org/abs/2303.00498).

<p align="center">
  <img width="350" height="400" src=fig/model.png>
</p>


## Requirements
- python 3.9
- numpy == 1.20.3
- scipy == 1.7.3
- pandas == 1.5.3
- torch == 1.11.0
- 
Dependencies can be installed using the following command:
```
pip install -r requirements.txt
```

## Data

Step1: The Milan dataset used in the paper can be downloaded from [Google Driver](https://drive.google.com/file/d/12xF8Gx5eQ5NCc1blzALlIAs3-wF-sL62/view?usp=drive_link) or [Baidu Pan](https://pan.baidu.com/s/1HOnapdts_JazgLdiWaqstg), password p9gx.

Step2: Process raw data 

```
# Create data directories
mkdir -p data/{Milan}

# Milan
python generate_training_data.py --output_dir=data/Milan --traffic_df_filename=data/data_mi_min.npy
```
## Train Commands

```
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```
## Main results
We compare our model with typical cellular traffic prediction methods including HA (Historical Average), LSTM, MVSTGN and AMF-STGCN, as well as the generic advanced spatial-temporal sequence prediction methods including Graph WaveNet, MTGNN and AGCRN.

<p align="center">
  <img width="650" height="300" src=fig/results.png>
</p>

## Citation
if you find this repository useful, please cite our paper.

```
@article{wang2023adaptive,
  title={Adaptive Hybrid Spatial-Temporal Graph Neural Network for Cellular Traffic Prediction},
  author={Wang, Xing and Yang, Kexin and Wang, Zhendong and Feng, Junlan and Zhu, Lin and Zhao, Juan and Deng, Chao},
  journal={arXiv preprint arXiv:2303.00498},
  year={2023}
}
```
## Acknowledgement
We appreciate the Graph WaveNet a lot for the valuable code base:
https://github.com/nnzhan/Graph-WaveNet