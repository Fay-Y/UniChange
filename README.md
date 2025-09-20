# UniChange

> __Multi-modal change interpretation based on diffusion models for remote sensing__  
> Liwen Zhang, Xiaofei Yu*, Jie Ma, Chang Li, Xiaoyu Zhang 

##  Model Architecture
The proposed UniChange consists of:
- A **joint diffusion framework** that simutaneously models the distribution of change captioning and change detection.
- A **difference-ware conditioning** block that fuse the change dynamics into the backward denosing propagation:



![flowchart](https://github.com/user-attachments/assets/9d77e58b-f1f6-4783-a391-15adf221b173)

### Datasets
#### LEVIR-MCI
- A large-scale RSICC dataset with 10,077 bi-temporal image pairs and 50,385 captions.
- Covers multiple semantic change types: buildings, roads, vegetation, parking lots, water.
- Resized images: 256×256.
- change map: buildings (white), roads (grey)

Download Source:
-Thanks for the Dataset by Liu et. al:[[GitHub](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)].
Put the content of downloaded dataset under the folder 'data'
```python
path to ./data:
                ├─LevirCCcaptions.json
                ├─images
                  ├─train
                  │  ├─A
                  │  ├─B
                  │  ├─label
                  ├─val
                  │  ├─A
                  │  ├─B
                  │  ├─label
                  ├─test
                  │  ├─A
                  │  ├─B
                  │  ├─label
```


## Installation and Dependencies
```python
git clone git@github.com:Fay-Y/UniChange.git
cd UniChange
conda create -n unichange python=3.8
conda activate unichange
pip install -r requirements.txt
```
## Preparation
Preprocess the raw captions and image pairs to extract features in advance:
```python
python caption_preprocess.py
python img_preprocess.py
```

## Training
 To train the proposed UniChange, run the following command:
```python
sh train.sh
```

## Inference
 For inference and visualization on the test dataset, run the following command
```python
python joint_sample_batch.py
```

## Visualization
```python
cd out_joint_step_50
```
In the paper, the predicted captions are saved in folder "result". 
## Prediction samples
Change captioning prediction results in test set with 5 Ground Truth captions are partly shown below, proving that our model can generate more semantically rich caption. 

<img  alt="experiment_cc" src="https://github.com/user-attachments/assets/afccb7ab-24f6-45fb-97a1-bda1dabf2296" alt="change captioning comparison" width="500"/>

Change detection prediction results in test set the Ground Truth label, demonstrating the effectiveness of UniChange. 
<img width="1000"  alt="experiment_cd" src="https://github.com/user-attachments/assets/5075ecb4-df20-4194-bc38-598cdea9a3bc" />


## TODO
- [ ] Release training logs and checkpoints
- [ ] Support more change interpretation datasets






