# UniChange

> __Multi-modal change interpretation based on diffusion models for remote sensing__  
> Liwen Zhang, Xiaofei Yu, Yitong Li, Jie Ma*， Chang Li*, Hanlin Wu  [[paper](https://arxiv.org/abs/2405.12875)]

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
                  ├─val
                  │  ├─A
                  │  ├─B
                  ├─test
                  │  ├─A
                  │  ├─B
```


## Installation and Dependencies
```python
git clone https://github.com/Fay-Y/UniChange
cd UniChange
conda create -n unichange python=3.8
conda activate unichange
pip install -r requirements.txt
```
## Preparation
Preprocess the raw captions and image pairs:
```python
python caption_preprocess.py
python img_preprocess.py
```

## Training
 To train the proposed UniChange, run the following command:
```python
sh demo.sh
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
Prediction results in test set with 5 Ground Truth captions are partly shown below, proving the effectiveness of our model. 
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/eaf7ba0c-1a4d-44cd-9d11-84bfda0058ab" alt="compare2" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/b61bad59-afd0-4313-9b97-d7ab859222eb" alt="compare1" width="500"/></td>
  </tr>
</table>

## TODO
- [ ] Release training logs and checkpoints
- [ ] Support more RSICC datasets






