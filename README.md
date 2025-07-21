# MVFE-MC

## Requirements

The model is implemented using PyTorch. The versions of the main packages:

+ tensorboard 
+ dashscope
+ torchvision               >=0.10.0
+ torch
+ tqdm
+ zstandard
+ numpy
+ oss2
+ transformers              4.20.1    
+ numpy                     1.21.6

If you are using a different version of the environment, you can try using parameter search to achieve the best score.

Install the other required packages:
``` bash
pip install -r requirements.txt
```


## TRAIN
### 1. Get Dataset
Please place the dataset in the path of 'Concrete-Image-Classification-main\our_dataset'.

If you have downloaded the original ConDef dataset, please execute the following command to obtain the preprocessed dataset:

```
cd our_dataset
bash run_preprocess.sh
```


### 2. Train
Please set your LLM's <api_key> and the OSS <access_key_id> and <access_key_secret> before starting training.
```
python train.py
```

### 3. Test
We provide the results of the obtained text modality for quick implementation If not necessary, please re execute the text modal classification results according to the comments
```
python test.py
```
