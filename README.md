# 4th Place Solution - NFL Player Contact Detection

## Camaro part

## 1. Environment
Easiest way to replicate kaggle environment.
```
docker run --gpus all -it --shm-size 32G --name kaggle gcr.io/kaggle-gpu-images/python /bin/bash
```

## 2. Setup
```
mkdir kaggle-nfl input
cd kaggle-nfl
git clone https://github.com/bamps53/kaggle-nfl2022
cd kaggle-nfl2022
pip install -r requirements.txt
```

## 3. Data
```
# competition dataset
kaggle competitions download -c nfl-player-contact-detection
unzip nfl-player-contact-detection.zip -d ../input/nfl-player-contact-detection

# game fold split
kaggle datasets download -d nyanpn/nfl-game-fold
unzip nfl-game-fold.zip -d ../input/nfl-game-fold

# pretrained model
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
python scripts/save_yolox_m_weight.py

```

## 4. preprocess
```
# split video to frames
python scripts/save_train_images.py

# preprocess for exp048
python scripts/preprocess_dataframe.py
python scripts/preprocess_dict.py

# preprocess for exp117/139
python scripts/preprocess_dataframe_v2.py --is_train
python scripts/preprocess_dict_v2.py --is_train

# preprocess for ep184/185
python preprocess_features.py 
```

## 5. Train
```
python exp048.py
python exp117.py
python exp139.py
python exp139.py --extract
python exp184.py
python exp185.py
```

## 6. Inference
Please refer to this notebook.  
Need to upload corresponding pretrained models to kaggle datasets.  
https://www.kaggle.com/code/bamps53/lb0796-exp184-185-lgb095?scriptVersionId=120623474