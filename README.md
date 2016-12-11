## raw data
Please put raw data in `raw_data` folder.

Before unzip the original file, please run `md5sum *.zip` and check if the result is the same as md5test.txt.

## Train Deep Learning Models
Using CNN + LSTM model, treat frequency domain as 

### Create Data
```
python image_generation.py data.pkl
```

### Train Model
Train pure CNN:
```
python train_cnn.py data.pkl submission.csv
```
Train CNN LSTM stacking model:
```
python train_cnn_lstm.py data.pkl submission.csv
```

## Train Feature based model

### Create Data
```
python feature_generation.py raw_data/train*/*.mat raw_data/test*/*.mat data.pkl
```

### Train Random Forest
```
python train_rf.py data.pkl submission.csv
```

### Train Extreme Gradient Boosting
```
python train_xgb.py data.pkl submission.csv
```
