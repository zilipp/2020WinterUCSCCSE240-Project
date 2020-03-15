#Google Analytics Customer Revenue Prediction

## Data should be in ./data
link for data:  www.kaggle.com/c/ga-customer-revenue-prediction/data

## files in ./src:
data_processing.py: data cleaning

visualization.py: for data visualization

Main.py: main file in this project, can choose
mod = 'XGBOOST'  # 'LGBOOST' / 'XGBOOST' /'CBOOST'/ 'ASSEMBLE'
to run on different model

Main_selected.py: drop some features, which selected by importance in Main.py

NN_features.py: fully connected NN model

## visualizations should be in directory in ./graphs

## LSTM model runs in Kaggle Kernel, the file is in 
./src/LSTM.ipynb

link is:
https://www.kaggle.com/sherrymay/fork-of-aiprediction-ga-customer-revenue-lstm#3-Train-model 




