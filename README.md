# Human activity recognition with pytorch-lightning
This is a Pytorch implementation of the basic CNN approach proposed in there: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/Your-project-name   

# Create and activate virtual env (OPTIONAL)
cd human-activity-recognition
python3 -m venv har
source har/bin/activate

# Install dependencies
pip install -r requirements.txt
 ```   
 Next, download the [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip) and unzip it in a `data/`. Your folder should look like that:
 ```bash
 .
├── README.md
├── data
│   └── UCI HAR Dataset
│       ├── README.txt
│       ├── activity_labels.txt
│       ├── features.txt
│       ├── features_info.txt
│       ├── test
│       └── train
├── requirements.txt
├── setup.py
└── src
```
 
 Next, navigate to the baseline code and run it.   
 ```bash
# module folder
cd src/cnn1d/baseline

# run module (example: mnist as your main contribution)   
python trainer.py    
```
With the proposed setting, after 10 epochs we obtain the following confusion matrix and an accuray of 90.7%:
```
[[462  16  18   0   0   0]
 [  6 439  26   0   0   0]
 [  2   5 413   0   0   0]
 [  0   9   0 375 107   0]
 [  1   2   0  70 459   0]
 [  0  14   0   0   0 523]]
 ```
