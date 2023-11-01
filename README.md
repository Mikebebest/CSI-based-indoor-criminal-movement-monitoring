# CSI-based-indoor-criminal-movement-monitoring 

## Introduction
WiFi sensing is a widely used issue nowadays, We extracted Channel State Information (CSI) data from wireless signals and monitored indoor target movement. 


## Dataset
Please download our dataset and organize it in the structure:
```
CSI
├── Data
│   ├── train
│   │   ├── anchor
│   │   ├── positive
│   │   └── negative
│   │   ├── A
│   │   │   ├── anchor
│   │   │   ├── positive
│   │   │   └── negative
│   │   └── B
│   │       ├── anchor
│   │       │   ├── positive
│   │       │   └── negative
│   └── test
│       ├── anchor
│       ├── positive
│       └── negative
│       ├── A
│       │   ├── anchor
│       │   ├── positive
│       │   └── negative
│       └── B
│           ├── anchor
│           ├── positive
│           └── negative
```
+ **CSI size: 600x600x3**
+ **number of class: 2**
+ **classes: close, far away**
+ **train number set: 60**
+ **test   number set: 40** 


## Run
execute the **Triplet_model.py** directly

output of the result: Accuracy, F1-score

Draw the confusion matrix and save in **Data** folder

## Extraction CSI tool
[PicoScenes](https://ps.zpj.io/index.html) 

A powerful tool that supports COTs NICs:

Intel Wi-Fi 6E AX210 (AX210), Intel Wi-Fi 6 AX200 (AX200), Qualcomm Atheros AR9300 (QCA9300) and Intel Wireless Link 5300 (IWL5300)
