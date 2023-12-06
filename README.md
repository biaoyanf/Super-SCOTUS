# Super-SCOTUS: A multi-sourced dataset for the Supreme Court of the US 

## Introduction

This repository contains code introduced in the following paper:

- Super-SCOTUS: A multi-sourced dataset for the Supreme Court of the US 

- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann 

- In NLLP2023 

## Dataset 

- The full Super-SCOTUS dataset, a multi-sourced dataset for the Supreme Court of the US is available under [Harvard Database](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/POWQIT). 

- Tne experiment dataset for case decision direction prediction task is available under the same database. 



## Case Decision Direction Prediction  
- All related codes described below are under `case_direction_prediction` directory 
- Change the dataset path in `experiments.conf` to the downloaded path 

## Training Instructions 
- Run with python 3
- Run `python train.py <experiment>`, where experiment can be found from `experiments.conf


## Evaluation for 
- Evaluation: `python evaluate.py <experiment>`, where the experiment listed above 

