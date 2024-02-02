# Super-SCOTUS: A multi-sourced dataset for the Supreme Court of the US 

## Introduction

This repository contains code introduced in the following paper:

- [Super-SCOTUS: A multi-sourced dataset for the Supreme Court of the US](https://aclanthology.org/2023.nllp-1.20.pdf)

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


## Evaluation 
- Evaluation: `python evaluate.py <experiment>`, where the experiment listed above 


## Related work 
- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann. 2023. More than Votes? Voting and Language based Partisanship in the US Supreme Court. In Findings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore. Association for Computational Linguistics. [Github](https://github.com/biaoyanf/SCOTUS-partisanship)

- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann. 2023. It’s not only What You Say, It’s also Who It’s Said to: Counterfactual Analysis of Interactive Behavior in the Courtroom. In Proceedings of The 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, Nusa Dua, Bali. Association for Computational Linguistics. [Github](https://github.com/biaoyanf/SCOTUS-counterfactual)
