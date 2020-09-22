# PlasmidNet

## Introduction
Feel free to tweak code, add comments, etc. I quickly tested it with both Linux and Mac (CPU/GPU) to see if it works, so be aware it might be buggy as it hasn’t been packaged yet, also it might not be thread-safe yet but it is quite fast already. Looking forward for feedback.

## Setup

Installation:
```
conda create -p plasmidnet_env python=3.8
source activate $(pwd)/plasmidnet_env
pip install -r requirements.txt
```

Example of usage:

```
prodigal -i test.fa -o test.genes -a test.faa -p meta
python bin/plasmidnet.py -f test.faa -o results -m model.zip
```

for big .faa files add more `--threads` for feature calculation parallelization (currently it does it in batches of 5000. 

## TODO

[] packaging

[] better model?

[] testing

[] benchmark

[] version control

## training AUC monitoring

```
---------------------------------------
| EPOCH |  train  |   valid  | total time (s)
| 1     | 0.70223 |  0.83537 |   5612.2
| 2     | 0.89170 |  0.88819 |   11546.6
| 3     | 0.92209 |  0.89378 |   16828.2
| 4     | 0.93310 |  0.89493 |   22165.7
| 5     | 0.93913 |  0.89281 |   27462.6
| 6     | 0.94319 |  0.89315 |   32748.7
| 7     | 0.94576 |  0.89494 |   38094.6
| 8     | 0.94728 |  0.89518 |   43399.8
| 9     | 0.94995 |  0.89568 |   48770.7
| 10    | 0.95148 |  0.89479 |   53140.2
| 11    | 0.95051 |  0.89264 |   55632.8
| 12    | 0.94934 |  0.89470 |   58169.3
| 13    | 0.95094 |  0.89290 |   60659.1
| 14    | 0.95217 |  0.86314 |   63152.3
| 15    | 0.95384 |  0.88983 |   70272.2
| 16    | 0.95515 |  0.89196 |   77692.9
| 17    | 0.95574 |  0.89301 |   84678.5
| 18    | 0.95678 |  0.89194 |   87178.8
| 19    | 0.95769 |  0.89262 |   90051.6
Early stopping occured at epoch 19
Training done in 90051.618 seconds.
---------------------------------------
```
## test contigs confusion matrix
```
array([[ 9935,   208],
       [ 1849, 14015]])
```
