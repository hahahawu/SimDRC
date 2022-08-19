# Response generation

We evaluate the task of multi-turn dialogue response generation on two commonly used dialogue datasets, 
including DailyDialog(English) and LCCC(Chinese).

## Setup

First, set up the experimental environment.

``pip install -r requirements.txt``

## Data

We have released our well processed train/dev/test datasets. You can download them from 
[google drive](https://drive.google.com/drive/folders/1wU-OYz-eoLyl9I8NTJL7d2RoNsW1qztS?usp=sharing).

## Run

For DailyDialog,

```
bash run_bart_dailydialog.sh
```

For LCCC,

```
bash run_bart_lccc.sh
```

- --model_name: Identify your current running;
- --output_dir: The dir of checkpoints;
- --locality_loss: Whether use locality loss;
- --cross_loss: Whether use isotropy loss;
- --margin: The value of \delta;
- --alpha: The value of \alpha.
