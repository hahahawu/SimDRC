# Conversational Semantic Role Labeling

This code is based on the codebase [CSAGN](https://github.com/hahahawu/CSAGN).

## Setup

### Run
`python train.py MODEL_NAME -batch_size 128 -bert_version BERT_MODEL -intra_loss -inter_loss`

### Evaluation

`python evaluator.py TEST_DATASET MODEL_NAME`
