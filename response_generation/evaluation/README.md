# Evaluation

The traditional n-gram overlap and text matching metrics, such as BLEU and ROUGE, are not good choices to 
open-domain dialogue systems which permit significant diversity and allow multiple plausible outputs for a 
given input. Therefore, we choose to measure the generation quality over following automatic evaluation metrics, 
including [BERTScore](https://github.com/Tiiiger/bert_score), [BARTScore](https://github.com/neulab/BARTScore), 
[BLEURT](https://github.com/google-research/bleurt) and Distinct2/4.

## BARTScore

#### 1. Prepare

Download the [BARTScore](https://github.com/neulab/BARTScore) repository and overwrite the BARTScore dictionary here.

``git clone https://github.com/neulab/BARTScore``
   
#### 2. Execute

``CUDA_VISIBLE_DEVICES=0 python eval_bart_score.py --lm [Pretrained_BART_Model] --file_path XXX --batch_size 32``

## BLEURTScore

Download the [bleurt] repository and overwrite the dictionary here. Then, download the BLEURT-base checkpoint.

```
# Downloads the BLEURT-base checkpoint.
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```

## BERTScore

Install bert_score via pip.

``pip install bert-score``
