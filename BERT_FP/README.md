# Multi-turn Response Retrieval

This code is reimplemented as a fork of [BERT-FP](https://github.com/hanjanghoon/BERT_FP).

**The next parts are the official guidelines of BERT-FP. You can directly follow it.**

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.8.0, and provides out of the box support with CUDA 11.2
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```


Preparing Data and Checkpoints
-------------

### Post-trained and fine-tuned Checkpoints

We provide following post-trained and fine-tuned checkpoints. 

- [fine-grained post-trained checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][3]
- [fine-tuned checkpoint for 3 benchmark datasets (ubuntu, douban, e-commerce)][4]


### Data pkl for Fine-tuning (Response Selection)
We used the following data for post-training and fine-tuning
- [fine-grained post-training dataset and fine-tuning dataset for 3 benchmarks (ubuntu, douban, e-commerce)][5]


Original version for each dataset is availble in [Ubuntu Corpus V1][6], [Douban Corpus][7], and [E-Commerce Corpus][8], respectively.


Fine-grained Post-Training
--------

##### Making Data for post-training and fine-tuning  

```
Data_processing.py
```


### Post-training Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

```shell
python -u FPT/ubuntu_final.py --num_train_epochs 25
python -u FPT/douban_final.py --num_train_epochs 27
python -u FPT/e_commmerce_final.py --num_train_epochs 34
```

### Fine-tuning Examples

##### (Ubuntu Corpus V1, Douban Corpus, E-commerce Corpus)

###### Taining 
```shell
To train the model, set `--is_training`
python -u Fine-Tuning/Response_selection.py --task ubuntu --is_training
python -u Fine-Tuning/Response_selection.py --task douban --is_training
python -u Fine-Tuning/Response_selection.py --task e_commerce --is_training
```
###### Testing
```shell
python -u Fine-Tuning/Response_selection.py --task ubuntu
python -u Fine-Tuning/Response_selection.py --task douban 
python -u Fine-Tuning/Response_selection.py --task e_commerce
```

[1]: https://github.com/huggingface/transformers
[2]: https://github.com/taesunwhang/BERT-ResSel
[3]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[4]: https://drive.google.com/file/d/1n2zigNDiIArWtsiV9iUQLwfSBgtNn7ws/view?usp=sharing
[5]: https://drive.google.com/file/d/16Rv8rSRneq7gfPRkpFZseNYfswuoqI4-/view?usp=sharing
[6]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[7]: https://github.com/MarkWuNLP/MultiTurnResponseSelection
[8]: https://github.com/cooelf/DeepUtteranceAggregation
