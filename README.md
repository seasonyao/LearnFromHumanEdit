# LearnFromHumanEdit

## Installation
If using `conda`, you can get this to work as follows:

```
conda create -n salt python=3.8
conda activate salt
```

We have experimented with 11.7 and 10.2 cuda version, but this release should work with more recent versions as well.
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```
or 

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
```

Install other packages:
```
conda install -c conda-forge matplotlib
conda install -c conda-forge spacy
conda install -c conda-forge scipy
python -m spacy download en_core_web_sm
pip install nltk
pip install ipdb
pip install rouge
pip install rouge-score
pip install trl
pip install minineedle
pip install nltk

pip install datasets
pip install transformers
```
If you want to use qlora for llm:
```
pip install -q -U bitsandbytes 
pip install -q -U git+https://github.com/huggingface/peft.git 
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

## Run the trainer

```
python DPO_trainer.py
python SFT_trainer.py
python SALT_trainer.py
```

## TODO
- Adapt the codes *_trainer.py 
    - Save output models
    - Save outputs
- Modify the classes in dpo.py and rename it to be more generic
- Add link to paper and bib
- Add dataset
- Do we need wandb instructions

## Citation

```
@article{yao2023improving,
  title={Improving Summarization with Human Edits},
  author={Yao, Zonghai and Schloss, Benjamin J and Selvaraj, Sai P},
  journal={arXiv preprint arXiv:2310.05857},
  year={2023}
}

@inproceedings{Mishra2023SyntheticIE,
  title={Synthetic Imitation Edit Feedback for Factual Alignment in Clinical Summarization},
  author={Prakamya Mishra and Zonghai Yao and Shuwei Chen and Beining Wang and Rohan Mittal and Hong Yu},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:264812518}
}
```
