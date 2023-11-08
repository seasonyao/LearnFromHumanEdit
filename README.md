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

## Run Synthetic Data Generation

```
python SyntheticData.py
```

### Instructions for Synthetic Data Generation
Use the above script for the generation of synthetic data of two types:
1) High to Low (`H2L`): where the chosen summary is the reference summary & rejected summary is the LLM hallucinated summary.
2) Low to High (`L2H`): where the rejected summary is the pre-trained model-generated summary & chosen summary is the factually improved summary.

Make the following changes based on different synthetic data generation settings:

1) Add the OpenAI API key in the `openai_api_key` variable.
2) Update the pre-trained model checkpoint path in `model_checkpoint` variable for low to high (L2H) synthetic generation.
3) Update the OpenAI model type in `gpt_model_type` variable. This model is used to generate hallucinated and factually improved summaries.
    - `gpt_model_type: gpt-3.5-turbo-0613` for using GPT-3.5 Turbo
    - `gpt_model_type: gpt-4-0613` for using GPT-4
4) Update the synthetic data generation type in `synthetic_data_type` variable.
    - `synthetic_data_type: H2L` for High to Low synthetic data.
    - `synthetic_data_type: L2H` for Low to High synthetic data.
5) Update `data_files` variable to update the path for the base dataset.
6) Use `num_samples` to control the size of the synthetic dataset.
   
- 

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

@article{mishra2023synthetic,
  title={Synthetic Imitation Edit Feedback for Factual Alignment in Clinical Summarization},
  author={Mishra, Prakamya and Yao, Zonghai and Chen, Shuwei and Wang, Beining and Mittal, Rohan and Yu, Hong},
  journal={arXiv preprint arXiv:2310.20033},
  year={2023}
}
```
