# LearnFromHumanEdit

### SALT ENV installation
If using `conda`, you can get this to work as follows:

```
conda create -n salt python=3.8
conda activate salt

according to your CUDA version, we try on these two (11.7 and 10.2) version:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

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


pip install datasets
pip install transformers

If qlora for llm:
pip install -q -U bitsandbytes 
pip install -q -U git+https://github.com/huggingface/peft.git 
pip install -q -U git+https://github.com/huggingface/accelerate.git
'''

TODO: remove HG auth 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 from jupyter and deactivate it, since it is in github