# LearnFromHumanEdit

### SALT ENV installation
conda create -n salt python=3.8
conda activate salt
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch (according to your CUDA version, we try on these two (11.7 and 10.2) version)
conda install -c conda-forge matplotlib
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
pip install nltk
pip install ipdb
pip install rouge
pip install rouge-score

pip install datasets
pip install transformers

# install sequence alignment pkgs
cd pkgs
pip install -e . (for minineedle)

If qlora for llm:
pip install -q -U bitsandbytes 
pip install -q -U git+https://github.com/huggingface/peft.git 
pip install -q -U git+https://github.com/huggingface/accelerate.git 
