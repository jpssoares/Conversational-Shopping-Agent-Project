# Conversational-Shopping-Agent-Project

Project made by Artur Stopa, João Soares and Ricardo Pereira.


### Instalation/Setup
#### Step 0 - Create env File
Create a .env file with the variables that are in the example. (Given in the labs)

```
API_USER=''
API_PASSWORD=''
```
#### Step 1 - Create a conda environment
```
conda create -n myenv python=3.9 ipykernel numpy scipy scikit-learn pandas tqdm jupyter matplotlib gensim flask flask_cors ipympl -c defaults -c conda-forge
```
Then you run that env:
```
conda activate myenv
```
Then you can check if you have all the required packages.
```
conda install pip
pip install -r requirements.txt
```

**Note:** From what I understand the `pip install -r requirements.txt` command will fail if any of the packages listed in the requirements.txt fail to install. Running each line with pip install may be a workaround.
```
cat requirements.txt | xargs -n 1 pip install
```

#### Step 2 - Installing PyTorch+HuggingFace+Spacy
Check lab0 for instructions:
https://wiki.novasearch.org/wiki/lab_setup

#### Step 3 - Run the program

```
python app.py
```
