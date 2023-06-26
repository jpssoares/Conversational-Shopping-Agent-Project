# Conversational-Shopping-Agent-Project

Project made by Artur Stopa, João Soares and Ricardo Pereira.

Conversational agent built with ML models, such as Clip, Blip and GPT.
![image](https://github.com/jpssoares/Conversational-Shopping-Agent-Project/assets/57997233/4178851f-b8d5-460a-8966-ee016cbb5045)


### Dialog State Manager Graph
![image](https://github.com/jpssoares/Conversational-Shopping-Agent-Project/assets/57997233/901f73f7-6f3e-4c8d-9477-2ca15aac9b33)


Read the paper [here](https://github.com/jpssoares/Conversational-Shopping-Agent-Project/blob/main/paper.pdf).

### Instalation/Setup
#### Step 0 - Create env File
Create a .env file with the variables that are in the example. (Given in the labs)

```
API_USER=''
API_PASSWORD=''
OPENAI_API_KEY=''
```
Note: If you don't have a Open AI API key, you can run the program without GPT. Everything that isn't a greeting or a recognized intent is treated as OOS(out of scope). 
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

#### Step 3 - Import the trained-models folder(for the dialog system)
Get the folder from here:
https://drive.google.com/file/d/1VGHkzIzag0bXj4wdD5DD1CTjYU4Fs54D/view
#### Step 4 - Run the program

```
python app.py
```
