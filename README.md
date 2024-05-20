# Word Prediction Model 

This repository contains the code and interface to train and implement a language model that allows for word prediction based on user inputs. 

There are two language models:
- trigram model (based on n-gram probabilites)
- recurrent neural network (RNN)

# How to run the code

In your terminal in the main directory of this repository,
1. Run `pip3 install requirements.txt`
2. Move a text folder containing sentences into a 'data/' (untracked) folder
3. To run the language models:
- trigram model: 
- Rnn model: 
    - Run `python3 RNN.py -d data/[data file name] -t data/[model name]` to train and save the model
    - Run `python3  RNN.py -m data/[model name] -d data/[data file name]` to test the model and predict 10 words. Number of words can be modified using `-n` 

