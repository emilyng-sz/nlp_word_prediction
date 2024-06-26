# Word Prediction Model 

This repository contains the code and interface to train and implement a language model that allows for word prediction based on user inputs. 

There are two language models:
- trigram model (based on n-gram probabilites)
- recurrent neural network (RNN)

# How to run the code

In your terminal in the main directory of this repository,
1. Run `pip3 install requirements.txt`
2. Move a text file (.txt) containing sentences into a 'data/' folder (untracked)
3. To run the language models:
- trigram model: 
    - Run `python TrigramTrain.py -f data/[data file name] -d [model name]`
    - Run GUI by `python WordPredictorSim.py' -f [model name]`
    - To generate statistics instead, you can run `python WordPredictorSim.py -f [model name] --stats data/[data file name]`
- Rnn model: 
    - Run `python3 RNN.py -d data/[data file name] -m data/[model name] -t` to train and save the model
    - Run the same command but without the training flag to load and test the model, i.e. `python3 RNN.py -d data/[data file name] -m data/[model name]`

# How to interact with the GUI

After running `python WordPredictorSim.py' -f [model name]`, you can choose to either enter the typing console mode by inputting 'type', or you can quit by inputting 'quit'.

To select the first word from the list of suggested words, type "1" and press enter. To choose the second word, type "2", and so on.

Type "reset" to clear the letters you’ve entered for the current word if you want to start over.

To complete a word, type a space and press enter. This will prompt you to start typing the next word.

To exit and return to the welcome screen, type "quit". To close the program completely, type "quit" again.

