import nltk
import codecs
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import argparse
import random
import time
import json

def process_files(f):
    """
    Processes the file f and returns word_to_index:dict, index_to_word:dict, and cleaned_sentences:list of strings.
    """
    print("Processing file...")
    word_to_index = {'<PAD>':0}
    index_to_word = {0:'<PAD>'}
    punctuation_pattern = re.compile(r'[^\w\s]')
    with codecs.open(f, 'r', 'utf-8') as text_file:
        text = text_file.read().encode('utf-8').decode() #Maintain capitalization
    
    cleaned_sentences = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.replace('\r\n', ' ')
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.replace('\r', ' ')
        sentence = re.sub(r'\d+', '', sentence)
        sentence = re.sub(punctuation_pattern, '', sentence)
        cleaned_sentences.append(sentence)
        for word in sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word[len(word_to_index)] = word
    return word_to_index, index_to_word, cleaned_sentences

## Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output[:, -1, :]) # [batch_size, seq_len, vocab_size] but we index what we want
        return output
    
def train_model(text_data, embedding_dim, hidden_dim, epochs):
    # Convert text data to sequence data
    input_sequences = []
    for line in text_data:
        token_list = line.split()
        sequence = [word_to_index[word] for word in token_list]
        max_sentence_length = min(80, len(sequence)) # truncate sentence if it is too long
        for i in range(1, max_sentence_length):
            n_gram_sequence = sequence[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences with 0s in front of the words 
    max_sequence_len = max([len(x) for x in input_sequences])
    print(f"max sequence length: {max_sequence_len}, total num sentences: {len(input_sequences)}")

    # to create sample if too many input sentences, OR to reshuffle
    input_sequences = random.sample(input_sequences, min(len(input_sequences), 30000))

    max_sequence_len = max([len(x) for x in input_sequences])
    
    input_sequences = np.array([[0] * (max_sequence_len - len(i)) + i for i in input_sequences])
    
    print(f"updated max sequence length: {max_sequence_len}, updated total num sentences: {len(input_sequences)}")
    
    # Convert to PyTorch tensors
    # X_train is the input sequence, y_train is the following word. e.g. "the" -> "quick", "the quick" -> "brown" etc
    X_train = torch.tensor(input_sequences[:, :-1], dtype=torch.long)
    y_train = torch.tensor(input_sequences[:,-1], dtype=torch.long)

    # Define model parameters, most are from function input
    vocab_size = len(word_to_index)

    # Instantiate the model
    model = RNNModel(vocab_size, embedding_dim, hidden_dim)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    time_start = time.time()
    start_of_new_epoch = time_start
    # Training the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train) 
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            print(f'time taken for these epochs: {(time.time() - start_of_new_epoch)/60} mins')
            start_of_new_epoch = time.time()

    print(f'Total taken for training: {(time.time() - time_start)/60} mins')
    return model

###
def save_index(index, path):
    # Save list to a JSON file
    with open(path, 'w') as json_file:
        json.dump(index, json_file)

def load_index(path):
    with open(path, 'w') as json_file:
        index = json.load(index, json_file)
    return index

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, vocab_size, embedding_dim, hidden_dim):
    model = RNNModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(path))

    # Set the model to evaluation mode
    model.eval()
    return model

# Function to predict the next word
def predict_next_word(seed_text, model):
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor([word_to_index[word] for word in seed_text.split()], dtype=torch.long).unsqueeze(0)
        output = model(sequence)
        _, predicted_index = torch.max(output, 1)
        predicted_word = index_to_word[predicted_index.item()]
        return predicted_word

# Function to predict x number of next words
def predict_next_words(seed_text, model, word_to_index, index_to_word, num_words):
    model.eval()
    for _ in range(num_words):
        with torch.no_grad():
            sequence = torch.tensor([word_to_index[word] for word in seed_text.split()], dtype=torch.long).unsqueeze(0)
            output = model(sequence)
            _, predicted_index = torch.max(output, 1)
            predicted_word = index_to_word[predicted_index.item()]
            seed_text += ' ' + predicted_word

    return seed_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or use an RNN for word prediction.")
    parser.add_argument('--train', '-t', action='store_true', help="Train a new model.")
    parser.add_argument('--model_path', '-m', type=str, help="path to save or load the model.")
    parser.add_argument('--data_path', '-d', type=str, help="data path")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training.")
    parser.add_argument('--embedding_dim', type=int, default=100, help="Dimension of word embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=150, help="Dimension of RNN hidden states.")
    parser.add_argument('--num_words', '-n', type=int, default=10, help="Number of words to predict.")

    args = parser.parse_args()

    if args.train and args.data:
        # Load and preprocess the text data
        word_to_index, index_to_word, text_data = process_files(args.data_path)
        
        # Train the model
        model = train_model(text_data, args.embedding_dim, args.hidden_dim, args.epochs)

        save_model(model, args.model_path)

        print(f"Model trained and saved at {args.model_path}")

    elif args.model_path: #and args.w2i_path and args.i2w_path:
        # Load the model and re-run word mappings
        word_to_index, index_to_word, text_data = process_files(args.data_path)
        vocab_size = len(word_to_index)
        model = load_model(args.model_path, vocab_size, args.embedding_dim, args.hidden_dim)
        print("Loaded model. Enter seed text to predict next words.")
        while True:
            seed_text = input("Enter seed text: ").strip().lower()
            predicted_sentence = predict_next_words(seed_text, model, word_to_index, index_to_word, args.num_words)
            print("Predicted sentence:", predicted_sentence)
    else:
        print("Please specify either -t (training option) with -d (data) or -m (pretrained model) with -d (data) for predictions.")