import nltk
import codecs
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import argparse
import torch.optim as optim

def process_files(f):
    """
    Parameters:
    f (str): The file path of the input file.
    """
    print("Processing file...")
    with codecs.open(f, 'r', 'utf-8') as text_file:
        text = text_file.read().encode('utf-8').decode() # Maintain capitalization
    words = nltk.word_tokenize(text)
    punctuation_pattern = re.compile(r'[^\w\s]')
    words = [re.sub(punctuation_pattern, '', word.lower()) for word in words]
    char_set = sorted(set(''.join(words)))
    char_to_index = {ch: i for i, ch in enumerate(char_set)}
    index_to_char = {i: ch for i, ch in enumerate(char_set)}
    word_set = sorted(set(words))
    word_to_index = {word: i for i, word in enumerate(word_set)}
    index_to_word = {i: word for i, word in enumerate(word_set)}
    return words, char_to_index, index_to_char, word_to_index, index_to_word

def create_input_output_pairs(words, char_to_index, word_to_index):
    input_sequences = []
    output_vectors = []
    for word in words:
        for i in range(1, len(word)):
            input_seq = word[:i]
            output_word = word
            input_seq = [char_to_index[ch] for ch in input_seq]
            input_sequences.append(input_seq)
            output_vectors.append(word_to_index[output_word])
    return input_sequences, output_vectors

## Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded) # do not make use of hidden layers
        # we extract the output of the last time step for each element in the batch. 
        output = self.fc(output[:, -1, :]) # [batch_size, seq_len, vocab_size] 
        return output
    

def train_model(input_sequences, output_vectors, char_to_index, word_to_index, embedding_dim=50, hidden_dim=128, epochs=100, learning_rate=0.01):
    vocab_size = len(word_to_index)  # output dimension is the number of unique words
    
    model = RNNModel(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Ensure all sequences are of the same length by padding
    max_len = max(len(seq) for seq in input_sequences)
    input_sequences = [torch.tensor(seq, dtype=torch.long) for seq in input_sequences]
    input_sequences = nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=0)
    
    output_vectors = torch.tensor(output_vectors, dtype=torch.long)
    print("Training starts for", input_sequences.shape, output_vectors.shape)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(input_sequences)
        loss = criterion(outputs, output_vectors)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

### helper functions

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, vocab_size, embedding_dim, hidden_dim):

    model = RNNModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(path))

    # Set the model to evaluation mode
    model.eval()
    return model

def suggest_words_RNN(seed_text, model, char_to_index, index_to_word):
    model.eval()
    #max_len = max(len(word) for word in word_to_index.keys()) 
    with torch.no_grad():
        sequence = torch.tensor([char_to_index[char] for char in seed_text], dtype=torch.long).unsqueeze(0)
        #sequence = torch.nn.functional.pad(sequence, (max_len - len(sequence), 0), 'constant', 0)
        output = model(sequence)
        top_3_words = []
        _, predicted_index = torch.max(output, 1)
        _, second_index = torch.topk(output, k=2, dim=1)
        _, third_index = torch.topk(output, k=3, dim=1)
        top_3_words.append(index_to_word[predicted_index.item()])
        top_3_words.append(index_to_word[second_index[:, 1].item()])
        top_3_words.append(index_to_word[third_index[:, 1].item()])
        return top_3_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or use an RNN for word prediction.")
    parser.add_argument('--train', '-t', action='store_true', help="Train a new model.")
    parser.add_argument('--model_path', '-m', type=str, help="path to save or load the model.")
    parser.add_argument('--data_path', '-d', type=str, help="data path")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training.")
    parser.add_argument('--embedding_dim', type=int, default=100, help="Dimension of word embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=150, help="Dimension of RNN hidden states.")

    args = parser.parse_args()

    if args.train and args.data_path:
        # Load and preprocess the text data
        words, char_to_index, index_to_char, word_to_index, index_to_word = process_files(args.data_path)
        input_sequences, output_words = create_input_output_pairs(words, char_to_index, word_to_index)
        # Train the model
        #print("before training: " , input_sequences[:10])
        #print("output words", output_words[:10])
        model = train_model(input_sequences, output_words, char_to_index, word_to_index, args.embedding_dim, args.hidden_dim, args.epochs)
        save_model(model, args.model_path)
        print(f"Model trained and saved at {args.model_path}")

    elif args.model_path and args.data_path:
        # Load the model and re-run word mappings
        #word_to_index, index_to_word, text_data = process_files(args.data_path)
        words, char_to_index, index_to_char, word_to_index, index_to_word = process_files(args.data_path)
        vocab_size = len(word_to_index)
        model = load_model(args.model_path, vocab_size, args.embedding_dim, args.hidden_dim)
        print("Loaded model. Enter characters to predict next words.")

        while True:
            seed_text = input("Enter seed text: ").strip().lower()
            if seed_text == "quit":
                break
            words = suggest_words_RNN(seed_text, model, char_to_index, index_to_word)
            print(f"\n1 - {words[0]}\n2 - {words[1]}\n3 - {words[2]}\n")
            print("Enter 'quit' to exit.\n")
            
    else:
        print("Please specify either -t (training option) with -d (data) and -m (model path) OR -m (pretrained model) with -d (data) for predictions.")