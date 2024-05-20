import nltk
import codecs
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import argparse


def process_files(f):
    """
    Processes the file f.
    """
    word_to_index = {'<PAD>':0}
    index_to_word = {0:'<PAD>'}
    punctuation_pattern = re.compile(r'[^\w\s]')
    with codecs.open(f, 'r', 'utf-8') as text_file:
        text = reader = text_file.read().encode('utf-8').decode() #Maintain capitalization
    
    cleaned_sentences = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.replace('\r\n', ' ')
        sentence = re.sub(punctuation_pattern, '', sentence)
        cleaned_sentences.append(sentence)
        for word in sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word[len(word_to_index)] = word
    return word_to_index, index_to_word, cleaned_sentences

# Process the files
word_to_index, index_to_word, text_data = process_files('kafka.txt')

# Convert text data to sequence data
input_sequences = []
for line in text_data:
    token_list = line.split()
    sequence = [word_to_index[word] for word in token_list]
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences with 0s in front of the words 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array([[0] * (max_sequence_len - len(i)) + i for i in input_sequences])

# Convert to PyTorch tensors
# X_train is the input sequence, y_train is the following word. e.g. "the" -> "quick", "the quick" -> "brown" etc
X_train = torch.tensor(input_sequences[:, :-1], dtype=torch.long)
y_train = torch.tensor(input_sequences[:,-1], dtype=torch.long)

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

# Define model parameters
vocab_size = len(word_to_index)
embedding_dim = 100
hidden_dim = 150

# Instantiate the model
model = RNNModel(vocab_size, embedding_dim, hidden_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train) 
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
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
def predict_next_words(seed_text, model, num_words):
    model.eval()
    for _ in range(num_words):
            input_seq = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
            output = model(input_seq)
            _, predicted_index = torch.max(output, 1)
            predicted_word = index_to_word[predicted_index.item()]
            sequence.append(predicted_index.item())
        
    # Convert indices back to words
    predicted_sequence = [index_to_word[idx] for idx in sequence]
    
    return ' '.join(predicted_sequence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or use an RNN for word prediction.")
    parser.add_argument('--train', action='store_true', help="Train a new model.")
    parser.add_argument('--predict', action='store_true', help="Predict next words using a loaded model.")
    parser.add_argument('--text', type=str, help="Text data for training.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training.")
    parser.add_argument('--embedding_dim', type=int, default=100, help="Dimension of word embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=150, help="Dimension of RNN hidden states.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument('--num_words', type=int, default=5, help="Number of words to predict.")

    args = parser.parse_args()

    if args.train and args.text:
        # Load and preprocess the text data
        with open(args.text, 'r') as file:
            text_data = file.readlines()
        text_data = [line.strip() for line in text_data]
        
        # Train the model
        model, word_to_index, index_to_word = train_model(
            text_data, args.embedding_dim, args.hidden_dim, args.epochs, args.learning_rate
        )
        print("Model trained and saved as 'rnn_model.pth'.")

    elif args.predict:
        # Load the model
        word_to_index, index_to_word = create_vocabulary(text_data)  # Replace with your vocabulary creation
        vocab_size = len(word_to_index)
        model = RNNModel(vocab_size, args.embedding_dim, args.hidden_dim)
        model.load_state_dict(torch.load('rnn_model.pth'))
        model.eval()
        
        print("Loaded model. Enter seed text to predict next words.")
        while True:
            seed_text = input("Enter seed text: ").strip().lower()
            predicted_sentence = predict_next_words(seed_text, model, word_to_index, index_to_word, args.num_words)
            print("Predicted sentence:", predicted_sentence)
    else:
        print("Please specify either --train with --text for training or --predict for prediction.")