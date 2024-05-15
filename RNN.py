import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# Sample text data
text_data = ["the quick brown fox jumps over the lazy dog",
             "the lazy dog sleeps all day",
             "all dogs are good pets"]

# Tokenize the text
word_to_index = {'<PAD>':0}
index_to_word = {0:'<PAD>'}
for sentence in text_data:
    for word in sentence.split():
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
            index_to_word[len(word_to_index)] = word

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

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output)
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
    output = model(X_train) # [batch_size, seq_len, vocab_size]

    ################## Required Debugging due to dimensions mismatch ##################
    
    #loss = criterion(y_train.view(-1),output.view(-1, vocab_size))
    loss = criterion(output.squeeze(), y_train.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Function to predict next word
def predict_next_word(seed_text, model):
    model.eval()
    with torch.no_grad():
        sequence = [word_to_index[word] for word in seed_text.split()]
        sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        output = model(sequence)
        _, predicted_index = torch.max(output[:, -1, :], 1)
        predicted_word = index_to_word[predicted_index.item()]
        return predicted_word

# Test the model
seed_text = "the quick brown"
next_word = predict_next_word(seed_text, model)
print("Next word:", next_word)
