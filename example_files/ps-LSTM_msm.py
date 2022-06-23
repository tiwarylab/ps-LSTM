import os
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

cuda = torch.device("cuda:0")
cpu = torch.device("cpu")

# ----------------------------------------------- #
# Parameters
# ----------------------------------------------- #
# Length of trajectory
MAX_ROWS=10000000
print('MAX_ROWS: {}'.format(MAX_ROWS))

# Number of states
vocab_size=3
print('vocab_size: {}'.format(vocab_size))

# Sequence length and shift in step between past (input) & future (output)
sequence_len =35 
shift=1
print('sequence_len: {}'.format(sequence_len))
print('shift: {}'.format(shift))

# Training parameters
batch_size=64
EPOCHS = 10
lr=0.001
print('batch_size: {}'.format(batch_size))
print('EPOCHS: {}'.format(EPOCHS))
print('lr: {}'.format(lr))

# Model parameters
embedding_dim = 8
rnn_units = 128
print('embedding_dim: {}'.format(embedding_dim))
print('rnn_units: {}'.format(rnn_units))

# Weights and loss function for each region
cel=nn.CrossEntropyLoss()

# Losses
loss_arr = []



# ----------------------------------------------- #
# Prediction methods
# ----------------------------------------------- #
def generate_text_FB(start_string, length):
    """
    Feed back type prediction: Generate prediction from the language model.
    """
    global model
    model = model.to(cpu)

    input_eval = torch.tensor([s for s in start_string], device=cpu).long()

    hidden = model.init_hidden(1,cpu)
    text_generated = []
    for i in range(length):
        prediction, hidden = model(input_eval.view(1,-1), hidden)  # add a dimension for batch=1.
        logits = prediction
        p = torch.nn.functional.softmax(logits, dim=-1)            # take first batch
        predicted_id = torch.multinomial(p[0,-1], 1)

        input_eval = predicted_id
        text_generated.append(predicted_id.item())

    return np.array(text_generated)


# ----------------------------------------------- #
# Model classes
# ----------------------------------------------- #
class FixedSampler(torch.utils.data.sampler.Sampler):
    """
    Customized sampler with random permutation defined in __init__ instead of __iter__.
    """
    def __init__(self, data):
        self.num_samples = len(data)
        self.idx_arr=np.random.permutation(self.num_samples)

    def __iter__(self):
        return iter(self.idx_arr)

    def __len__(self):
        return self.num_samples
    
class seq_data(Dataset):
    """
    Create dataset (x,y) where x=past sequence, y=one-step future.
    """
    def __init__(self, traj, seq_length, shift):
        self.traj = traj
        self.seq_length = seq_length
        self.shift = shift
    
    def __len__(self):
        return self.traj[self.shift:].shape[0]//self.seq_length
    
    def __getitem__(self, idx):
        x = self.traj[:-self.shift][idx*self.seq_length:(idx+1)*self.seq_length]
        y = self.traj[self.shift:][idx*self.seq_length:(idx+1)*self.seq_length]
        return x, y

class NLP(nn.Module):
    """
    LSTM Language model.
    """
    def __init__(self, input_dim, embedding_dim, rnn_units):
        
        super(NLP, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = rnn_units
        self.num_layers = 1
        
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.input_dim)
    
    def init_hidden(self, batch_size, device):
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
    
    def forward(self, input, hidden):
        
        batch_size = input.shape[0]
        embedding_out = self.embedding(input)
        lstm_in = embedding_out.view(batch_size, input.shape[1], self.embedding_dim)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        y_pred = self.linear(lstm_out)
        
        return y_pred, hidden


# ----------------------------------------------- #
# Training for sequence transformer
# ----------------------------------------------- #
def train(i):
    """
    Training.
    """
    global model
    start_t = time.time()
    model = NLP(vocab_size, embedding_dim, rnn_units).to(cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('{}-th training: '.format(i))
    print('Epoch  loss_tot')
    for epoch in range(EPOCHS):

        for (batch_X_train, batch_Y_train) in dataset:

            batch_X_train = batch_X_train.to(cuda)
            y = batch_Y_train.to(cuda)

            hidden = model.init_hidden(batch_size, cuda)
            y_pred, _ = model(batch_X_train, hidden)
            p = torch.nn.functional.softmax(y_pred, dim=-1)

            loss = cel(y_pred.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_arr.append(loss.item())
        print('{}  {:.5f}'.format(epoch, loss.item()) )

    print ('Time taken for training {} sec\n'.format(time.time() - start_t))
    
    
# ----------------------------------------------- #
# Save model state
# ----------------------------------------------- #
def save(i):
    """
    Save model state/parameters
    """
    # Saved directory
    save_dir='./Output/{}/'.format(i)
    # Create directory
    os.mkdir(save_dir)
    
    torch.save( model.state_dict(), os.path.join(save_dir, 'model_save') )
    
    np.save( os.path.join(save_dir, 'prediction'), prediction )
    

# ----------------------------------------------- #
# Obtaining transformed sequence
# ----------------------------------------------- #
def predict(i):
    """
    Make prediction and save
    """
    global prediction
    start_p = time.time()
    
    prediction=generate_text_FB( input_x[:100], 300000 )
    print ('Time taken for prediction {} sec\n'.format(time.time() - start_p))
    
    
    
# ----------------------------------------------- #
# Read data
# ----------------------------------------------- #
infile = '../../DATA/Train/Markov3s/markov3s-eric.npy'
input_x = np.load(infile)


# ----------------------------------------------- #
# Create dataset
# ----------------------------------------------- #
dataset = seq_data(input_x, sequence_len, shift)
sampler=FixedSampler(dataset)
dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)


# ----------------------------------------------- #
# Compute
# ----------------------------------------------- #
for i in range(100):
    
    train(i)
    predict(i)
    save(i)
    
print('Done')
