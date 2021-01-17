import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0")



infile='./DATA_Linear/xvyw1beta9.5gammax1.0gammay1.0epsln1.0sgma1.0A1.0x01.122w0.8B0.15a1.0_h0.01_mix1.txt'
input_x, _=np.loadtxt(infile, unpack=True)



num_bins=3
sm_length=20
def running_mean(x, N):
    """Use convolution to do running average."""
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def find_nearest(key_arr, target):
    """key_arr: array-like, storing keys.
       target: the value which we want to be closest to."""
    idx=np.abs(key_arr-target).argmin()
    return idx

def Rm_peaks_steps(traj):
    global threshold
    """
    Remove sudden changes in the trajectory such as peaks and small steps.
    In this method, I used gradient to identify the changes. If two nonzero
    gradients are too close (< threshold), we shall take this range as noise.
    """
    traj=np.array(traj)
    grad_traj=np.gradient(traj) # gradient of trajectory
    idx_grad=np.where(grad_traj!=0)[0]
    threshold=20
    idx0=idx_grad[0]
    for idx in idx_grad:
        window=idx-idx0
        if window <= 1: # neighbor
            continue
        elif window > 1 and window <= threshold:
            traj[idx0:idx0+window//2+1]=traj[idx0]
            traj[idx0+window//2+1:idx+1]=traj[idx+1]
            idx0=idx
        elif window > threshold:
            idx0=idx
    return traj

def nn_vec(idx_s):
    """
    Compute nearest-neighbor vector.
    """
    nn_vec=torch.zeros(num_bins)
    if idx_s-1 >= 0:
        nn_vec[idx_s-1] = 1
    if idx_s+1 <= num_bins-1:
        nn_vec[idx_s+1] = 1
        
    return nn_vec


class seq_data(Dataset):
    
    def __init__(self, traj, seq_length, shift):
        self.traj = traj
        self.seq_length = seq_length
        self.shift = shift
    
    def __len__(self):
        return self.traj[self.shift:].shape[0]//self.seq_length
    
    def __getitem__(self, idx):
        x = self.traj[:-self.shift][idx*self.seq_length:idx*self.seq_length+self.seq_length]
        y = self.traj[self.shift:][idx*self.seq_length:idx*self.seq_length+self.seq_length]
        return x, y



class NLP(nn.Module):
    
    def __init__(self, input_dim, embedding_dim, rnn_units):
        
        super(NLP, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = rnn_units
        
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.input_dim)
    
    def forward(self, input):
        
        batch_size = input.shape[0]
        
        embedding_out = self.embedding(input)
        lstm_in = embedding_out.view(batch_size, input.shape[1], self.embedding_dim)
        lstm_out, hidden = self.lstm(lstm_in)
        y_pred = self.linear(lstm_out)
        
        return y_pred

    
def generate_text(start_string):
    
    input_eval = torch.tensor([char2idx[s] for s in start_string], device=device)
    
    text_generated = np.empty(1)
    for i in range(40000):
        input_eval = input_eval[np.newaxis, ...] # add a dimension for batch=1.
        prediction=model(input_eval)
        logits=prediction
        p=torch.nn.functional.softmax(logits,dim=-1) 
        predicted_id=torch.multinomial(p[0,-1], 1)  # take first batch and last prediction
        
        input_eval = predicted_id
        
        text_generated = np.vstack((text_generated, idx2char[predicted_id].tolist()))

    return text_generated


X = [1.5, 0, -1.5]
input_x = running_mean(input_x, sm_length) # smooothen data.
idx_x = map(lambda x: find_nearest(X, x), input_x) # convert to three bins.
idx_2d=list(idx_x) # list(zip(idx_x, idx_y))
idx_2d = Rm_peaks_steps(idx_2d) # remove peaks and short steps
text = idx_2d

all_combs = [i for i in range(num_bins)]
vocab=sorted(all_combs)


# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])



# The maximum length sentence we want for a single input in characters
# Penalty factor
mu = 1.
# Penalty update factor
sigma = 1.1
# Penalty safeguard
mu_max=5

EPOCHS = 20
sequence_len = 100
shift=1
batch_size=64

dataset = seq_data(text_as_int, 100, 1)
dataset = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

vocab_size = len(vocab)
embedding_dim = 8
rnn_units = 32
batch_size=64

model = NLP(vocab_size, embedding_dim, rnn_units).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

output_t = torch.zeros((64, 100, num_bins)).to(device)
for epoch in range(EPOCHS):
    
    mu=min(sigma*mu, mu_max)
    for batch_X_train, batch_Y_train in dataset:
        
        batch_X_train = batch_X_train.to(device)
        batch_Y_train = batch_Y_train.to(device)
        y_pred = model(batch_X_train)
        y=batch_Y_train.to(device)
        
        for i in range(output_t.shape[0]):
            for j in range(output_t.shape[1]):
                output_t[i,j,:]=nn_vec(batch_X_train[i,j])
        
        p=torch.nn.functional.softmax(y_pred, dim=-1)
        
        loss = loss_fn(y_pred.view(-1, vocab_size), y.view(-1)) + 0.5*mu*(torch.sum(p*output_t)-1)**2
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(epoch, loss.item())
    
    
    

# generate prediction:
start0 = time.time()

text = idx_2d[:10000]

prediction=generate_text(text)

print ('Time taken for total {} sec\n'.format(time.time() - start0))

# Save prediction:
np.savetxt('prediction',prediction[1:])

print("Done")
