import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0")

start0 = time.time()
# =============================================================================
# Read simulation or experimental trajectory
# =============================================================================
infile='./DATA4s/xvyw1beta9.5gammax1.0gammay1.0epsln1.0sgma1.0A1.0x01.122w0.8B0.15a1.0_h0.01_mix1.txt'
input_x, _=np.loadtxt(infile, unpack=True)


# =============================================================================
# Parameters and dictionaries defined here.
# =============================================================================
# Number of bins and smoothen length
num_bins=4
sm_length=20
threshold=20

# Sequence length and shift in step between past (input) & future (output)
sequence_len = 100
shift=1

# Training parameters
batch_size=64
EPOCHS = 500
lr=0.001

# Constraint parameters for fixed penalty and augmented methods.
mu = 1        # Penalty factor
sigma = 1.0   # Penalty update factor
mu_max=1000   # Penalty safeguard

# Model parameters
embedding_dim = 8
rnn_units = 64

# Empty lists for storing losses.
loss1s=[]
loss2s=[]
constraints=[]


# =============================================================================
# These classes are used to preprocess the trajectory.
# =============================================================================
def running_mean(x, N):
    """
    Use convolution to do running average.
    """
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def find_nearest(key_arr, target):
    """
    key_arr: array-like, storing keys.
    target: the value which we want to be closest to.
    """
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


# =============================================================================
# These classes are used to build pytorch-compatible datasets.
# =============================================================================
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

# =============================================================================
# These classes are used for building models and making prediction.
# =============================================================================
class NLP(nn.Module):
    """
    This class defines the language model.
    """
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
    """
    Generate prediction from the language model.
    """
    input_eval = torch.tensor([char2idx[s] for s in start_string], device=device)
    
    text_generated = np.empty(1)
    for i in range(40000):
        input_eval = input_eval[np.newaxis, ...]    # add a dimension for batch=1.
        prediction=model(input_eval)
        logits=prediction
        p=torch.nn.functional.softmax(logits,dim=-1) 
        predicted_id=torch.multinomial(p[0,-1], 1)  # take first batch and last prediction
        
        input_eval = predicted_id
        
        text_generated = np.vstack((text_generated, idx2char[predicted_id].tolist()))

    return text_generated


# =============================================================================
# Data preprocessing
# =============================================================================
X = [2.0, 0.5, -0.5, -2.0]                         # The x-position of metastable states.
input_x = running_mean(input_x, sm_length)         # smooothen data.
idx_x = map(lambda x: find_nearest(X, x), input_x) # convert to three bins.
idx_2d=list(idx_x)                                 # For 2-d: list(zip(idx_x, idx_y))
idx_2d = Rm_peaks_steps(idx_2d)                    # remove peaks and short steps
text = idx_2d

all_combs = [i for i in range(num_bins)]
vocab=sorted(all_combs)
vocab_size = len(vocab)

# Sample state-0 and state-1 by known probability distribution.
prob4s = np.array([0.27499667, 0.23632767, 0.23187383, 0.25680183])
p2=prob4s[2:]/np.sum(prob4s[2:])
for i in range(len(text)):
    if text[i]<=1:
        text[i]=np.random.choice(2, 1, p=p2)

# Create a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])


# =============================================================================
# Generate NN vector for constraint term.
# =============================================================================
in_vec = text_as_int
vp = in_vec+1
vm = in_vec-1
nn_vec = np.zeros((in_vec.size, in_vec.max()+1), dtype='float32')
nn_vec[ np.where(vp<2), vp[np.where(vp<2)] ] = 1.
nn_vec[ np.where((vm>=0) & (vm <2)), vm[np.where((vm>=0) & (vm <2))] ] = 1



# =============================================================================
# Create pytorch dataset
# =============================================================================
dataset = seq_data(text_as_int, sequence_len, shift)
sampler=FixedSampler(dataset)
dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)

dataset_nn = seq_data(nn_vec, sequence_len, shift)
dataset_nn = DataLoader(dataset_nn, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)


# =============================================================================
# Model
# =============================================================================
model = NLP(vocab_size, embedding_dim, rnn_units).to(device)
print(model)

weight_for_loss=torch.tensor([0,1,1,1]).float().cuda() 
loss_fn = nn.CrossEntropyLoss(weight=weight_for_loss)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# =============================================================================
# Training
# =============================================================================
for epoch in range(EPOCHS):
    
    mu=min(sigma*mu, mu_max)
    for (batch_X_train, batch_Y_train), (nn_X_train, _) in zip(dataset, dataset_nn):
        
        batch_X_train = batch_X_train.to(device)
        nn_X_train = nn_X_train.to(device)
        batch_Y_train = batch_Y_train.to(device)
        y_pred = model(batch_X_train)
        y=batch_Y_train.to(device)
        
        p=torch.nn.functional.softmax(y_pred, dim=-1)
        
        loss1 = loss_fn(y_pred.view(-1, vocab_size), y.view(-1))
        loss2 = 0.5*mu*(torch.sum(p*nn_X_train)-1)**2

        loss = loss1 + loss2
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    constraint = (torch.sum(p*nn_X_train)-1).cpu().detach().numpy()
    loss1s.append(loss1.item())
    loss2s.append(loss2.item())
    constraints.append(constraint)
    print(epoch, loss.item())
    
loss_log=np.vstack((loss1s,loss2s,constraints))
np.savetxt('loss_log_mu{}.txt'.format(mu), loss_log)  # save loss in log file


# =============================================================================
# Prediction
# =============================================================================
text = idx_2d[:10000] # initialized the model with 10000 characters
prediction=generate_text(text)
print ('Time taken for total {} sec\n'.format(time.time() - start0))

np.savetxt('prediction_mc{}'.format(mu), prediction[1:]) # save prediction
print("Done")
