import torch
import torch.nn as nn

class FastRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FastRNN, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(input_size, hidden_size) 
        self.U_h = nn.Linear(hidden_size, hidden_size)  
        self.b_h = nn.Parameter(torch.zeros(hidden_size)) 

        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.activation = nn.Tanh()
    
    def forward(self, x):
        '''
        x: input sequence
        h: hidden state
        h* = tanh(x * W + h_{t-1} * U +b)
        h = alpha * h* + beta * h{t-1}
        '''
        batch_size, seq_len, _ = x.size()
        # batch_size, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  
            h_tilde = self.activation(self.W_h(x_t) + self.U_h(h) + self.b_h)
            
            h_t = self.alpha * h_tilde + self.beta * h  
            h = h_t

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        return outputs

class CNNRNNModel(nn.Module):
    def __init__(self, hidden_dim, num_classes=2):
        super(CNNRNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )

        self.rnn = FastRNN(8 * 20 * 12, hidden_dim)  
        self.fc = nn.Linear(hidden_dim, num_classes) 

    def forward(self, x):
        # batch_size, seq_len, c, h, w = x.size()
        batch_size, c, h, w = x.size()
        cnn_features = []

        # for i in range(seq_len):
        cnn_out = self.cnn(x[:, :, :, :]) 
        cnn_out = cnn_out.view(batch_size, -1)  
        cnn_features.append(cnn_out)

        cnn_features = torch.stack(cnn_features, dim=1)  
        print(cnn_features.size())

        rnn_out = self.rnn(cnn_features)

        output = self.fc(rnn_out)  

        print(output.size())
        return output
