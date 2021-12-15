from numpy.typing import _16Bit
import torch
from torch import nn
from torch.nn import init

class Sequential_model(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size, batch_first):
        super(Sequential_model, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size =  hidden_size
        self.output_size = output_size
        self.drop_out = 0.5

        # feature_dim, hidden_dim
        self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=batch_first, dropout=self.drop_out)
        self.rnn2 = nn.LSTM(hidden_size, hidden_size * 2, batch_first=batch_first, dropout=self.drop_out)
        self.ln1 = nn.Linear(hidden_size * 2 * seq_len, output_size)
        self.softmax = nn.LogSoftmax(dim=output_size)

        # self.init_param()

    def forward(self, input):
        output1, (hn1, cn1) = self.rnn1(input)
        output2, (hn2, cn2) = self.rnn2(output1)
        output2 = output2.reshape(output2.shape[0], -1)
        output = self.ln1(output2)
        return output

    # def init_parm(self):
    #     init.normal_()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Sequential_model(seq_len=100, input_size=128, hidden_size=256, output_size=5, batch_first=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # batch * seqlen * feature_dim
    input = torch.randn(5, 100, 128).to(device)
    label = torch.tensor([1,0,2,3,4]).to(device)
    # output shape : batch * 5
    output = model(input)
    criterion = nn.NLLLoss().to(device)

    for epoch in range(10):
        # for i in range(len(input)):
        output = model(input)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()

    print(output.shape)

