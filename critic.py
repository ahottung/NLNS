import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output


class VrpCriticModel(nn.Module):

    def __init__(self, hidden_size):
        super(VrpCriticModel, self).__init__()

        self.encoder = Encoder(4, hidden_size)
        self.encoder_2 = Encoder(4, hidden_size)

        self.fc1 = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static_input, dynamic_input_float):
        input = torch.cat((static_input.permute(0, 2, 1), dynamic_input_float.permute(0, 2, 1)), dim=1)

        hidden_1 = self.encoder(input)

        hidden_2 = self.encoder_2(input)
        static_sum = torch.tanh(torch.sum(hidden_2, 2).squeeze())
        static_sum = static_sum.unsqueeze(2).expand_as(hidden_1)
        hidden = torch.cat((hidden_1, static_sum), dim=1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output
