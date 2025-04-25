import torch
import torch.nn as nn
import numpy as np

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=8):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AQIPredictor:
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=8):
        self.model = LSTMPredictor(input_dim, hidden_dim, num_layers, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def train(self, train_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                outputs = self.model(batch_x)
                predictions.append(outputs.numpy())
        return np.concatenate(predictions) 