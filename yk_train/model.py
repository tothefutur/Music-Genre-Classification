import torch.nn
from torch import nn
from torch.utils.data import DataLoader

# lr, epochs, batch_size = 0.01, 500, 128
# epochs:500 accuracy is about 42%
model_linear = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)

# lr, epochs, batch_size = 0.01, 500, 128
# epochs: 300 accuracy is about 50%
# epochs: 500 accuracy is about 55%
# epochs: 1000 accuracy is about 60%
model_linear_more_features_40 = nn.Sequential(
    nn.Linear(40, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)

# in_features = 57, loss下降，但是accuracy始终在0.1左右
model_linear_more_features_57 = nn.Sequential(
    nn.Linear(28, 2048),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 10)
)

def test_acc(model, test_loader, device):
    if isinstance(model, nn.Module):
        model.eval()

    correct_num = 0
    total_num = 0
    total_loss = 0
    l = 0
    loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in test_loader:
            x.to(device)
            y.to(device)
            prediction_y = model(x)
            prediction_y = torch.squeeze(prediction_y)
            l = loss(prediction_y, y)
            total_loss += l.item()
            correct_num += (prediction_y.argmax(1) == y).sum().item()
            total_num += y.numel()

    return correct_num / total_num

def train_model(dataset, model, epochs=100, lr=0.1, batch_size=128,
                device=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_dataset = dataset.trainDataset
    test_dataset = dataset.testDataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    for epoch in range(epochs):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            prediction_y = model(x)
            prediction_y = torch.squeeze(prediction_y)
            l = loss(prediction_y, y)
            l.backward()
            optimizer.step()
            # print(prediction_y)

        if epoch % 5 == 0:
            test_accuracy = test_acc(model, test_loader, device)
            print(f"epoch {epoch}, test accuracy: {test_accuracy}")
            print(f"epoch {epoch}, loss: {l.item()}")
