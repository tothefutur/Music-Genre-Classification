import numpy as np
import torch
import model
import process_data_simple
import net.AlexNet

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


lr, epochs, batch_size = 0.0001, 2000, 128
# dataset = process_data_simple.GTZANDataset(r"..\dataset\archive\Data\features_3_sec.csv", resize=)
dataset = process_data_simple.GTZANDataset_more_features(r"..\dataset\archive\Data\features_3_sec.csv", resize=224)
m = net.AlexNet.AlexNet()
print(m)
model.train_model(dataset=dataset, model=m, epochs=epochs, lr=lr, batch_size=batch_size, device=device)



