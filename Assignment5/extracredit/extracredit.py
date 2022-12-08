import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def trainmodel():
    # Well, you might want to create a model a little better than this...
    model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=1))

    # ... and if you do, this initialization might not be relevant any more ...
    model[1].weight.data = initialize_weights()
    model[1].bias.data = torch.zeros(1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    learning_rate = 1e-8
    # loss_function = torch.nn.Linear()
    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)

    for epoch in range(5000):
        for x,y in trainloader:
            if epoch % 50 == 0:
                print(epoch)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad

    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()