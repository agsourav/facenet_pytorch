from __future__ import print_function, division
from triplet_selection import *

def train_inception(dataloader, batch_size, net, criterion, optimizer, use_gpu, num_epochs=100):
    losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)

        ts = TripletSelection(dataloader)
        anchors, positives, negatives = ts.tripletSelection()
        data = torch.cat((anchors, positives, negatives), dim = 0)
        output = net(data.float())
        loss = criterion(output[:batch_size], output[batch_size:2*batch_size], output[2*batch_size:3*batch_size])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('loss: {0:2.5f}'.format(loss.item()))
    return losses

def train_classification(dataloader, batch_size, model_inc, net, criterion, optimizer, use_gpu, num_epochs = 100):
    dataiter = iter(dataloader)
    losses = []
    for epoch in range(num_epochs):
        images, labels = dataiter.next()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)
        feature_out = model_inc(images)
        output = net(feature_out)

        pred = torch.argmax(output, dim = 1)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('loss: {0:2.5f}'.format(loss.item()))
    
    return losses
        