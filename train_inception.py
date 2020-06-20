from __future__ import print_function, division
from triplet_selection import *

def train_inception(dataloader, batch_size, net, criterion, optimizer, device, num_epochs=100):
    net.to(device)
    losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)

        ts = TripletSelection(dataloader, device)
        anchors, positives, negatives = ts.tripletSelection()
        data = torch.cat((anchors, positives, negatives), dim = 0)
        output = net(data.float())
        loss = criterion(output[:batch_size], output[batch_size:2*batch_size], output[2*batch_size:3*batch_size])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('loss: {0:2.5f}'.format(loss.item()))
    return losses

def train_classification(dataloader, batch_size, model_inc, net, criterion, optimizer, device, num_epochs = 100):
    model_inc.to(device)
    net.to(device)
    dataiter = iter(dataloader)
    losses = []
    for epoch in range(num_epochs):
        try:
            images, labels = dataiter.next()
        except:
            dataiter = iter(dataloader)
            images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)
        feature_out = model_inc(images)
        output = net(feature_out)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('loss: {0:2.5f}'.format(loss.item()))
    
    return losses
        