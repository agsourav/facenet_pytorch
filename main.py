from utils import *
from models.inception_resnet import *
import torch
import torch.nn as nn
from loss import *
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from train_inception import *
from inference import *

args = arg_parse()
root_path = args.rootdir
batch_size = int(args.batch)
epochs = int(args.epochs)
lr = float(args.lr_rate)
train = int(args.train)
checkpoint = int(args.checkpoint)
training_module = args.training_module

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')

classes = np.array(os.listdir(root_path))
print(classes)
dataloader = load_dataset(root_path,batch_size)

#inception module
model = InceptionResnetv1(classify = False,pretrained = 'casia-webface2')
for param in model.parameters():
            param.requires_grad = True
last_linear = model.last_linear
final_in_features = last_linear.in_features
final_out_features = 128
model.last_linear = nn.Linear(final_in_features, final_out_features, bias = False)
model.last_bn = nn.BatchNorm1d(final_out_features, eps = model.last_bn.eps, momentum = model.last_bn.momentum)
loss_func = TripletLoss()
opt = optim.SGD(model.parameters(), lr = lr)
#classification module
classify = classification(final_out_features, len(classes))
loss_cl = nn.CrossEntropyLoss()
opt_cl = optim.SGD(classify.parameters(), lr = lr, momentum = 0.9)

if train:

    if training_module == 'inception':
        previous_losses = []
        if checkpoint:
            checkpoint = torch.load('checkpoints/inception.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            previous_losses = checkpoint['losses']

        losses = train_inception(dataloader, batch_size, model, loss_func, opt, device, epochs)    
        
        previous_losses.extend(losses)
        plt.figure()
        plt.plot(previous_losses)
        plt.title('Inception: loss vs epoch')
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        plt.savefig('training_inception.png')
        plt.show()
        #checkpointing
        checkpoint_inception = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'losses': losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }
        torch.save(checkpoint_inception, 'checkpoints/inception.pt')

    elif training_module == 'classification':
        check = torch.load('checkpoints/inception.pt')
        model.load_state_dict(check['model_state_dict'])
        opt.load_state_dict(check['optimizer_state_dict'])
        previous_losses = []
        if checkpoint:
            checkpoint = torch.load('checkpoints/classification.pt')
            classify.load_state_dict(checkpoint['model_state_dict'])
            opt_cl.load_state_dict(checkpoint['optimizer_state_dict'])
            previous_losses = checkpoint['losses']

        losses = train_classification(dataloader, batch_size, model, classify, loss_cl, opt_cl, device, num_epochs = epochs)
        previous_losses.extend(losses)
        plt.figure()
        plt.plot(previous_losses)
        plt.title('Classification: loss vs epoch')
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        plt.savefig('training_classification.png')
        plt.show()
        #checkpointing
        checkpoint_classification = {
            'learning_rate': lr,
             'batch_size': batch_size,
             'epochs': epochs,
             'losses': losses,
             'model_state_dict': classify.state_dict(),
             'optimizer_state_dict': opt_cl.state_dict()
            }
        torch.save(checkpoint_classification, 'checkpoints/classification.pt')

    else:
        print('ERROR!! Incorrect training module specified\nselect either inception or \
        classification\n default: [inception]')
        exit()

elif not train:

    checkpoint_inception = torch.load('checkpoints/inception.pt')
    checkpoint_classification = torch.load('checkpoints/classification.pt')
    model.load_state_dict(checkpoint_inception['model_state_dict'])
    classify.load_state_dict(checkpoint_classification['model_state_dict'])
    dataloader = load_dataset(root_path,batch_size, shuffle = True)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    topk = 2
    pred = evaluate(images, labels, model, classify, topk)

    plt.figure()
    for i,image in enumerate(images):
        img = image.permute(1,2,0)
        plt.imshow(img)
        plt.title(classes[pred[i]])
        plt.show()


    
    
    

    
