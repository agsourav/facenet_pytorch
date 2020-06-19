from utils import *
from models.inception_resnet import *
import torch
import torch.nn as nn
from loss import *
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from train_inception import *

args = arg_parse()
root_path = args.rootdir
batch_size = int(args.batch)
epochs = int(args.epochs)
lr_in = float(args.lr_rate_inception_module)
lr_cl = float(args.lr_rate_classification_module)
train = int(args.train)
checkpoint = int(args.checkpoint)
check_path = args.checkpoint_path
classes = np.array(os.listdir(root_path))
print(classes)
dataloader = load_dataset(root_path,batch_size)

model = InceptionResnetv1(classify = False,pretrained = 'casia-webface2')
for param in model.parameters():
            param.requires_grad = True
last_linear = model.last_linear
final_in_features = last_linear.in_features
final_out_features = 128
model.last_linear = nn.Linear(final_in_features, final_out_features, bias = False)
model.last_bn = nn.BatchNorm1d(final_out_features, eps = model.last_bn.eps, momentum = model.last_bn.momentum)
loss_func = TripletLoss()
opt = optim.SGD(model.parameters(), lr = lr_in)

#Classification module
classify = classification(final_out_features, len(classes))
loss_cl = nn.CrossEntropyLoss()
opt_cl = optim.SGD(classify.parameters(), lr = lr_cl)
            
#Inception module
if checkpoint == 1:
    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

losses = train_inception(dataloader, batch_size, model, loss_func, opt, False, epochs)    

#checkpointing
checkpoint_inception = {
    'learning_rate': lr_in,
    'batch_size': batch_size,
    'epochs': epochs,
    'losses': losses,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict()
}
torch.save(checkpoint_inception, check_path)

plt.figure()
plt.plot(losses)
plt.show()

