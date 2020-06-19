import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from requests.adapters import HTTPAdapter
import os

class Basic_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size = kernel_size,
            stride = stride, padding = padding,
            bias = False
        )
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps = 0.001,
            momentum = 0.1,
            affine = True
        )
        self.relu = nn.ReLU(inplace = False)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block35(nn.Module): # 5x Inception resnet_v2-A [output: 35*35*256]

    def __init__(self, scale = 1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = Basic_Conv2d(256, 32, kernel_size = 1, stride = 1)

        self.branch1 = nn.Sequential(
            Basic_Conv2d(256, 32, kernel_size = 1, stride = 1),
            Basic_Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        )

        self.branch2 = nn.Sequential(
            Basic_Conv2d(256, 32, kernel_size = 1, stride = 1),
            Basic_Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            Basic_Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        )

        self.conv = nn.Conv2d(96, 256, kernel_size = 1, stride = 1)     #(32 + 32 + 32: stacked branches)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.conv(out)
        out = out * self.scale + x      #residual network

        out = self.relu(out)
        return out

class Block17(nn.Module):

    def __init__(self, scale = 1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = Basic_Conv2d(896, 128, kernel_size = 1, stride = 1)

        self.branch1 = nn.Sequential(
            Basic_Conv2d(896, 128, kernel_size = 1, stride = 1),
            Basic_Conv2d(128, 128, kernel_size = (1, 7), stride = 1, padding = (0,3)),
            Basic_Conv2d(128, 128, kernel_size = (7, 1), stride = 1, padding = (3,0))
        )

        self.conv = nn.Conv2d(256, 896, kernel_size = 1, stride = 1)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)
        out = self.conv(out)
        out = out * self.scale + x

        out = self.relu(out)
        return out

class Block8(nn.Module):

    def __init__(self, scale = 1.0, noReLU = False):

        super().__init__()
        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = Basic_Conv2d(1792, 192, kernel_size = 1, stride = 1)
        
        self.branch1 = nn.Sequential(
            Basic_Conv2d(1792, 192, kernel_size = 1, stride = 1),
            Basic_Conv2d(192, 192, kernel_size = (1,3), stride = 1, padding = (0, 1)),
            Basic_Conv2d(192, 192, kernel_size = (3,1), stride = 1, padding = (1,0))
        )

        self.conv = nn.Conv2d(384, 1792, kernel_size = 1, stride = 1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)
        out = self.conv(out)
        out = out * self.scale + x

        if not self.noReLU:
            out = self.relu(out)

        return out

class Reduction_a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = Basic_Conv2d(256, 384, kernel_size = 3, stride = 2)

        self.branch1 = nn.Sequential(
            Basic_Conv2d(256,192, kernel_size = 1, stride = 1),
            Basic_Conv2d(192,192, kernel_size = 3, stride = 1, padding = 1),
            Basic_Conv2d(192,256, kernel_size = 3, stride = 2)
        )
        self.branch2 = nn.MaxPool2d(3, stride = 2)

    def forward(self,x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        return out                      #[output: 35x35x256 -> 17x17x896]

class Reduction_b(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            Basic_Conv2d(896, 256, kernel_size = 1, stride = 1),
            Basic_Conv2d(256, 384, kernel_size = 3, stride = 2)
        )

        self.branch1 = nn.Sequential(
            Basic_Conv2d(896, 256, kernel_size = 1, stride = 1),
            Basic_Conv2d(256, 256, kernel_size = 3, stride = 2)
        )

        self.branch2 = nn.Sequential(
            Basic_Conv2d(896, 256, kernel_size = 1, stride = 1),
            Basic_Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            Basic_Conv2d(256, 256, kernel_size = 3, stride = 2)
        )

        self.branch3 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3), 1)

        return out

class InceptionResnetv1(nn.Module):

    def __init__(self, pretrained = None, classify = False, num_classes = None, drop_prob = 0.6, device = None):
        super().__init__()

        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes
        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface2':
            tmp_classes = 10575
        elif pretrained is None and self.num_classes is None:
            raise Exception('At least one of "pretrained" or "num_classes" must be specified')
        else:
            tmp_classes = self.num_classes

        #define layers
        #stem of inception resnet v1
        self.conv2d_1a = Basic_Conv2d(3, 32, kernel_size = 3, stride = 2)
        self.conv2d_2a = Basic_Conv2d(32, 32, kernel_size = 3, stride = 1)
        self.conv2d_2b = Basic_Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool_3a = nn.MaxPool2d(3, stride = 2)
        
        self.conv2d_3b = Basic_Conv2d(64, 80, kernel_size = 1, stride = 1)
        self.conv2d_4a = Basic_Conv2d(80, 192, kernel_size = 3, stride = 1)
        self.conv2d_4b = Basic_Conv2d(192, 256, kernel_size = 3, stride = 2)
        # input[229x229x3] -> output[35x35x256]
    
        #5xInception resnet A (repeat network)
        self.repeat_1 = nn.Sequential(
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17)
        )
        #input[35x35x256] -> output[35x35x256]

        # reduction_a
        self.mixed_6a = Reduction_a()    # [35x35x256] -> [17x17x896]

        #10* Inception resent b
        self.repeat_2 = nn.Sequential(
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10)
        )
        # input[17x17x896] -> output[17x17x896]

        #reduction_b
        self.mixed_7a = Reduction_b()    #[17x17x896] -> [8x8x1792]

        # 5* Inception resent_c
        self.repeat_3 = nn.Sequential(
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20)
        )
        #input[8x8x1792] -> output[8x8x1792]

        self.block8 = Block8(noReLU=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #input[8x8x1792] -> output[1792]
        self.dropout = nn.Dropout(drop_prob)
        self.last_linear = nn.Linear(1792, 512, bias = False)
        self.last_bn = nn.BatchNorm1d(512, eps = 0.001, momentum = 0.1, affine = True)
        self.logits = nn.Linear(512, tmp_classes)

        if pretrained is not None:
            load_weights(self, pretrained)
        
        if self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)
        
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        # Arguments: x - torch.tensor -- batch of image tensors
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)

        print('log: stem integration successful\toutput shape: {0}'.format(x.shape))
        x = self.repeat_1(x)
        print('log: 5x Inception resnet A successful\toutput shape: {0}'.format(x.shape))
        x = self.mixed_6a(x)
        print('log: reduction layer A successful\toutput shape: {0}'.format(x.shape))
        x = self.repeat_2(x)
        print('log: 10x Inception resnet B successful\toutput shape: {0}'.format(x.shape))
        x = self.mixed_7a(x)
        print('log: reduction layer B successful\toutput shape: {0}'.format(x.shape))
        x = self.repeat_3(x)
        print('log: repeater inception C successful\toutput shape: {0}'.format(x.shape))
        x = self.block8(x)  #why this layer is added?
        x = self.avg_pool(x)
        print('log: average pooling successful\toutput shape: {0}'.format(x.shape))
        x = self.dropout(x)
        x = self.last_linear(x)
        print('log: last linear layer successful\toutput shape: {0}'.format(x.shape))
        x = self.last_bn(x)

        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p = 2, dim = 1)
        return x

class classification(nn.Module):
    def __init__(self,in_features, num_classes):
        super(classification, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.logits = nn.Linear(self.in_features, self.num_classes)
        #self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        x = self.logits(x)
        #x = self.softmax(x)
        return x

def load_weights(model, dname):
    if dname == 'vggface2':
        features_path = 'https://drive.google.com/uc?export=download&id=1cWLH_hPns8kSfMz9kKl9PsG5aNV2VSMn'
        logits_path = 'https://drive.google.com/uc?export=download&id=1mAie3nzZeno9UIzFXvmVZrDG3kwML46X'
    
    elif dname == 'casia-webface2':
        features_path = 'https://drive.google.com/uc?export=download&id=1LSHHee_IQj5W3vjBcRyVaALv4py1XaGy'
        logits_path = 'https://drive.google.com/uc?export=download&id=1QrhPgn1bGlDxAil2uc07ctunCQoDnCzT'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface2" ')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok= True)

    state_dict = {}
    for i, path in enumerate([features_path, logits_path]):
        cached_file = os.path.join(model_dir, '{}_{}.pt'.format(dname, path[-10:]))
        if not os.path.exists(cached_file):
            print('Downloading parameters ({}/2)'.format(i+1))
            s = requests.Session()
            s.mount('https://', HTTPAdapter(max_retries=10))
            r = s.get(path, allow_redirects = True)
            with open(cached_file, 'wb') as f:
                f.write(r.content)
        state_dict.update(torch.load(cached_file))

    # resolving naming mismatch
    ''' bug fix
    pretrained state dict contains key names which do not match 
    the param key names for the model

    fix : replacing the state_dict key names as per the model's param key names
    '''

    key_state_dict = list(state_dict.keys())
    model_key = list(model.state_dict().keys())
    for key in model_key:
        if key not in key_state_dict:
            id_ = model_key.index(key)
            key_state_dict[id_] = key
    state_dict_vals = list(state_dict.values())
    state_dict = dict(zip(key_state_dict, state_dict_vals))

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print('state dict keys matching error!!')
        print('--------------------------------')
        print(e)
        print('-------------------------------')
        exit()

def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home
