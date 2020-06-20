import torch
import numpy as np

class TripletSelection():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)
    def tripletSelection(self, device):
        images, labels = self.dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        grouping = {}
        if images.shape[0] < 2:
            print('Insufficient images, cannot train further')
            exit()
        else:
            for i, label in enumerate(labels):
                grouping[label.item()] = grouping.get(label.item(), [])
                grouping[label.item()].append(i)
            anchors = images
            positives = []
            negatives = []
            for i,image in enumerate(images):
                #positive image
                label = labels[i].item()
                choice = np.random.choice(grouping[label])
                positive = images[choice:choice+1]
                positives.append(positive)
                #negative image
                keys = list(grouping.keys())
                try:
                    keys.remove(label)
                    class_choice = np.random.choice(keys)
                    image_choice = np.random.choice(grouping[class_choice])
                    #print('anchor label: {0}\tnegative label: {1}, image_index: {2}'.format(label, class_choice, image_choice))
                    negative = images[image_choice:image_choice+1]
                    negatives.append(negative)
                except Exception as e:
                    print('Insufficient keys for negative sampling...')
                    print('-'*20)
                    print(e)
                    exit()

            positives = torch.cat(positives)
            negatives = torch.cat(negatives)
        return anchors, positives, negatives

