# facenet_pytorch
Google's facenet implementation in pytorch for face identification

#directory structure
root
  --datasets
    --custom_images
      --class_label1
        --image1
        '
        '
        '
  --models
    --inception_resnet.py
  --loss.py
  --main.py
  --train_inception.py
  --triplet_selection.py
  --utils.py
  
 #understanding files
 inception_resnet.py : implements inception_resnet_v1 in pytorch
      functions()    : load_weights(model, dname)
                          -model to load the weights to
                          -dname : 'vggface2' / 'casia-webface2', pretrained weights to load
                     : get_torch_home()
                          -storing pretrained weights in the cache folder created in the home path
                           it helps to restore the weights if available in the cache, otherwise download them
loss.py              : implements TripletLoss class
                          -computes the triplet loss on anchor, positive and negative images
train_inception.py   : implements the training module for inception resnet
      functions()    : train_inception(dataloader, batch_size, net, criterion, optimizer, use_gpu, num_epochs)
                          -training the inception_resnet module using triplet loss
triplet_selection.py : implements tripletselection class
      functions()    : __init__()
                     : tripletSelection()
                          -loading batch of images and labels, creating a dictionary of key:value pairs
                            key: class label
                            value: list of indexes of image of key label
                          -computing triplet loss for loaded anchor images from dataloader, selected positives and negatives from 
                           grouping dictionary
utils.py            : utility functions for the module
     functions()    : infer(model, image)
                          -model
                          -image
                        outputs: feature vector of inception module
                      load_dataset(root, batchsize, shuffle)
                          -generate a dataloader
                      arg_parse()
                          parser to provide command line inputs
                            --rootdir : root directory containing images to train (eg: 'datasets/custom_images')
                            --lrin    : learning rate for inception resnet module
                            --lrcl    : learning rate for classification module
                            --bs      : batch size for training
                            --epochs  : number of epochs for training
                            --train   : 1 for training/ 0 for evaluation
main.py             : main file                         
      

