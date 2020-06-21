# facenet_pytorch
Google's facenet implementation in pytorch for face identification

# directory structure
* root
  * --datasets
    * --custom_images
      * --class_label1
        * --image1
        '
        '
        '
  * --models
    * --inception_resnet.py
  * --loss.py
  * --main.py
  * --train_inception.py
  * --triplet_selection.py
  * --utils.py
  
 # understanding files
 1. inception_resnet.py : implements inception_resnet_v1 in pytorch
      * functions()    : load_weights(model, dname)

                          -model to load the weights to
                          -dname : 'vggface2' / 'casia-webface2', pretrained weights to load
                       : get_torch_home()
                          -storing pretrained weights in the cache folder created in the home path
                           it helps to restore the weights if available in the cache, otherwise download them
2. loss.py              : implements TripletLoss class
                          -computes the triplet loss on anchor, positive and negative images
3. train_inception.py   : implements the training module for inception resnet
      * functions()    : train_inception(dataloader, batch_size, net, criterion, optimizer, use_gpu, num_epochs)

                          -training the inception_resnet module using triplet loss
4. triplet_selection.py : implements tripletselection class
      * functions()    : __init__()

                     : tripletSelection()

                          -loading batch of images and labels, creating a dictionary of key:value pairs
                            key: class label
                            value: list of indexes of image of key label
                          -computing triplet loss for loaded anchor images from dataloader, selected positives and negatives from 
                           grouping dictionary
5. utils.py            : utility functions for the module
     * functions()    : infer(model, image)

                          -model
                          -image
                        outputs: feature vector of inception module

                      : load_dataset(root, batchsize, shuffle)

                          -generate a dataloader
                      : arg_parse()
                      
                          parser to provide command line inputs
                            --rootdir : root directory containing images to train (eg: 'datasets/custom_images')
                            --lrin    : learning rate for inception resnet module
                            --lrcl    : learning rate for classification module
                            --bs      : batch size for training
                            --epochs  : number of epochs for training
                            --train   : 1 for training/ 0 for evaluation
6. main.py             : main file  

------------------------------------------------------------------------------------------
# Commands

## fine tuning inception_resnet 
### without checkout (when the model is being fine tuned the first time)
$ python main.py --lr 0.05 --bs 8 --train 1 --check 0 --ep 100 --train-module inception --rootdir datasets/custom_images

### with checkpoint
$ python main.py --lr 0.05 --bs 8 --train 1 --check 1 --ep 100 --train-module inception --rootdir datasets/custom_images

## training classification module
### training the first time
$ python main.py --lr 0.001 --bs 16 --train 1 --check 0 --ep 50 --train-module classification --rootdir datasets/custom_images

### training with checkpoint
$ python main.py --lr 0.001 --bs 16 --train 1 --check 0 --ep 50 --train-module classification --rootdir datasets/custom_images

## evaluation
$ python main.py --train 0

---------------------------------------------------------
# References

1. github's repo: akshaybahadur21/Facial-Recognition-using-Facenet

2. Google's Facenet paper

The inception resnet module is loaded using either 'casia-webface2' pretrained model or 'vggface2' pretrained inception resnet model, which can be selected at the time of initialising the model using the attribute "pretrained". 

## Note: 
The project is build for learning purposes and the custom images has been taken with due permission of my friends for a demo use case only. For all other practical purposes, this dataset must be avoided and instead other custom images must be used.

"I thank my fellow colleagues for allowing me to use their images for demo purposes"
      

