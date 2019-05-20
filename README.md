# Blood Cell Detection Using Deep Learning

In this work, I have created a patch-based sliding window approach for detecting White Blood Cells in Blood Images. LeNet was chosen as the Convolutional Neural Network architecture. The input to the network is the image of a patch of size 81x81, whereas the output is a 'Class Image' which assigns a label to the pixel value for the image centered around that pixel. 

Steps:

1. pkl_generator.ipynb: In order to generate the output labels for each of the pixels of the images, we obtain the bounding boxes for the object of interest and assign all pixels within that bounding box the class label. This image is then stored as a pickle file. 
2. patch_generator.ipynb: Once we have the pixel-wise class labels, we can generate patches for training. For each class, we arbitarily select pixels belonging to that class from the image and take an 81x81 crop around that pixel and store it for training the model.
3. LeNet.ipynb: This script contains the actual implementation of the model. 

The folder 'prob_maps' contains the resultant images obtained. Each subfolder contains the original image, the individual probability image for each of the individual classes and unalikability maps. 

For generating the output image, we will simply iterate over the entire image pixel-by-pixel, take a crop around that pixel and pass it through the model. The output label obtained will be the label for that particular pixel. 

In order to compare the results obtained, we use an Autoencoder+SVM model. The Autoencoder is used for extracting features from the images, which are then inputted into the SVM. The scripts and results are stored in the folder called 'svm'. 

### Unalikability Maps

We observe that the LeNet model shows patches of platelets where there are instances of WBCs. There is also some amount of ambiguity in the boundary regions between Red Blood Cells and Background.

In order to analyse this observation, we calculate the unalikability parameter for every pixel, which is given by:

1-\sum _{{i=1}}^{K}p_{i}^{2}

A complete analysis of the work is presented in the report 'Report_WBC.pdf'
