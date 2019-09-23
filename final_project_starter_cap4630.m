%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Summer 2019 
% Student Name: Tsz Shing Tsoi
%% Final Project - Starter code (2-class classifier: cats vs. dogs)

% Inspired by the example "Deep Learning for Pet Classification" 
% (Copyright 2016 The MathWorks, Inc.)

%% Part 1: Download, load and inspect Pre-trained Convolutional Neural Network (CNN)

% You will need to download a pre-trained CNN model for this example.
% There are several pre-trained networks that have gained popularity.

% Most of these have been trained on the ImageNet dataset, which has 1000
% object categories and 1.2 million training images[1]. "AlexNet" is one
% such model [2]. 

% See
% https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html 
% for a list of pre-trained networks available in MATLAB.

%% 1.1: Loading a pre-trained "AlexNet"

% Ensure that you have downloaded and installed the 
% "Deep Learning Toolbox Model for AlexNet Network" support package. 

% See https://www.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network
% and
% https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
% for additional information.
clc;
clear;
close all;

model = alexnet;

%% 1.3: Inspect the CNN's layers

model.Layers

% The intermediate layers make up the bulk of the CNN. These are a series
% of convolutional layers, interspersed with rectified linear units (ReLU)
% and max-pooling layers [2]. Following the these layers are 3
% fully-connected layers.

% The final layer is the classification layer and its properties depend on
% the classification task. In this example, the CNN model that was loaded
% was trained to solve a 1000-way classification problem. Thus the
% classification layer has 1000 classes from the ImageNet dataset. 

% Inspect the last layer
model.Layers(end)

% Number of class names for ImageNet classification task
numel(model.Layers(end).ClassNames)

% Analyze the AlexNet in more detail using the network analyzer 
% to display an interactive visualization of the network architecture 
% and detailed information about the network layers.
analyzeNetwork(model)

% Note that the CNN model is not going to be used for the original
% classification task. It is going to be re-purposed to solve a different
% classification task on the pets dataset.

%% 1.4: Inspect the network weights for the second convolutional layer

% Each layer of a CNN produces a response, or activation, to an input
% image. However, there are only a few layers within a CNN that are
% suitable for image feature extraction. The layers at the beginning of the
% network capture basic image features, such as edges and blobs. To see
% this, visualize the network filter weights from the first convolutional
% layer. This can help build up an intuition as to why the features
% extracted from CNNs work so well for image recognition tasks. Note that
% visualizing deeper layer weights is beyond the scope of this example. You
% can read more about that in the work of Zeiler and Fergus [4].

% Get the network weights for the second convolutional layer
w1 = model.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. 
figure
montage(w1)
title('First convolutional layer weights')

% Notice how the first layer of the network has learned filters for
% capturing blob and edge features. These "primitive" features are then
% processed by deeper network layers, which combine the early features to
% form higher level image features. These higher level features are better
% suited for recognition tasks because they combine all the primitive
% features into a richer image representation [5].

%% Part 2: Set up image data
%% 2.1: Load simplified dataset and build image store
rng(0,'twister');

dataFolder = './data/PetImages';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
disp (tbl)

% Use the smallest overlap set
% (useful when the two classes have different number of elements but not
% needed in this case)
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%% 2.2: Pre-process Images For CNN
% AlexNet can only process RGB images that are 227-by-227.
% To avoid re-saving all the images to this format, setup the |imds|
% read function, |imds.ReadFcn|, to pre-process images on-the-fly.
% The |imds.ReadFcn| is called every time an image is read from the
% |ImageDatastore|.
%
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% 2.3: Divide data into training and validation sets
[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomized');

countEachLabel(trainingSet)
countEachLabel(validationSet)

%% Part 3: Transfer Learning 

% The convolutional layers of the network extract image features that the 
% last learnable layer and the final classification layer use to classify 
% the input image. 

% To retrain a pretrained network to classify new images, we must replace these 
% last layers with new layers adapted to the new data set.

%% 3.1: Freeze all but last three layers

layersTransfer = model.Layers(1:end-3);
numClasses = 2; % cat and dog

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% 3.2: Configure training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% 3.3: Retrain network

modelTransfer = trainNetwork(trainingSet,layers,options);

%% 3.4: Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(modelTransfer,validationSet);

%% 3.5: Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation);
fprintf("The validation accuracy is: %.2f %%\n", accuracy * 100);

%% 3.6: Test it on unseen images
newImage1 = './dog.jpg'; % any dog image should do!
img1 = readAndPreprocessImage(newImage1);
YPred1 = predict(modelTransfer,img1);
[confidence1,idx1] = max(YPred1);
label1 = categories{idx1};
% Display test image and assigned label
figure
imshow(img1)
title(string(label1) + ", " + num2str(100*confidence1) + "%");

newImage2 = './cat.jpg'; % any cat image should do!
img2 = readAndPreprocessImage(newImage2);
YPred2 = predict(modelTransfer,img2);
[confidence2,idx2] = max(YPred2);
label2 = categories{idx2};
% Display test image and assigned label
figure
imshow(img2)
title(string(label2) + ", " + num2str(100*confidence2) + "%");
   
%% 3.7: Test it on unseen images: Your turn!

% What about the iconic "Doge"?
% ENTER YOUR CODE HERE
newImage3 = './doge.jpg'; % any dog image should do!
img3 = readAndPreprocessImage(newImage3);
YPred3 = predict(modelTransfer,img3);
[confidence3,idx3] = max(YPred3);
label3 = categories{idx3};
% Display test image and assigned label
figure
imshow(img3)
title(string(label3) + ", " + num2str(100*confidence3) + "%");

%% Part 4: Data augmentation

% Data augmentation helps prevent the network from overfitting and
% memorizing the exact details of the training images.

% In MATLAB, this can be done using the "Augmented Image Datastore"
% (https://www.mathworks.com/help/deeplearning/ref/augmentedimagedatastore.html)

% Despite its name, however, it DOES NOT increase the actual number of
% samples. When you use an augmented image datastore as a source of
% training images, the datastore randomly perturbs the training data for
% each epoch, so that each epoch uses a slightly different data set. The
% actual number of training images at each epoch does not change. The
% transformed images are not stored in memory.

%% 4.1: Defining the imageAugmenter object 
% In our case, we shall use an augmented image datastore to randomly flip
% the training images along the vertical axis and randomly translate them
% up to 30 pixels and scale them up to 10% horizontally and vertically.

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

%% 4.2: Building the augmented training and validation sets

inputSize = model.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);

disp(augimdsTrain.NumObservations) % You should see 28

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);

disp(augimdsValidation.NumObservations) % You should see 12

%% 4.3: Train the network with augmented datasets

miniBatchSize = 10;
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',3e-4, ...
    'ValidationData',augimdsValidation, ...
    'Verbose',false, ...
    'Plots','training-progress');

modelAug = trainNetwork(augimdsTrain,layers,options);

%% 4.4: Classify the validation images using the fine-tuned network.

[YPredAug,probsAug] = classify(modelAug,augimdsValidation); %changed from augimdsValidation to validationSet

%% 4.5: Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidationAug = validationSet.Labels;
accuracyAug = mean(YPredAug == YValidationAug);
fprintf("The validation accuracy is: %.2f %%\n", accuracyAug * 100);

%% Part 5: Larger datasets

% One possible reason why the accuracy of the classifier was so low 
% might have to do with not enough training data.
% Of course, there are plenty of dogs and cats around (in the Kaggle
% dataset and elsewhere) to circumvent this problem.

% In this part, you will use the Kaggle dataset and basically repeat the
% steps in Parts 2 and 3 (and 4, if you wish) using larger training and
% validation datasets.

% See Guidelines for instructions.
%% 5.1: Set up image data from full Kaggle training dataset

rng(0,'twister');

% Load full training dataset and build image store
dataFolderFull = './data/train';
catFilesFull = fullfile(dataFolderFull,'cat.*.jpg');
dogFilesFull = fullfile(dataFolderFull,'dog.*.jpg');
dataLabelsFull = [repmat(categorical({'cat'}),numel(dir(catFilesFull)),1);...
    repmat(categorical({'dog'}),numel(dir(dogFilesFull)),1)];
imdsFull = imageDatastore({catFilesFull dogFilesFull},'Labels', dataLabelsFull);
tblFull = countEachLabel(imdsFull); 
disp (tblFull)

% Use the smallest overlap set
% (useful when the two classes have different number of elements)
minSetCountFull = min(tblFull{:,2});

% Use splitEachLabel method to trim the set.
imdsFull = splitEachLabel(imdsFull, minSetCountFull, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imdsFull)

% Set the ImageDatastore ReadFcn
imdsFull.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Divide data into training and validation sets
[trainingSetFull, validationSetFull] = splitEachLabel(imdsFull, 0.8, 'randomized');

countEachLabel(trainingSetFull)
countEachLabel(validationSetFull)

%% 5.2: Transfer learning

fileNameTemp = fullfile('./modelTransferFull.mat');

% If trained model file does not exists, train model; otherwise load model
if ~exist(fileNameTemp,'file')
    
    % create checkpoint folder if not exist
    if ~exist(fullfile('./checkpoint'),'dir')
        mkdir(fullfile('./checkpoint'))
    end

    % Configure training options
    optionsFull = trainingOptions('sgdm', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',6, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',validationSetFull, ...
        'ValidationFrequency',100, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'CheckPointPath',fullfile('./checkpoint'));

    % Retrain network
    modelTransferFull = trainNetwork(trainingSetFull,layers,optionsFull);
    save(fileNameTemp,'modelTransferFull');
else
    load(fileNameTemp,'modelTransferFull');
end

%% 5.3: Classify the validation images using the fine-tuned network.

[YPredFull,scoresFull] = classify(modelTransferFull,validationSetFull);

% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.
YValidationFull = validationSetFull.Labels;
accuracyFull = mean(YPredFull == YValidationFull);
fprintf("The validation accuracy is: %.2f %%\n", accuracyFull * 100);

%% References
% [1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image
% database." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
% Conference on. IEEE, 2009.
%
% [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
% classification with deep convolutional neural networks." Advances in
% neural information processing systems. 2012.
%
% [3] Vedaldi, Andrea, and Karel Lenc. "MatConvNet-convolutional neural
% networks for MATLAB." arXiv preprint arXiv:1412.4564 (2014).
%
% [4] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding
% convolutional networks." Computer Vision-ECCV 2014. Springer
% International Publishing, 2014. 818-833.
%
% [5] Donahue, Jeff, et al. "Decaf: A deep convolutional activation feature
% for generic visual recognition." arXiv preprint arXiv:1310.1531 (2013).
