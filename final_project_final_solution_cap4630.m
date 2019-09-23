%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Summer 2019 
% Student Name: Tsz Shing Tsoi
%% Final Project - Final Solution (2-class classifier: cats vs. dogs)

%% Part 1: Data Setup
% In this part, the full Kaggle training dataset will be loaded and
% partitioned into training and hold-out validation sets.
% These training and validation datasets will be used for developing 
% baseline classifier, improved classifier 1, and improved classifier 2.

%% 1.1: Load image data from full Kaggle training dataset
clc;
clear;
close all;

% Load full training dataset and build image store
dataFolderFull = './data/train';
catFilesFull = fullfile(dataFolderFull,'cat.*.jpg');
dogFilesFull = fullfile(dataFolderFull,'dog.*.jpg');
dataLabelsFull = [repmat(categorical({'cat'}),numel(dir(catFilesFull)),1);...
    repmat(categorical({'dog'}),numel(dir(dogFilesFull)),1)];
imdsFull = imageDatastore({catFilesFull dogFilesFull},'Labels', dataLabelsFull);

%% 1.2: Display sample images

rng(0,'twister'); % seed random generator for consistency

% extract indices of cats and dogs
idxCat = find(imdsFull.Labels == 'cat');
idxDog = find(imdsFull.Labels == 'dog');

% randomly select up to 6 images
selectIdxCat= randperm(size(idxCat,1),min(size(idxCat,1),6));
selectIdxDog= randperm(size(idxDog,1),min(size(idxCat,1),6));

% diaplay montage of images
figure, montage(subset(imdsFull, idxCat(selectIdxCat)));
title('Sample Images of Cats');
figure, montage(subset(imdsFull, idxDog(selectIdxDog)));
title('Sample Images of Dogs');

%% 1.3: Set up image data store

rng(0,'twister'); % seed random generator for consistency

tblFull = countEachLabel(imdsFull); 
disp (tblFull)

% Use the smallest overlap set
% (useful when the two classes have different number of elements)
minSetCountFull = min(tblFull{:,2});

% Use splitEachLabel method to trim the set.
imdsFull = splitEachLabel(imdsFull, minSetCountFull, 'randomize');

% Set the ImageDatastore ReadFcn
imdsFull.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Divide data into training and validation sets
[trainingSetFull, validationSetFull] = splitEachLabel(imdsFull, 0.8, 'randomized');

countEachLabel(trainingSetFull)
countEachLabel(validationSetFull)

%% 1.4: Set up layers for the modified CNN

model = alexnet;

% Freeze all but last three layers
layersTransferFull = model.Layers(1:end-3);
numClasses = 2; % cat and dog

layersFull = [
    layersTransferFull
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% create checkpoint folder if not exist
if ~exist(fullfile('./checkpoint'),'dir')
    mkdir(fullfile('./checkpoint'))
end

%% Part 2: Baseline Classifier
%% 2.1: Transfer Learning for Baseline Classifier

fileNameTemp = fullfile('./modelTransferFull.mat');

% If trained model file does not exists, train model; otherwise load model
if ~exist(fileNameTemp,'file')

    % Configure training options
    % ValidationFrequency does not affect the final results and value could be
    % increased to decrease the training time.
    optionsFull = trainingOptions('sgdm', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',6, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',validationSetFull, ...
        'ValidationFrequency',100, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'CheckPointPath',fullfile('./checkpoint'), ...
        'OutputFcn', @(info)saveTrainingPlot(info, 'modelTransferFullTrainingPlot.png'));
    
    % close all figures if opened
    close(findall(groot, 'Type', 'Figure'));
    % Retrain network
    modelTransferFull = trainNetwork(trainingSetFull,layersFull,optionsFull);
    save(fileNameTemp,'modelTransferFull');
else
    load(fileNameTemp,'modelTransferFull');
    figure, imshow(imread(fullfile('./modelTransferFullTrainingPlot.png')));
    title('Training Plot for Baseline Classifier');
end

%% 2.2: Classify the validation images using the fine-tuned network.

fileNameTemp = fullfile('./YPredFull.mat');
if ~exist(fileNameTemp,'file')
    [YPredFull,scoresFull] = classify(modelTransferFull,validationSetFull);
    save(fileNameTemp,'YPredFull');
else
    load(fileNameTemp,'YPredFull');
end

% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.
YValidationFull = validationSetFull.Labels;
accuracyFull = mean(YPredFull == YValidationFull);
fprintf("The validation accuracy for Baseline Classifier is: %.2f %%\n", accuracyFull * 100);

%% Part 3: Improved Classifier 1
% Improved Classifier 1 uses data augmentation.
% Data augmentation helps prevent the network from overfitting and
% memorizing the exact details of the training images.

%% 3.1: Defining the imageAugmenter object 
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

%% 3.2: Building the augmented training and validation sets

inputSize = model.Layers(1).InputSize;
augimdsTrainFull = augmentedImageDatastore(inputSize(1:2),trainingSetFull, ...
    'DataAugmentation',imageAugmenter);
augimdsValidationFull = augmentedImageDatastore(inputSize(1:2),validationSetFull);

%% 3.3: Train the network with augmented datasets

fileNameTemp = fullfile('./modelAugFull.mat');

% If trained model file does not exists, train model; otherwise load model
if ~exist(fileNameTemp,'file')
    
    % Configure training options
    % Same as the baseline classifier
    optionsAugFull = trainingOptions('sgdm', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',6, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidationFull, ...
        'ValidationFrequency',100, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'CheckPointPath',fullfile('./checkpoint'),...
        'OutputFcn', @(info)saveTrainingPlot(info, 'modelAugFullTrainingPlot.png'));

    % close all figures if opened
    close(findall(groot, 'Type', 'Figure'));
    % Retrain network
    modelAugFull = trainNetwork(augimdsTrainFull,layersFull,optionsAugFull);
    save(fileNameTemp,'modelAugFull');
else
    load(fileNameTemp,'modelAugFull');
    figure, imshow(imread(fullfile('./modelAugFullTrainingPlot.png')));
    title('Training Plot for Improved Classifier 1');
end

%% 3.4: Classify the validation images using the fine-tuned network.

fileNameTemp = fullfile('./YPredAugFull.mat');
if ~exist(fileNameTemp,'file')
    [YPredAugFull,probsAugFull] = classify(modelAugFull,augimdsValidationFull);
    save(fileNameTemp,'YPredAugFull');
else
    load(fileNameTemp,'YPredAugFull');
end

% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.
YValidationAugFull = validationSetFull.Labels;
accuracyAugFull = mean(YPredAugFull == YValidationAugFull);
fprintf("The validation accuracy for Improved Classifier 1 is: %.2f %%\n", accuracyAugFull * 100);

%% Part 4: Improved Classifier 2
% This part attempts to identify the optimal set of hyperparameters,
% including learning rate, mini-batch size and number of epochs using
% heuristics. The heuristics used is a reduced dataset, i.e. 5% of the full
% training dataset. Different learning rates and mini-batch sizes are tested
% to identify the optimal values. Variable learning rates may be used to
% speed up training if deemed appropriate. The number of epoches is then 
% determined based on the optimal set of learning rate and mini-batch size.
% The Improved Classifier 2 are then developed based on the optimal set of
% hyperparameters.

%% 4.1: Split training and validation sets as heuristics

%seed random generator for consistency
rng(0,'twister'); 
trainingSetHeu = splitEachLabel(trainingSetFull, 0.05, 'randomized');
validationSetHeu = splitEachLabel(validationSetFull, 0.05, 'randomized');

%% 4.2: Set the set of hyperparaters to be tested

% Test 3 different learning rates.
% The first element is control.
testLearnRate = [1e-4, ...
    1e-3, 5e-4, 1e-5, ...
    1e-4, 1e-4, 1e-4]';

% Test 3 different mini-batch sizes
testMiniBatchSize = [10, ...
    10, 10, 10,  ...
    100, 50, 5]';

% Placeholder for final validation accuracy
testValidationAcc = zeros(size(testLearnRate,1), 1);

testTable = table(testLearnRate, testMiniBatchSize, testValidationAcc, ...
    'VariableNames', {'LearningRate' 'MiniBatchSize' 'ValidationAccuracy'});

%% 4.3: Train model with different hyperparameters on heuristics dataset

fileNameTemp = fullfile('./modelTestHeuInfoSummary.mat');

% If the result matrix file does not exists, conduct test; otherwise load
% the result matrix file
if ~exist(fileNameTemp,'file')
    modelTestHeuInfoSummary = struct;
    for i = 1:size(testTable, 1)

         %seed random generator for consistency
         rng(0,'twister');

        % Configure training options
        % MaxEpochs is increased to account for decreased learning rate in some
        % scenarios
        optionsHeu = trainingOptions('sgdm', ...
            'MiniBatchSize',testTable.MiniBatchSize(i), ...
            'MaxEpochs',10, ...
            'InitialLearnRate',testTable.LearningRate(i), ...
            'Shuffle','every-epoch', ...
            'ValidationData',validationSetHeu, ...
            'ValidationFrequency',10, ...
            'Verbose',false);

        % train network
        fprintf("Training scenario %d begins...\n", i);
        [modelTestHeu, modelTestHeuInfo] = trainNetwork(trainingSetHeu,layersFull,optionsHeu);
        fprintf("Training scenario %d ended.\n", i);
        
        % save network info
        testTable.ValidationAccuracy(i) =  modelTestHeuInfo.ValidationAccuracy(end);
        modelTestHeuInfoSummary(i).modelInfo = modelTestHeuInfo;
        save(fileNameTemp, 'modelTestHeuInfoSummary');
    end
else
    load(fileNameTemp,'modelTestHeuInfoSummary');
    for i = 1:min(size(testTable, 1),size(modelTestHeuInfoSummary, 2))
        testTable.ValidationAccuracy(i) = modelTestHeuInfoSummary(i).modelInfo.ValidationAccuracy(end);
    end    
end

%% 4.4: Display final validation accuracy and plot training accuracy for each scenario

testTable

figure, hold on
for i =1:size(modelTestHeuInfoSummary, 2)
    plot (rmmissing(modelTestHeuInfoSummary(i).modelInfo.ValidationAccuracy))
end
legend(strcat('Learning Rate: ',num2str(testTable.LearningRate), ...
    '; MiniBatchSize: ',int2str(testTable.MiniBatchSize)), ...
    'location','east');
title('Validation Accuracies over Training Iterations on Heuristics Dataset');
xlabel('Number of Training Iterations (10s)');
ylabel('Validation Accuracy (%)');
hold off

%% 4.5: Train on heuristics dataset with selected preliminary hyperparameters

fileNameTemp = fullfile('./modelSelectedHeuInfo.mat');

% If the result struct file does not exists, conduct test; otherwise load
% the result struct file
if ~exist(fileNameTemp,'file')
    
    %seed random generator for consistency
    rng(0,'twister');

    % Configure selected preliminary training options
    selectedMiniBS = 100;
    selectedMaxEpochs = 50;
    selectedInitialLR = 1e-4;
    selectedLRSchedule = 'piecewise';
    selectedLRDropPeriod = 10;
    selectedLRDropFactor = 0.5;

    optionsHeuSelected = trainingOptions('sgdm', ...
        'MiniBatchSize',selectedMiniBS, ...
        'MaxEpochs',selectedMaxEpochs, ...
        'InitialLearnRate',selectedInitialLR, ...
        'LearnRateSchedule', selectedLRSchedule, ...
        'LearnRateDropPeriod', selectedLRDropPeriod, ...
        'LearnRateDropFactor', selectedLRDropFactor, ...
        'Shuffle','every-epoch', ...
        'ValidationData',validationSetHeu, ...
        'ValidationFrequency',10, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'OutputFcn', @(info)saveTrainingPlot(info, 'modelSelectedHeuTrainingPlot.png'));

    % close all figures if opened
    close(findall(groot, 'Type', 'Figure'));
    % train network
    [modelSelectedHeu, modelSelectedHeuInfo] = trainNetwork(trainingSetHeu,layersFull,optionsHeuSelected);

    % save network info
    fprintf("The validation accuracy with heuristics dataset is: %.2f %%\n", modelSelectedHeuInfo.ValidationAccuracy(end));
    save(fileNameTemp, 'modelSelectedHeuInfo');

else
    load(fileNameTemp,'modelSelectedHeuInfo');
    figure, imshow(imread(fullfile('./modelSelectedHeuTrainingPlot.png')));
    title('Training Plot for Selected Hyperparameters (Preliminary) on Heuristic Dataset');
    fprintf("The validation accuracy with heuristics dataset is: %.2f %%\n", modelSelectedHeuInfo.ValidationAccuracy(end));
end

%% 4.6: Train on full dataset with final selected hyperparameters

fileNameTemp = fullfile('./modelIC2Full.mat');

% If trained model file does not exists, train model; otherwise load model
if ~exist(fileNameTemp,'file')

    % Configure selected final training options
    selectedMiniBS = 80; %MiniBatchSize of 80 is used due to memory limitation of GPU on hand
    selectedMaxEpochs = 10;
    selectedInitialLR = 1e-4;
    
    optionsIC2Full = trainingOptions('sgdm', ...
        'MiniBatchSize',selectedMiniBS, ...
        'MaxEpochs',selectedMaxEpochs, ...
        'InitialLearnRate',selectedInitialLR, ...
        'Shuffle','every-epoch', ...
        'ValidationData',validationSetFull, ...
        'ValidationFrequency',100, ...
        'Verbose',false, ...
        'Plots','training-progress', ...
        'CheckPointPath',fullfile('./checkpoint'), ...
        'OutputFcn', @(info)saveTrainingPlot(info, 'modelIC2FullTrainingPlot.png'));
    
    % close all figures if opened
    close(findall(groot, 'Type', 'Figure'));
    % train network
    modelIC2Full = trainNetwork(trainingSetFull,layersFull,optionsIC2Full);

    % save network info
    save(fileNameTemp, 'modelIC2Full');

else
    load(fileNameTemp,'modelIC2Full');
    figure, imshow(imread(fullfile('./modelIC2FullTrainingPlot.png')));
    title('Training Plot for Improved Classifier 2');
end

%% 4.7: Classify the validation images using the fine-tuned network.

fileNameTemp = fullfile('./YPredIC2Full.mat');
if ~exist(fileNameTemp,'file')
    [YPredIC2Full,probsIC2Full] = classify(modelIC2Full,validationSetFull);
    save(fileNameTemp,'YPredIC2Full');
else
    load(fileNameTemp,'YPredIC2Full');
end

% Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.
YValidationIC2Full = validationSetFull.Labels;
accuracyIC2Full = mean(YPredIC2Full == YValidationIC2Full);
fprintf("The validation accuracy for Improved Classifier 2 is: %.2f %%\n", accuracyIC2Full * 100);

%% Part 5: Performance Evaluation
% This part evaluates each of the three classifiers: baseline classifiers,
% improved classifier 1 and improved classifier 2

%% 5.1: Load test data

% Load full test dataset and build image store
dataFolderTestFull = './data/test';
imdsTestFull = imageDatastore(fullfile(dataFolderTestFull));
% Set the ImageDatastore ReadFcn
imdsTestFull.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% 5.2: Classify test data

fileNameTemp = fullfile('./ClassifierResult.mat');

% If classifier results do not exists, classify; otherwise load classifier
% results
if ~exist(fileNameTemp,'file')
    ClassifierResult = struct;
    % Baseline Classifier: classify the test images using the fine-tuned network.
    ClassifierResult(1).name = 'Baseline Classifier';
    [ClassifierResult(1).YPred, ClassifierResult(1).probs] = classify(modelTransferFull,imdsTestFull);
    
    % Improved Classifier 1: classify the test images using the fine-tuned network.
    ClassifierResult(2).name = 'Improved Classifier 1';
    [ClassifierResult(2).YPred, ClassifierResult(2).probs] = classify(modelAugFull,imdsTestFull);

    % Improved Classifier 2: classify the test images using the fine-tuned network.
     ClassifierResult(3).name = 'Improved Classifier 2';
     [ClassifierResult(3).YPred, ClassifierResult(3).probs] = classify(modelIC2Full,imdsTestFull);

    % Ensemble of Classifiers
    ClassifierResult(4).name = 'Ensemble of Classifiers';
    for i = 1:2
        ClassifierResult(4).probs(:,i) = mean([ClassifierResult(1).probs(:,i), ...
            ClassifierResult(2).probs(:,i), ...
            ClassifierResult(3).probs(:,i)], 2);
    end
    ClassifierResult(4).YPred = categorical([ClassifierResult(4).probs(:,1) < 0.5],[0 1],{'cat' 'dog'});
    
    save(fileNameTemp,'ClassifierResult');
else
    load(fileNameTemp,'ClassifierResult');
end

%% 5.3: Save classified images into folder for manual checking and load manually classified labels

fileNameTemp = fullfile('./imageLabelTestSetGroundTruth.csv');

% Manually prepare the ground truth label file named
% "imageLabelTestSetGroundTruth.csv" with the first column as the image file name
% and the second column as the label. The order of the image file name
% should be preserved.

% If ground truth label does not exists, create prediction label for manual classification;
% otherwise load ground truth label
if ~exist(fileNameTemp,'file')

    % create cat folder if not exist
    if ~exist(fullfile('./data/test_cat'),'dir')
        mkdir(fullfile('./data/test_cat'))
    end
    % create dog folder if not exist
    if ~exist(fullfile('./data/test_dog'),'dir')
        mkdir(fullfile('./data/test_dog'))
    end
    for i = 1:size(imdsTestFull.Files,1)
        destfile = fullfile('./data',strcat('test_',char(ClassifierResult(4).YPred(i))));
        copyfile (char(imdsTestFull.Files(i)), destfile);
    end

    % export image labels for manual checking
    imageLabel = cell(size(imdsTestFull.Files,1),2);
    for i = 1: size(imdsTestFull.Files,1)
        [~,imageName,imageExt] = fileparts(char(imdsTestFull.Files(i)));
        imageLabel(i,1) = {strcat(imageName,imageExt)};
        imageLabel(i,2) = {char(ClassifierResult(4).YPred(i))};
    end
    writecell(imageLabel,fullfile('./imageLabelTestSetPred.csv'));
else
    imageLabel = cell(size(imdsTestFull.Files,1),1);
    for i = 1: size(imdsTestFull.Files,1)
        [~,imageName,imageExt] = fileparts(char(imdsTestFull.Files(i)));
        imageLabel(i,1) = {strcat(imageName,imageExt)};
    end
    
    imageLabelTestSetGroundTruth = readcell(fileNameTemp);
    % check if the image file names are in the correct order
    if isequal(imageLabel(:,1),imageLabelTestSetGroundTruth(:,1))
        % read label into ImageDateStore
        imdsTestFull.Labels = categorical(imageLabelTestSetGroundTruth(:,2));
    % if not, check if the label file names match the imageDatestore file names
    elseif size(intersect(imageLabel(:,1),imageLabelTestSetGroundTruth(:,1)),1) == size(imageLabel(:,1),1)
        tempLabels = cell(size(imageLabel(:,1)));
        for i = 1:size(imageLabel(:,1),1)
            for j = 1:size(imageLabelTestSetGroundTruth(:,1),1)
                if isequal(imageLabel(i,1),imageLabelTestSetGroundTruth(j,1))
                    tempLabels(i,1)= imageLabelTestSetGroundTruth(j,2);
                    imageLabelTestSetGroundTruth(j,:)=[];
                    break;
                end
            end
        end
        imdsTestFull.Labels = categorical(tempLabels);
    else % if label does not match imageDatestore file names
        fprintf("Ground truth labels do not match imageDatastore files!!!");
    end
end

%% 5.4: Calculate the classification accuracy on the test set and display sample images

YGroundTruthLabelTestFull = imdsTestFull.Labels;

% extract indices of ground truth cats and dogs
idxGroundTruthCat = find(YGroundTruthLabelTestFull == 'cat');
idxGroundTruthDog = find(YGroundTruthLabelTestFull == 'dog');

%reset imageDataStore read function
imdsTestFull.ReadFcn = @(filename)imread(filename);

for i = 1:size(ClassifierResult,2)
    ClassifierResult(i).accuracy = mean(ClassifierResult(i).YPred == YGroundTruthLabelTestFull);
    fprintf("The classification accuracy for %s is: %.2f %%\n", ClassifierResult(i).name, ...
        ClassifierResult(i).accuracy * 100);
    ClassifierResult(i).confMat = confusionmat(YGroundTruthLabelTestFull,ClassifierResult(i).YPred);
    % Display confusion chart
    figure, confusionchart(ClassifierResult(i).confMat,{'cat','dog'});
    title(strcat("Confusion Matrix on Test Dataset,  ", ClassifierResult(i).name));
    
    % Compute the ROC curve, positive class assumes to be dog
    [X,Y,T,AUC] = perfcurve(YGroundTruthLabelTestFull, ClassifierResult(i).probs(:,2), 'dog'); 

    % Plot the ROC curve
    figure, plot(X,Y)
    xlabel('False positive (dog) rate'); ylabel('True positive (dog) rate');
    title(strcat("ROC Curves on Test Dataset,  ", ClassifierResult(i).name));

    % Display the area under the curve.
    fprintf("Area under curve for %s is: %.4f \n", ClassifierResult(i).name, AUC)
    
    % Display montage of correctly identified images
    % sorted indices by probability
    [~,idxSortByCat] = sort(ClassifierResult(i).probs(:,1));
    [~,idxSortByDog] = sort(ClassifierResult(i).probs(:,2));
    
    % indices of correct and incorrect classifications
    idxCorrectPred = find (ClassifierResult(i).YPred == YGroundTruthLabelTestFull);
    idxIncorrectPred = find (~(ClassifierResult(i).YPred == YGroundTruthLabelTestFull));
    
    % indices of correctly classified cats and dogs
    idxCorrectCatPred = intersect(idxGroundTruthCat,idxCorrectPred);
    idxCorrectDogPred = intersect(idxGroundTruthDog,idxCorrectPred);

    % indices of cats incorrectly classified as dogs
    idxIncorrectCatPred = intersect(idxGroundTruthCat,idxIncorrectPred);
    
    % indices of dogs incorrectly classified as cats
    idxIncorrectDogPred = intersect(idxGroundTruthDog,idxIncorrectPred);
    
    % display sample images correctly classified as cats with the highest confidence
    subIdxCorrectCat = intersect(idxSortByCat,idxCorrectCatPred, 'stable');
    if ~isempty(subIdxCorrectCat)
        subIdxCorrectCat = subIdxCorrectCat (max(size(subIdxCorrectCat,1) - 5,1): end);
        figure, montage(subset(imdsTestFull, subIdxCorrectCat));
        % construct confidence % of the images 
        lowerRange = ClassifierResult(i).probs(subIdxCorrectCat(1), 1);
        upperRange = ClassifierResult(i).probs(subIdxCorrectCat(end), 1);
        if lowerRange == upperRange
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "%");
        else
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "% to ", ...
                num2str(upperRange * 100, '%.2f'), "%");
        end
        title(strcat("Sample - Correctly Classified as Cats, ", ClassifierResult(i).name, ...
            " | Confidence: ", confidenceStr));
    end
    
    % display sample images correctly classified as dogs with the highest confidence
    subIdxCorrectDog = intersect(idxSortByDog,idxCorrectDogPred, 'stable');
    if ~isempty(subIdxCorrectDog)
        subIdxCorrectDog = subIdxCorrectDog (max(size(subIdxCorrectDog,1) - 5,1): end);
        figure, montage(subset(imdsTestFull, subIdxCorrectDog));
        % construct confidence % of the images 
        lowerRange = ClassifierResult(i).probs(subIdxCorrectDog(1), 2);
        upperRange = ClassifierResult(i).probs(subIdxCorrectDog(end), 2);
        if lowerRange == upperRange
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "%");
        else
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "% to ", ...
                num2str(upperRange * 100, '%.2f'), "%");
        end
        title(strcat("Sample - Correctly Classified as Dogs, ", ClassifierResult(i).name, ...
            " | Confidence: ", confidenceStr));
    end
    
    % display sample images incorrectly classified as dogs with the lowest confidence
    subIdxIncorrectCat = intersect(idxSortByDog,idxIncorrectCatPred, 'stable');
    if ~isempty(subIdxIncorrectCat)
        subIdxIncorrectCat = subIdxIncorrectCat (1: min(size(subIdxIncorrectCat,1), 6));
        figure, montage(subset(imdsTestFull, subIdxIncorrectCat));
        % construct confidence % of the images 
        lowerRange = ClassifierResult(i).probs(subIdxIncorrectCat(1), 2);
        upperRange = ClassifierResult(i).probs(subIdxIncorrectCat(end), 2);
        if lowerRange == upperRange
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "%");
        else
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "% to ", ...
                num2str(upperRange * 100, '%.2f'), "%");
        end
        title(strcat("Sample - Incorrectly Classified as Dogs (Should be Cats), ", ClassifierResult(i).name, ...
            " | Confidence: ", confidenceStr));
    end
    
    % display sample images incorrectly classified as cats with the lowest confidence
    subIdxIncorrectDog = intersect(idxSortByCat,idxIncorrectDogPred, 'stable');
    if ~isempty(subIdxIncorrectDog)
        subIdxIncorrectDog = subIdxIncorrectDog (1: min(size(subIdxIncorrectDog,1), 6));
        figure, montage(subset(imdsTestFull, subIdxIncorrectDog));
        % construct confidence % of the images 
        lowerRange = ClassifierResult(i).probs(subIdxIncorrectDog(1), 1);
        upperRange = ClassifierResult(i).probs(subIdxIncorrectDog(end), 1);
        if lowerRange == upperRange
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "%");
        else
            confidenceStr = strcat(num2str(lowerRange * 100, '%.2f'), "% to ", ...
                num2str(upperRange * 100, '%.2f'), "%");
        end
        title(strcat("Sample - Incorrectly Classified as Cats (Should be Dogs), ", ClassifierResult(i).name, ...
            " | Confidence: ", confidenceStr));
    end
end

%% Auxiliary Function
% This function saves the training plot of a CNN at the end of the
% training. Please close all figures before training to avoid errors.
function stop = saveTrainingPlot(info, filename)
stop = false;  %prevents this function from ending trainNetwork prematurely
if info.State == 'done'   %check if all iterations have completed
% if true
    tempFigHandle = findall(groot, 'Type', 'Figure');
    set(tempFigHandle(1), 'Position', get(0, 'Screensize'));
    saveas(tempFigHandle(1),fullfile('./', filename));
end
end
