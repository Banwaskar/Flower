imds = imageDatastore('FLR_IMGS', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
%% 
% view

numImagesTrain = numel(imdsTrain.Labels);
idx = randperm(numImagesTrain,16);

% for i = 1:16
%     I{i} = readimage(imdsTrain,idx(i));
% end

% figure, imshow(imtile(I));
%% 
% load network and siplay Layer

net = alexnet;
net.Layers
%% 
% layer image size

inputSize = net.Layers(1).InputSize
%% 
% extracted image feature

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows')
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows')
%% 
% Extract the class labels from the training and test data.

YTrain = imdsTrain.Labels
YTest = imdsTest.Labels
%% 
% *Fit Image Classifier*
% 
% Use the features extracted from the training images as predictor variables 
% and fit a multiclass support vector machine (SVM) using |fitcecoc| (Statistics 
% and Machine Learning Toolbox).

mdl = fitcecoc(featuresTrain,YTrain)
%% 
% *Classify Test Images*

YPred = predict(mdl,featuresTest)
%% 
% Display four sample test images with their predicted labels.

idx = randi([1 900],4,4,1);
figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(string(label));
end
%% 
% Calculate the classification accuracy on the test set.

accuracy = mean(YPred == YTest)