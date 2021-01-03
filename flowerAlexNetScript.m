%% Load the dataset ,resize it, divide it into training and validation set

FlowerData =imageDatastore("FLR_IMGS","IncludeSubfolders",true,"LabelSource","foldernames");

% size= 227;   % resize dimension for alexnet training 227 x 277 x 3
% for i=1:length(FlowerData.Files)
%     
%     a =imread(char(FlowerData.Files(i)));
%     a=imresize(a,[size,size]);
%     imwrite(a,char(FlowerData.Files(i)));
% end

[imdsTrain, imdsValidation] = splitEachLabel(FlowerData,0.8); % 
% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([227 227 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227 3],imdsValidation);
%% Set Training Options
% Specify options to use when training.

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MiniBatchSize",20,...
    "MaxEpochs",20,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);
%% Create Transfer Learning Layer model FlowerNet using Alexnet

FlowerNet = alexnet; % this will copy the layer architecture of alexnet into 
                     % FlowerNet Varibale
                     
layers = FlowerNet.Layers;
layers(end-2)=fullyConnectedLayer(30,'Name','Fully Connected '); % 30 types of flower
layers(end)=classificationLayer;
layers
%% 
% 
%% Train Network
% Train the network using the specified options and training data.

[FlowerNet, traininfo] = trainNetwork(augimdsTrain,layers,opts);
save FlowerNet