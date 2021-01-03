%Selection of Data from Database , 70% for Training and 30% for testing

clear;clc;
% imds = imageDatastore('E:\flower Project\P_Processed\', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames')

 imds = imageDatastore('E:\flower Project\P_P1\', ... 'IncludeSubfolders',true, ...
  'LabelSource','foldernames')

numImages = numel(imds.Labels);


for i = 1:numImages
    I{i} = readimage(imds,i);
end

%figure
%imshow(imtile(I))
%% 
% Feature Extraction using GLCM and Red, Green , Blue average density

 [ TestSet] = FeatureExtractorFunctionNew(imds)
% [ TestSet] = FeatureExtractorFunction(imdsTest);
 load trainedclassifier
 YPred = trainedClassifier.predictFcn(TestSet)
load FlowerNet
augimds = augmentedImageDatastore([227 227 3],imds);
YPred = classify(FlowerNet,augimds)
%% 
% 
%%


queryno = 2;
BucketSize=10;
foldername = strcat('E:\flower Project\FLR_IMGS\',char(YPred(queryno)))
imdsDatabase = imageDatastore(foldername)
[ TrainingSet] = FeatureExtractorFunctionNew(imdsDatabase)
[ TestSet] = FeatureExtractorFunctionNew(imds);
%% 
% *KD Tree Algorithm code Implementation*
% 
% *( |*'chebychev','cityblock','euclidean','minkowski')|

TrainData=table2array(TrainingSet);
TestData=table2array(TestSet);
%% 
% |'minkowski' - method|

Mdl = KDTreeSearcher(TrainData,'Distance','minkowski','BucketSize',BucketSize)
IdxNN = knnsearch(Mdl,TestData(queryno,:),'K',8)

figure
for i = 1:numel(IdxNN)
    subplot(3,3,i+1)
    I = readimage(imdsDatabase,IdxNN(i));    
    imshow(I)
    title(strcat('output',num2str(i)))
end
subplot(3,3,1)
    I = readimage(imds,queryno);    
    imshow(I)
    title('QueryImage')
    
%% 
% |'euclidean' - method|

Md2 = KDTreeSearcher(TrainData,'Distance','euclidean','BucketSize',BucketSize)
IdxNN = knnsearch(Md2,TestData(queryno,:),'K',8)

figure
for i = 1:numel(IdxNN)
    subplot(3,3,i+1)
    I = readimage(imdsDatabase,IdxNN(i));    
    imshow(I)
    title(strcat('output',num2str(i)))
end
subplot(3,3,1)
    I = readimage(imds,queryno);    
    imshow(I)
    title('QueryImage')
    
    
%% 
% |'cityblock' - method|

Md3 = KDTreeSearcher(TrainData,'Distance','cityblock','BucketSize',BucketSize)
IdxNN = knnsearch(Md3,TestData(queryno,:),'K',8)

figure
for i = 1:numel(IdxNN)
    subplot(3,3,i+1)
    I = readimage(imdsDatabase,IdxNN(i));    
    imshow(I)
    title(strcat('output',num2str(i)))
end
subplot(3,3,1)
    I = readimage(imds,queryno);    
    imshow(I)
    title('QueryImage')
    
    
%% 
% |'chebychev' - method|

Md4 = KDTreeSearcher(TrainData,'Distance','chebychev','BucketSize',BucketSize)
IdxNN = knnsearch(Md4,TestData(queryno,:),'K',8)

figure
for i = 1:numel(IdxNN)
    subplot(3,3,i+1)
    I = readimage(imdsDatabase,IdxNN(i));    
    imshow(I)
    title(strcat('output',num2str(i)))
end
subplot(3,3,1)
    I = readimage(imds,queryno);    
    imshow(I)
    title('QueryImage')
