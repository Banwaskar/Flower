function [ TrainingSet] = FeatureExtractorFunctionNew(imdsTrain)
%% 
% Feature Extraction form GLCM Texture Method and storing all the result 
% in feature variable

numImagesTrain = numel(imdsTrain.Labels);

feature=nan(numImagesTrain,4);
for i = 1:numImagesTrain   
RGB=readimage(imdsTrain,i);
I=rgb2gray(RGB);
% I=histeq(I);
%imshow(I );
% offsets0 = [zeros(40,1) (1:40)'];
% [glcms,SI] = graycomatrix(I,'Offset',offsets);
[glcms,SI] = graycomatrix(I);
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
feature(i,:)=[stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
end

%% 
% Add R , G, B average value and standard deviation  too as a feature for 
% true color of flower

Colorfeature=nan(numImagesTrain,6);
for i = 1:numImagesTrain  
    RGB=readimage(imdsTrain,i);
    r=RGB(:,:,1);
    g=RGB(:,:,2);
    b=RGB(:,:,3);
    mr= mean(r(r>10))/255;
    stdr=std(double(r(r>10)));
    mg= mean(g(g>10))/255;
     stdg=std(double(g(g>10)));
    mb= mean(b(b>10))/255;
     stdb=std(double(b(b>10)));
    Colorfeature(i,:)=[mr stdr  mg  stdg mb stdb];
end
%% 
% Output response 



TrainingSet = array2table([feature,Colorfeature],...
    'VariableNames',{'Contrast','Correlation','Energy','Homogenity','Red','stdr','green','stdg','blue','stdb'});


end