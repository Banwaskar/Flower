clc;clear;

load FlowerNet
imdsPrediction=imageDatastore('E:\flower Project\P1')
oldFolder=string('E:\flower Project\P1');
for i=1:length(imdsPrediction.Files)
    a=char(imdsPrediction.Files(i));
    a=imread(a);
   
  [~,RGBB] = FlowerBackgroundFilter(a);
  [~,RGBW] = WhiteColorAdder(a);
  [~,RGBP] = PurpleFlower(a);
  [~,RGBO] = OrangeFlower(a);
  RGB=max(RGBB,RGBW);                  %(RGBB,RGBW,RGBP,RGBO);
  RGB=max(RGB,RGBP);
  RGB=max(RGB,RGBO);
    figure;
    imshow(RGB)
    fileName= string(char(imdsPrediction.Files(i)));
    newFileName = strrep(fileName,oldFolder,'E:\flower Project\P_P1');
    imwrite(RGB,char(newFileName));
end

imdsPrediction=imageDatastore('E:\flower Project\P_P1');
augimdsPrediction = augmentedImageDatastore([227 227 3],imdsPrediction);
y=classify(FlowerNet,augimdsPrediction)
