clc;
clear all;

%% 
% Input Image Data
% path ='D:\Files\Projects\HISKY\SpeckleDenoising\MyContribution\HiskyData\';

% % path = 'D:\Files\Projects\HISKY\SpeckleDenoising\MyContribution\BayesianNLM\'
% % 
% % file = dir(fullfile(path,'*.png')); % (*.dat)
% % 
% % fileNames = {file.name}';
% % 
% % numFiles = size(fileNames,1);
% % 
% % for i = 1:1
% %     singleImgName = strcat(path, fileNames(i));
% % %     fid = fopen(singleImgName{1});
% % %     tline = fread(fid, [512, 512], 'int16');
% % %     R = tline';
% % %     R(R<0) = 0; 
% % %     img = R;
% % %     figure
% % %     imshow(img,[0, 150])
% %     img = imread(singleImgName{1});
% %     figure
% %     imshow(img)
% % end

img = imread('D:\Files\Projects\HISKY\SpeckleDenoising\MyContribution\BayesianNLM\noisyImage.png');
img = ImgNormalize(img);

timeSize = 0.05; % time step size for interation
iterationMaxStep = 200; % one of the iteration determination conditions
decayFactor = 1;

despeckledImg = SpeckleReducingAD(img, iterationMaxStep, timeSize, decayFactor);

figure
subplot 131
imshow(img)
title('Origin Image')

subplot 132
imshow(despeckledImg)
title('Despecked Image')

subplot 133
delta = ~logical(img - despeckledImg);
imshow(double(delta))
title('Subtraction Image')
