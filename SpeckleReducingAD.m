

function despeckledImg = SpeckleReducingAD(img, iterationMaxStep, timeSize, decayFactor)

% function despeckledImg = MySRAD(img, iterationMaxStep, threshodConvergence, timeSize, decayFactor)
% Algorithm: Speckle Reducing Anisotropic Diffusion Method
% Input:
%   - img: the input image with speckle noise; data type [any kinds of], size N*M  
%   - interationMaxStep: the maximum iterative step; data type [integer(>0)]
%   - thresholdCovergence: the second ending rule for iteration; data type
%   [undine]
%   - timeSize: the time step in each iteration; data type [float or
%   integer], scalar
%   - dacayFactor: the decay factor for the exp function related to
%   "smoothing and edge preserving"; data type [float, integer]. scalar

% Output:
%   - despeckledImg: the despeckled image; data type [same as the img], size
%   N*M

% Author: Shuwei Xing
% Date: 2019-09-25

% Reference: 
%  - Yu, Yongjian, and Scott T. Acton. "Speckle reducing anisotropic diffusion." IEEE Transactions on image processing 11.11 (2002): 1260-1270.
%  - Matlab function imdiffusefilt

%
% INPUT PARAMETERS
%
    spatialSize = 1; %spatial step size
    t = 0; % starting timestap
    thresholdDiffusion = 0.0001; % threshold for diffusion coefficient to guarantee the lower vaule (equals 0)
    thresholdConvergence = 10^(-13); % one of the iteration determination conditions
    [ROW, COL] = size(img); % input image size
    epsilon = 10^(-13); % handle 0/0 case 

    iterationNumber = 0; % variable of iterating step
    diviationConvergence = 1; % the convergence diviation each iteration
   
    originalClass = class(img);

    if ~isa(img,'double')
        img = single(img);
    end

    img_i_j_ = img;
    temp = img_i_j_(1:spatialSize:end, :);
    img_i_j =temp(:,1:spatialSize:end);
    
    % translate one row (add + spatialSize)
%     img_ia1_j_ = [img; img((end-spatialSize+1) :end,:)];
%     img_ia1_j_(1:spatialSize,:) = [];
%     temp = img_ia1_j_(1:spatialSize:end,:);
%     img_ia1_j = temp(:,1:spatialSize:end);
    
    img_ia1_j = [img_i_j; img_i_j(end,:)];
    img_ia1_j(1,:) = [];
    
    % translate one row (subtraction - spatialSize)
%     img_is1_j_ = [img(1:spatialSize,:); img];
%     img_is1_j_((end-spatialSize+1):end,:) = [];
%     temp = img_is1_j_(1:spatialSize:end, :);
%     img_is1_j = temp(:, 1:spatialSize:end);
    
    img_is1_j = [img_i_j(1,:); img_i_j];
    img_is1_j(end,:) = [];

    % translate one col (add + spatialSize)
%     img_i_ja1_ = [img img(:, (end-spatialSize+1):end)];
%     img_i_ja1_(:,1:spatialSize) = [];
%     temp = img_i_ja1_(1:spatialSize:end, :);
%     img_i_ja1 = temp(:, 1:spatialSize:end);
    
    img_i_ja1 = [img_i_j img_i_j(:, end)];
    img_i_ja1(:, 1) = [];
    
    
    % translate one col (subtraction - spatialSize)
%     img_i_js1_ = [img(:,1:spatialSize) img];
%     img_i_js1_(:,(end-spatialSize+1):end) = [];
%     temp = img_i_js1_(1:spatialSize:end, :);
%     img_i_js1 = temp(:, 1:spatialSize:end);
    
    img_i_js1 = [img_i_j(:,1) img_i_j];
    img_i_js1(:,end) = [];
    
    %
    % STEP1: compute derivative approximation and Laplacian approximation
    %
    deltaR1Img = (img_ia1_j - img_i_j)/spatialSize;
    deltaR2Img = (img_i_ja1 - img_i_j)/spatialSize;
    
    deltaL1Img = (img_i_j - img_is1_j)/spatialSize;    
    deltaL2Img = (img_i_j - img_i_js1)/spatialSize;
  
    delta2Img = (img_ia1_j + img_is1_j + img_i_ja1 + img_i_js1 - 4*img_i_j)/(spatialSize*spatialSize);
    
    %
    % STEP2: caculate the diffusion coefficient
    %
    
    % normalizing the gradient of each image point
    gradientTotal = sqrt(deltaR1Img.*deltaR1Img + deltaR2Img.*deltaR2Img + deltaL1Img.*deltaL1Img + deltaL2Img.*deltaL2Img);
    deltaImgNormal = gradientTotal./(img_i_j+ epsilon);
    
    % normalizing the Laplacian of each imge point
    delta2ImgNormal = delta2Img./(img_i_j+epsilon);
    
    % compute the initial Q
    temp1 = (deltaImgNormal.*deltaImgNormal)*0.5 - delta2ImgNormal.*delta2ImgNormal/16;
    temp2 = 1+0.25*delta2ImgNormal;
    temp3 = temp2.*temp2;
    q = sqrt(temp1./temp3);
    Q = q; 
    
    Q0 = logical(Q);
    Q0 = single(Q0);
    

    Img_i_j = img_i_j;

%%

while iterationNumber <= iterationMaxStep 
    
    img_i_j = Img_i_j;
    
    % translate one row (add + spatialSize)
    img_ia1_j = [img_i_j; img_i_j(end,:)];
    img_ia1_j(1,:) = [];
    
    % translate one row (subtraction - spatialSize)
    img_is1_j = [img_i_j(1,:); img_i_j];
    img_is1_j(end,:) = [];
    

    % translate one col (add + spatialSize)
    img_i_ja1 = [img_i_j img_i_j(:, end)];
    img_i_ja1(:, 1) = [];
    
    
    % translate one col (subtraction - spatialSize)    
    img_i_js1 = [img_i_j(:,1) img_i_j];
    img_i_js1(:,end) = [];
    
    %
    % STEP1: compute derivative approximation and Laplacian approximation
    %
    deltaR1Img = (img_ia1_j - img_i_j)/spatialSize;
    deltaR2Img = (img_i_ja1 - img_i_j)/spatialSize;
    deltaL1Img = (img_i_j - img_is1_j)/spatialSize;    
    deltaL2Img = (img_i_j - img_i_js1)/spatialSize;
  
    delta2Img = (img_ia1_j + img_is1_j + img_i_ja1 + img_i_js1 - 4*img_i_j)/(spatialSize*spatialSize);
    
    %
    % STEP2: caculate the diffusion coefficient
    %
    
    % normalizing the gradient of each image point
    gradientTotal = sqrt(deltaR1Img.*deltaR1Img + deltaR2Img.*deltaR2Img + deltaL1Img.*deltaL1Img + deltaL2Img.*deltaL2Img);
    deltaImgNormal = gradientTotal./(img_i_j+ epsilon);
    
    % normalizing the Laplacian of each imge point
    delta2ImgNormal = delta2Img./(img_i_j+epsilon);
    
    % compute the diffusion cefficient
    temp1 = (deltaImgNormal.*deltaImgNormal)*0.5 - delta2ImgNormal.*delta2ImgNormal/16;
    temp2 = 1+0.25*delta2ImgNormal;
    temp3 = temp2.*temp2;
    q = sqrt(temp1./temp3);
    
    
    q0 = Q0*exp(-decayFactor*t);
    
    % method 1: coefficientDiff(q) = 1/{1+[q(t)*q(t) - q(t=0)*q(t=0)]/[q(t=0)*q(t=0)(1+q(t=0)*q(t=0))]}
    temp4 = q0.*q0;
    temp5 = q.*q;
%     coefficientDiff = 1 + (temp5 - temp4)./(temp4.*(1+ temp4) + epsilon);
%     coefficientDiff = 1./(coefficientDiff + epsilon);
%     
    % method 2: coefficientDiff(q) = exp^{-[q(t)*q(t) - q(t=0)*q(t=0)]/[q(t=0)*q(t=0)(1+q(t)*q(t))]}
    
    temp6 = (temp5-temp4)./(temp4.*(1+temp4) + epsilon);
    coefficientDiff = exp(-temp6/6);
    
    %
    % STEP3: caculate the divergence of diffusion function
    %
    
    coe_i_j = coefficientDiff;
    
    coe_ia1_j = [coefficientDiff; coefficientDiff(end,:)];
    coe_ia1_j(1,:) = [];
    
    coe_i_ja1 = [coefficientDiff coefficientDiff(:,end)];
    coe_i_ja1(:,1) = [];
    
    temp6 = coe_ia1_j.*deltaR1Img - coe_i_j.*deltaL1Img + coe_i_ja1.*deltaR2Img - coe_i_j.*deltaL2Img;
    
    div = temp6/spatialSize;
    
    
    % STEP4: define the iteration rule
    Img_i_j = img_i_j + (timeSize/4)*div;
    
    t = t + timeSize; 
    iterationNumber = iterationNumber + 1;  
   
end

%
%  STEP5: restore the image
%

mask = ones(ROW, COL);
spatialSize = 1;
mask(1:spatialSize:end, 1:spatialSize:end) = 0;

despeckledImg = img;
despeckledImg = img.*mask;

[ROW_, COL_] = size(Img_i_j);


for i = 1: ROW_
    for j = 1:COL_
        
        despeckledImg(i*spatialSize, j*spatialSize) = Img_i_j(i,j);
        
    end
end

% restore the original data type
despeckledImg = cast(despeckledImg, originalClass);






end