
function despeckledImg = SpeckleReducingAD_New(img, iterationMaxStep, timeSize, decayFactor, conductionMethod)

% Algorithm: Speckle Reducing Anisotropic Diffusion Method

% Input:
%   - img: the input image with speckle noise; data type [any kinds of], size N*M  
%   - interationMaxStep: the maximum iterative step; data type [integer(>0)]
%   - thresholdCovergence: the second ending rule for iteration; data type
%   [undefine]
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
%  - Matlab function: imdiffusefilt(*)

%
% INPUT PARAMETERS
%
    spatialSize = 1; %spatial step size
    t = 0; % starting timestap
    thresholdDiffusion = 0.0001; % threshold for diffusion coefficient to guarantee the lower vaule (equals 0)
    thresholdConvergence = 10^(-13); % one of the iteration determination conditions
    [ROW, COL] = size(img); % input image size
    iterationNumber = 0; % variable of iterating step
    diviationConvergence = 1; % the convergence diviation each iteratio
    originalClass = class(img);
    epsilon = 10^(-13); % handle 0/0 case

    if ~isa(img,'double')
        img = single(img);
    end

    img_i_j_ = img;
    temp = img_i_j_(1:spatialSize:end, :);
    img_i_j =temp(:,1:spatialSize:end);
    
    paddedImg = padarray(img_i_j, [1, 1], 'replicate');
    diffImgRow = (paddedImg(2:end, 2:end-1) - paddedImg(1:end-1, 2:end-1))/spatialSize;
    diffImgCol = (paddedImg(2:end-1, 2:end) - paddedImg(2:end-1, 1:end-1))/spatialSize;
    
%
% compute derivative approximation and Laplacian approximation
%
    delta2Img_ = (paddedImg(3:end, 2:end-1) + paddedImg(1:end-2, 2:end-1) + paddedImg(2:end-1, 3:end) + paddedImg(2:end-1, 1:end-2) -4*paddedImg(2:end-1, 2:end-1));
    delta2Img = delta2Img_/(spatialSize*spatialSize);
    
    % normalizing the gradient of each image point
    diffImgRow2 = diffImgRow.*diffImgRow;
    diffImgCol2 = diffImgCol.*diffImgCol;
    gradientTotal = sqrt(diffImgRow2(2:end,:) + diffImgRow2(1:end-1,:) + diffImgCol2(:,2:end) + diffImgCol2(:,1:end-1));
    deltaImgNormal = gradientTotal./(img_i_j+ epsilon);
    
    % normalizing the Laplacian of each imge point
    delta2ImgNormal = delta2Img./(img_i_j+epsilon);
    
    % compute the initial Q
    temp1 = (deltaImgNormal.*deltaImgNormal)*0.5 - delta2ImgNormal.*delta2ImgNormal/16;
    temp2 = 1+0.25*delta2ImgNormal;
    temp3 = temp2.*temp2;
    q = sqrt(temp1./temp3);
    Q = q; 
    %Note:the initial value about Q0 is very very important
    Q0 = logical(Q);
    Q0 = single(Q0);
    Img_i_j = img_i_j;

%%
while iterationNumber <= iterationMaxStep 
    
    img_i_j = Img_i_j;
%
% STEP1: COMPUTE DERIVATIVE APPROXIMATION AND LAPLACIAN APPROXIMATION
%
    paddedImg = padarray(img_i_j, [1, 1], 'replicate');
    diffImgRow = (paddedImg(2:end, 2:end-1) - paddedImg(1:end-1, 2:end-1))/spatialSize;
    diffImgCol = (paddedImg(2:end-1, 2:end) - paddedImg(2:end-1, 1:end-1))/spatialSize;
   
    delta2Img_ = (paddedImg(3:end, 2:end-1) + paddedImg(1:end-2, 2:end-1) + paddedImg(2:end-1, 3:end) + paddedImg(2:end-1, 1:end-2) -4*paddedImg(2:end-1, 2:end-1));
    delta2Img = delta2Img_/(spatialSize*spatialSize);
    
%
% STEP2: CACULATE THE DIFFUSION COEFFICIENT
%
    
    % normalizing the gradient of each image point 
    diffImgRow2 = diffImgRow.*diffImgRow;
    diffImgCol2 = diffImgCol.*diffImgCol;
    gradientTotal = sqrt(diffImgRow2(2:end,:) + diffImgRow2(1:end-1,:) + diffImgCol2(:,2:end) + diffImgCol2(:,1:end-1));
    deltaImgNormal = gradientTotal./(img_i_j+ epsilon);
    
    % normalizing the Laplacian of each imge point
    delta2ImgNormal = delta2Img./(img_i_j+epsilon);
    
    % compute the diffusion cefficient
    temp1 = (deltaImgNormal.*deltaImgNormal)*0.5 - delta2ImgNormal.*delta2ImgNormal/16;
    temp2 = 1+0.25*delta2ImgNormal;
    temp3 = temp2.*temp2;
    q = sqrt(temp1./temp3);
    q0 = Q0*exp(-decayFactor*t);
    
    temp4 = q0.*q0;
    temp5 = q.*q;
    
    switch conductionMethod
        case 'exponential'
            % method : coefficientDiff(q) = exp^{-[q(t)*q(t) - q(t=0)*q(t=0)]/[q(t=0)*q(t=0)(1+q(t)*q(t))]}
            temp6 = (temp5-temp4)./(temp4.*(1+temp4) + epsilon);
            coefficientDiff = exp(-temp6);
        
        case 'quadratic'
            % method : coefficientDiff(q) = 1/{1+[q(t)*q(t) - q(t=0)*q(t=0)]/[q(t=0)*q(t=0)(1+q(t=0)*q(t=0))]}
            coefficientDiff = 1 + (temp5 - temp4)./(temp4.*(1+ temp4) + epsilon);
            coefficientDiff = 1./(coefficientDiff + epsilon);
    end
    
%
% STEP3: CACULATE THE DIVERGENCE OF DIFFUSION FUNCTION
%
    
    coe_i_j = coefficientDiff;   
    coe_ia1_j = [coefficientDiff; coefficientDiff(end,:)];
    coe_ia1_j(1,:) = [];   
    coe_i_ja1 = [coefficientDiff coefficientDiff(:,end)];
    coe_i_ja1(:,1) = [];    
    temp6 = coe_ia1_j.*diffImgRow(2:end,:) - coe_i_j.*(diffImgRow(1:end-1,:)+ diffImgCol(:,1:end-1)) + coe_i_ja1.*diffImgCol(:,2:end);   
    div = temp6/spatialSize;
    
%
% STEP4: DEFINE THE ITERATION RULE
%
    Img_i_j = img_i_j + (timeSize/4)*div;
    t = t + timeSize; 
    iterationNumber = iterationNumber + 1;  
end

%
%  STEP5: RESTORE THE IMAGE
%
    mask = ones(ROW, COL);
    mask(1:spatialSize:end, 1:spatialSize:end) = 0;
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