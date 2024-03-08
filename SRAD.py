import numpy as np
import cv2

# functions

## SRAD function
def SRAD(img, iterationMaxStep, timeSize, decayFactor):

    # # function despeckledImg = MySRAD(img, iterationMaxStep, threshodConvergence, timeSize, decayFactor)
    # # Algorithm: Speckle Reducing Anisotropic Diffusion Method
    # # Input:
    # #   - img: the input image with speckle noise; data type [any kinds of], size N*M  
    # #   - interationMaxStep: the maximum iterative step; data type [integer(>0)]
    # #   - thresholdCovergence: the second ending rule for iteration; data type
    # #   [undefine]
    # #   - timeSize: the time step in each iteration; data type [float or
    # #   integer], scalar
    # #   - dacayFactor: the decay factor for the exp function related to
    # #   "smoothing and edge preserving"; data type [float, integer]. scalar

    # # Output:
    # #   - despeckledImg: the despeckled image; data type [same as the img], size
    # #   N*M

    # #
    # # INPUT PARAMETERS
    # #
    spatialSize = 1 ##spatial step size
    t = 0 ## starting timestap
    thresholdDiffusion = 0.0001 ## threshold for diffusion coefficient to guarantee the lower vaule (equals 0)
    thresholdConvergence = 10**(-13) ## one of the iteration determination conditions
    ROW, COL = img.shape ## input image size
    epsilon = 10**(-13) ## handle 0/0 case 

    iterationNumber = 0 ## variable of iterating step
    diviationConvergence = 1 ## the convergence diviation each iteration
   
    originalClass = type(img)

    img = img.astype('float32')
    #img = np.genfromtxt('test.txt', delimiter=',')
    

    img_i_j_ = img
    temp = img_i_j_[::spatialSize, :]
    img_i_j =temp[:,::spatialSize]
    
    ##   # translate one row (add + spatialSize)
    # #     img_ia1_j_ = [img; img((end-spatialSize+1) :end,:)];
    # #     img_ia1_j_(1:spatialSize,:) = [];
    # #     temp = img_ia1_j_(1:spatialSize:end,:);
    # #     img_ia1_j = temp(:,1:spatialSize:end);
    
    img_ia1_j = np.row_stack((img_i_j, img_i_j[-1,:]))
    img_ia1_j = img_ia1_j[1:,:]
    
    ##     # translate one row (subtraction - spatialSize)
    # #     img_is1_j_ = [img(1:spatialSize,:); img];
    # #     img_is1_j_((end-spatialSize+1):end,:) = [];
    # #     temp = img_is1_j_(1:spatialSize:end, :);
    # #     img_is1_j = temp(:, 1:spatialSize:end);
    
    img_is1_j = np.row_stack((img_i_j[0,:], img_i_j))
    img_is1_j = img_is1_j[:-1,:]

    ##     # translate one col (add + spatialSize)
    # #     img_i_ja1_ = [img img(:, (end-spatialSize+1):end)];
    # #     img_i_ja1_(:,1:spatialSize) = [];
    # #     temp = img_i_ja1_(1:spatialSize:end, :);
    # #     img_i_ja1 = temp(:, 1:spatialSize:end);
    
    img_i_ja1 = np.column_stack((img_i_j, img_i_j[:,-1]))
    img_i_ja1 = img_i_ja1[:,1:]
    
    
    #     # translate one col (subtraction - spatialSize)
    # #     img_i_js1_ = [img(:,1:spatialSize) img];
    # #     img_i_js1_(:,(end-spatialSize+1):end) = [];
    # #     temp = img_i_js1_(1:spatialSize:end, :);
    # #     img_i_js1 = temp(:, 1:spatialSize:end);
    
    img_i_js1 = np.column_stack((img_i_j[:,0], img_i_j))
    img_i_js1 = img_i_js1[:,:-1]

    ## #
    # # STEP1: compute derivative approximation and Laplacian approximation
    # #
    deltaR1Img = (img_ia1_j - img_i_j)/spatialSize
    deltaR2Img = (img_i_ja1 - img_i_j)/spatialSize
    
    deltaL1Img = (img_i_j - img_is1_j)/spatialSize
    deltaL2Img = (img_i_j - img_i_js1)/spatialSize
  
    delta2Img = (img_ia1_j + img_is1_j + img_i_ja1 + img_i_js1 - 4*img_i_j)/(spatialSize*spatialSize)

    ## #
    # # STEP2: caculate the diffusion coefficient
    # #
    
    ## normalizing the gradient of each image point
    gradientTotal = np.sqrt( np.multiply(deltaR1Img,deltaR1Img) + np.multiply(deltaR2Img,deltaR2Img) + np.multiply(deltaL1Img,deltaL1Img) + np.multiply(deltaL2Img,deltaL2Img) )
    deltaImgNormal = np.divide(gradientTotal,(img_i_j + epsilon))
    
    ## normalizing the Laplacian of each imge point
    delta2ImgNormal = np.divide(delta2Img,(img_i_j+epsilon))
    
    ## # compute the initial Q
    temp1 = np.multiply(deltaImgNormal,deltaImgNormal)*0.5 - np.multiply(delta2ImgNormal,delta2ImgNormal)/16
    temp2 = 1+0.25*delta2ImgNormal
    temp3 = np.multiply(temp2,temp2)
        
    q = np.sqrt(np.divide(temp1,temp3))
    Q = q
    
    #IDK# Q_0 = logical(Q)
    Q_0 = Q != 0
    #IDK# Q_0 = single(Q_0)
    Q_0 = Q_0.astype('float32')

    Img_i_j = img_i_j
    

####

    while iterationNumber <= iterationMaxStep:
        img_i_j = Img_i_j

        # # translate one row (add + spatialSize)
        img_ia1_j = np.row_stack((img_i_j, img_i_j[-1,:]))
        img_ia1_j = img_ia1_j[1:,:]
        
        # # translate one row (subtraction - spatialSize)
        img_is1_j = np.row_stack((img_i_j[0,:], img_i_j))
        img_is1_j = img_is1_j[:-1,:]
        

        # # translate one col (add + spatialSize)
        img_i_ja1 = np.column_stack((img_i_j, img_i_j[:,-1]))
        img_i_ja1 = img_i_ja1[:,1:]
        
        
        # # translate one col (subtraction - spatialSize)    
        img_i_js1 = np.column_stack((img_i_j[:,0], img_i_j))
        img_i_js1 = img_i_js1[:,:-1]
        
        # #
        # # STEP1: compute derivative approximation and Laplacian approximation
        # #
        deltaR1Img = (img_ia1_j - img_i_j)/spatialSize
        deltaR2Img = (img_i_ja1 - img_i_j)/spatialSize
        deltaL1Img = (img_i_j - img_is1_j)/spatialSize
        deltaL2Img = (img_i_j - img_i_js1)/spatialSize

        delta2Img = (img_ia1_j + img_is1_j + img_i_ja1 + img_i_js1 - 4*img_i_j)/(spatialSize*spatialSize)
        
        # #
        # # STEP2: caculate the diffusion coefficient
        # #
        
        # # normalizing the gradient of each image point
        gradientTotal = np.sqrt( np.multiply(deltaR1Img,deltaR1Img) + np.multiply(deltaR2Img,deltaR2Img) + np.multiply(deltaL1Img,deltaL1Img) + np.multiply(deltaL2Img,deltaL2Img) )
        deltaImgNormal = np.divide(gradientTotal,(img_i_j+ epsilon))
        
        # # normalizing the Laplacian of each imge point
        delta2ImgNormal = np.divide(delta2Img,(img_i_j+epsilon))
        
        # # compute the diffusion cefficient
        temp1 = np.multiply(deltaImgNormal,deltaImgNormal)*0.5 - np.multiply(delta2ImgNormal,delta2ImgNormal)/16
        temp2 = 1 + 0.25*delta2ImgNormal
        temp3 = np.multiply(temp2,temp2)
        q = np.sqrt(np.divide(temp1,temp3))
        
        
        q_0 = Q_0 * np.exp(-decayFactor*t)
        
        ## method 1: coefficientDiff(q) = 1/{1+[q(t)*q(t) - q(t=0)*q(t=0)]/[q(t=0)*q(t=0)(1+q(t=0)*q(t=0))]}
        temp4 = np.multiply(q_0,q_0)
        temp5 = np.multiply(q,q)
        ##     coefficientDiff = 1 + (temp5 - temp4)./(temp4.*(1+ temp4) + epsilon);
        ##     coefficientDiff = 1./(coefficientDiff + epsilon);
        ##     
        ## method 2: coefficientDiff(q) = exp^{-[q(t)*q(t) - q(t=0)*q(t=0)]/[q(t=0)*q(t=0)(1+q(t)*q(t))]}
        
        temp6 = np.divide( (temp5-temp4), ( np.multiply(temp4, (1+temp4) ) + epsilon) )
        coefficientDiff = np.exp(-temp6/6)
        
        # #
        # # STEP3: caculate the divergence of diffusion function
        # #
        
        coe_i_j = coefficientDiff
        
        coe_ia1_j = np.row_stack((coefficientDiff, coefficientDiff[-1,:]))
        coe_ia1_j = coe_ia1_j[1:,:]
        
        coe_i_ja1 = np.column_stack((coefficientDiff, coefficientDiff[:,-1]))
        coe_i_ja1 = coe_i_ja1[:,1:]
        temp6 = np.multiply(coe_ia1_j,deltaR1Img) - np.multiply(coe_i_j,deltaL1Img) + np.multiply(coe_i_ja1,deltaR2Img) - np.multiply(coe_i_j,deltaL2Img)
        div = temp6/spatialSize

        # # STEP4: define the iteration rule
        Img_i_j = img_i_j + (timeSize/4)*div
        
        t = t + timeSize
        
        iterationNumber = iterationNumber + 1
    
    # #
    # #  STEP5: restore the image
    # #

    mask = np.ones((ROW, COL))
    spatialSize = 1
    mask[::spatialSize, ::spatialSize] = 0


    despeckledImg = img
    despeckledImg = np.multiply(img, mask)

    ROW_, COL_ = Img_i_j.shape

    for i in range(ROW_):
        for j in range(COL_): 
            despeckledImg[i*spatialSize][j*spatialSize] = Img_i_j[i][j]
        
    ## # restore the original data type
    # despeckledImg = cast(despeckledImg, originalClass);
    return despeckledImg.astype("uint8")

f = 'data/noisyImage.png'
img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#np.savetxt("initial.csv", img, delimiter=",")


iterationMaxStep, timeSize, decayFactor = 200,.05,1
img = SRAD(img, iterationMaxStep, timeSize, decayFactor)

cv2.imwrite('denoised.png',img)