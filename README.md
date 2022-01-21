# Speckle-Reducing-Anisotropic-Diffusion(SRAD)
Speckle Reducing Anisotropic Diffusion (SRAD) Algorithm 

# Description
- SpeckleReducingAD.m: Speckle Reducing Anisotropic Diffusion Algortithm (General Version)

- SpeckleReducingAD_New.m: Speckle Reducing Anisotropic Diffusion Algortihm (Optimized Version for my data)

- testSRAD.m: test our SRAD algorithm using the input (noisyImage.png) and get the output

# Reference Paper
- [Yu, Yongjian, and Scott T. Acton. "Speckle reducing anisotropic diffusion." IEEE Transactions on image processing 11.11 (2002): 1260-1270.](https://ieeexplore.ieee.org/document/1097762)

- [Perona P, Malik J. Scale-space and edge detection using anisotropic diffusion[J]. IEEE Transactions on pattern analysis and machine intelligence, 1990, 12(7): 629-639.](https://ieeexplore.ieee.org/document/56205)

# Basic Principle
This algorithm orignated from Anisotropic Diffusion, the mathematical model is :
![AD_Principle](https://github.com/Xingorno/Figures/blob/master/Image_Folder/AD_Principle.png?raw=true)

Note: the most important part of the model is to define the diffusion function 
<img src="http://latex.codecogs.com/svg.latex? c(*)" border="0"/> and the diffusion efficient 
<img src="http://latex.codecogs.com/svg.latex? q" border="0"/>

# Pseudocode
![pseudocode](https://github.com/Xingorno/Figures/blob/master/Image_Folder/pseudocode.png?raw=true)
