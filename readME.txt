
Action Classification:

- considering the interactions between different body parts and joints of the person who is performing the action.

- 3D skeleton sequences extracted from depth data via a Kinect sensor. 

- We group the joints into different body parts and recognize the action by encoding the spatial changes between the joint locations over time and between the body parts as a dis- criminative descriptor for the video sequences

- We use Fisher Vector to encode these spatial feature matrices.

- SVM classification

- on MSR3D-action dataset.- We applied 28 different techniques including forward feature selection / fisher vector / bag of words technique / considering all lines in the skeleton / considering anatomy skeleton / using PCA to reduce features / HMM / SVM / different spatio-temporal features / …

- The main function is “SkeletonActRecog.m