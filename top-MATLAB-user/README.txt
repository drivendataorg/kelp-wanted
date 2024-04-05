MATLAB Solution Requirements:

    MATLAB 2024a Prerelease (Deep Learning Toolbox, Computer Vision Toolbox, Image Processing Toolbox)
    A system with Nvidia GPU (with 24GB memory) and CUDA toolkit installed.

Steps for running the codes:

-	If you wish to use pretrained networks, you need to change the value of “use_pretrained” variable to True at the beginning of the “src/inference.m” file. The code uses the pretrained models to predict the labels and will save them under the “data/final_labels” folder.

-	Run the “src/main.m” file to train the 19 models. Trained networks will be saved under the “trained_models” folder.

-	Run the “src/inference.m” file (the value of “use_pretrained” variable should be changed to False). The code uses the trained models to predict the labels and will save them under the “data/final_labels” folder. 
