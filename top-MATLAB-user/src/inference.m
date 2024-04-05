clear
close all
clc
addpath("models/")
addpath("utils/")

%% Using pretrained models?
% If you want to train new models you need to run the main file first 
% Then, you need to set the "use_pretrained" variable to false. 
% If false, this code uses newly trained model saved on trained_models folder
use_pretrained = true;
if use_pretrained
    folder_trained = "../pretrained_models/";
else
    folder_trained = "../trained_models/";
end

%% Providing data path  
s3Path = '../data';

%% Importing and Pre-processing the Data
kelp_train_Folder = fullfile(s3Path, 'train_features/train_satellite/');
kelp_test_Folder = fullfile(s3Path, 'test_features/test_satellite/');


%% Labeling images

Files = dir(strcat(folder_trained+'*.mat'));

for i=1:size(Files,1)

    file_name = strcat(folder_trained, Files(i).name);
    load(file_name);
    if use_pretrained
        net = net;
    else
        net = trained_net;
    end

    % Test
    imtest =  imageDatastore(kelp_test_Folder, 'ReadFcn', @(filename)readTest(filename), 'LabelSource', 'foldernames');
    if strcmp('trainedNetwork_unet_rand_52.mat', Files(i).name) % Using different image reading fucntions for this file
        imtest =  imageDatastore(kelp_test_Folder, 'ReadFcn', @(filename)readTest2(filename), 'LabelSource', 'foldernames');
    end

    % Saving predicted labels for each network
    for ii=1:size(imtest.Files,1)
        input_label = imtest.read; % Reading input 
        predict_test = predict(net, input_label, 'ExecutionEnvironment', 'gpu'); % Predicting labels
        predict_test_bin = predict_test>0.5;
        outputFilenameParts = imtest.Files{ii};
        name =   strcat(outputFilenameParts(end-21:end-14), "_kelp.tif");
        save_loc = ['../data/test_labels/' Files(i).name '/'];
        if ~exist(save_loc, 'dir')
            mkdir(save_loc)
        end
        filename = fullfile(save_loc, name);
        imwrite(predict_test_bin, filename);
    end

    % Saving sigmoid outpus for each network
    imtest.reset
    for ii=1:size(imtest.Files,1)
        input_label = imtest.read; % Reading input
        predict_test = predict(net, input_label, 'ExecutionEnvironment', 'gpu'); % Predicting labels
 
        outputFilenameParts = imtest.Files{ii};
        name =   strcat(outputFilenameParts(end-21:end-14), "_kelp.tif");
        save_loc = ['../data/sigmoid_labels/' Files(i).name '/'];
        if ~exist(save_loc, 'dir')
            mkdir(save_loc)
        end
        filename = fullfile(save_loc, name);
        save(filename, 'predict_test');
    end
end

%% Combining (average) labels that are obtained from differert models
models_list_dir = dir('../data/sigmoid_labels/*.mat');
models_list = string({models_list_dir.name})';
for ii=1:size(imtest.Files,1)
    temp =zeros(350,350);
    outputFilenameParts = imtest.Files{ii};
    name =   strcat('/',outputFilenameParts(end-21:end-14), "_kelp.tif");

    for jj=1:size(models_list,1)
        filename = '../data/sigmoid_labels/' + models_list(jj)+name;
        img = matfile(filename);
        temp = temp + img.predict_test;
    end
    new_img = temp/(size(models_list,1));
    predict_test_bin = new_img>=0.5;

    save_loc = "../data/final_lablels/"; % Save labels that are obtained from combining all models
    if ~exist(save_loc, 'dir')
        mkdir(save_loc)
    end
    filename = fullfile(save_loc, name);
    imwrite(predict_test_bin, filename);
    
end

%% Helper functions

function output = readTest(outputFilename)
% Reading test images
    fullfilename = outputFilename;
    im = imread(fullfilename);
    im = double(im);
    
    output = im;

end

function output = readTest2(outputFilename)
% Reading test images and removing outliers
    fullfilename = outputFilename;
    im = imread(fullfilename);
    im = double(im);
    im(im>43636 ) = 43636 ;
    im(im<0 ) = 0 ;
    output = im;

end

