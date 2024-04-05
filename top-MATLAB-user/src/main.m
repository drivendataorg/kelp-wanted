clear
close all
clc
addpath("models/")
addpath("utils/")

%% Defining the models (19 U-net models)

net1 = trainedNetwork_12();  net2 = trainedNetwork_13(); net3 = trainedNetwork_15();
net4 = trainedNetwork_16();  net5 = trainedNetwork_17(); net6 = trainedNetwork_21();
net7 = trainedNetwork_26();  net8 = trainedNetwork_unet5(); net9 = trainedNetwork_unet9();
net10 = trainedNetwork_unet14();  net11 = trainedNetwork_unet15(); net12 = trainedNetwork1_kf3(); 
net13 = trainedNetwork1_kf5(); net14 = trainedNetwork_unet_rand_8();net15 = trainedNetwork_unet_rand_15();  
net16 = trainedNetwork_unet_rand_17(); net17 = trainedNetwork_unet_rand_23();
net18 = trainedNetwork_unet_rand_25();  net19 = trainedNetwork_unet_rand_52();

% [model, loss_type, squared_dice, Alpha, Gamma, remove_images, ...
% return_images, model_remove, dice_threshold, union_threshold, partition_type, i_rand, network_name]
models = {{net1, 1, false, 10, 2, false, true, nan, nan, nan, 1, -1, 'trainedNetwork_12'}; 
    {net2, 1, false, 10, 2, false, true, nan, nan, nan, 1, 0, 'trainedNetwork_13'}; 
    {net3, 2, false, 10, 2, false, true, nan, nan, nan, 1, 13, 'trainedNetwork_15'}; 
    {net4, 2, false, 10, 2, false, true, nan, nan, nan, 1, -1, 'trainedNetwork_16'}; 
    {net5, 1, false, 10, 2, true, true, 'trainedNetwork_16.m', 0.017, 0, 1, -1, 'trainedNetwork_17'}; 
    {net6, 1, false, 10, 2, true, true, 'trainedNetwork_17.m', 0.05, 20, 1, -1, 'trainedNetwork_21'}; 
    {net7, 1, false, 1000, 2, true, true, 'trainedNetwork_17.m', 0.05, 20, 1, -1, 'trainedNetwork_26'}; 
    {net8, 2, true, 20, 2, true, true, 'trainedNetwork_17.m', 0.05, 20, 1, -1, 'trainedNetwork_unet5'}; 
    {net9, 1, true, 20, 2.5, true, true, 'trainedNetwork_17.m', 0.05, 20, 1, -1, 'trainedNetwork_unet9'}; 
    {net10, 1, true, 20, 2.5, true, true, 'trainedNetwork_17.m', 0.05, 20, 1, -1, 'trainedNetwork_unet14'}; 
    {net11, 1, true, 20, 2.0, true, true, 'trainedNetwork_17.m', 0.05, 20, 1, -1, 'trainedNetwork_unet15'};
    {net12, 1, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 2, 3, 'trainedNetwork1_kf3'};
    {net13, 1, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 2, 5, 'trainedNetwork1_kf5'};
    {net14, 2, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 3, 8, 'trainedNetwork_unet_rand_8'}
    {net15, 2, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 4, 15, 'trainedNetwork_unet_rand_15'}
    {net16, 2, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 4, 17, 'trainedNetwork_unet_rand_17'}
    {net17, 2, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 4, 23, 'trainedNetwork_unet_rand_23'}
    {net18, 2, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 4, 25, 'trainedNetwork_unet_rand_25'}
    {net19, 2, false, 20, 2, true, true,'trainedNetwork_17.m', 0.05, 20, 5, 52, 'trainedNetwork_unet_rand_52'};
    };

%%  Data path  
s3Path = '../data/';
kelp_train_Folder = fullfile(s3Path, 'train_features/train_satellite/');
kelp_test_Folder = fullfile(s3Path, 'test_features/test_satellite/');

%% Iteration over models and training networks
for ii =1:size(models, 1)
    disp(["model:  ", ii])
    model_temp = models{ii}{1};
    loss_type = models{ii}{2};
    squared_dice = models{ii}{3};
    Alpha =  models{ii}{4};
    Gamma =  models{ii}{5};
    remove_images =  models{ii}{6};
    return_images =  models{ii}{7};
    model_remove =  models{ii}{8};
    dice_thresh =  models{ii}{9};
    union_thresh =  models{ii}{10};
    partitiontype =  models{ii}{11};
    i_rand =  models{ii}{12};
    net_name =  models{ii}{13};

    if strcmp(net_name, 'trainedNetwork_16')
        last_val= true;
    else
        last_val= false;
    end

    % Returning removed images (they are removed because of poor labeling in the previous iterations) to the main image folder
    return_images_func(return_images)

    % Dividing data to training and validation sets
    % Reading input data
    if ~strcmp(net_name, 'trainedNetwork_unet_rand_52')
        imInput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readTrainingSatelliteData(filename),'LabelSource', 'foldernames');
    else
        imInput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readTrainingSatelliteData2(filename), 'LabelSource', 'foldernames');
    end
    % Output data
    imOutput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readlabel(filename), 'LabelSource', 'foldernames');
    dsInput= combine(imInput, imOutput);


    % Remove poor classified images 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if any(strcmp({'trainedNetwork_12', 'trainedNetwork_13', 'trainedNetwork_15', 'trainedNetwork_16'}, net_name))
        model_add = nan;
        rm_imgs(dsInput, model_add, dice_thresh, union_thresh, remove_images)

    elseif any(strcmp({'trainedNetwork_17'}, net_name))
        model_add = strcat('../trained_models/', 'trainedNetwork_16', '.mat');
        rm_imgs(dsInput, model_add, dice_thresh, union_thresh, remove_images)
         % Reading train data after removing 
        imInput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readTrainingSatelliteData(filename),'LabelSource', 'foldernames');
        imOutput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readlabel(filename), 'LabelSource', 'foldernames');

    else
        model_add = strcat('../trained_models/', 'trainedNetwork_17', '.mat');
        rm_imgs(dsInput, model_add, dice_thresh, union_thresh,  remove_images)
        % Reading train data after removing 
        if ~strcmp(net_name, 'trainedNetwork_unet_rand_52')
            imInput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readTrainingSatelliteData(filename),'LabelSource', 'foldernames');
        else
            imInput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readTrainingSatelliteData2(filename), 'LabelSource', 'foldernames');
        end
        imOutput = imageDatastore(kelp_train_Folder, 'ReadFcn', @(filename)readlabel(filename), 'LabelSource', 'foldernames');

    end
    % Divide data
    [inputTrain, inputVal, outputTrain, outputVal] = partitionData(imInput, imOutput, i_rand, partitiontype, last_val);
    dsTrain = combine(inputTrain, outputTrain);
    dsVal = combine(inputVal, outputVal);
    % Preparing network and adding loss function
    lgraph = layerGraph(model_temp);
    lgraph = lgraph.addLayers(loss_layer('loss_layer', loss_type, squared_dice, Alpha, Gamma));
    lgraph = lgraph.connectLayers('NetworksoftmaxLayer-Layer','loss_layer');

    % Training options
    options = trainingOptions('adam', ...
        'LearnRateSchedule','piecewise', ...
        'MiniBatchSize', 16, ...
        'MaxEpochs', 200, ...
        'InitialLearnRate',0.0002, ...,
        'Shuffle', 'every-epoch',...
        'LearnRateDropFactor',0.5,...
        'LearnRateDropPeriod',50,...
        'ValidationData', dsVal, ...
        'OutputNetwork', 'best-validation-loss', ...
        'Verbose', true,...
        'ValidationFrequency', 292,...
        'Plots','training-progress', ...
        'OutputNetwork', 'best-validation-loss', ...
        ExecutionEnvironment="gpu");
    

    % Training network
    trained_net = trainNetwork(dsTrain, lgraph, options);

    save_loc = "../trained_models/" ;
    if ~exist(save_loc, 'dir')
        mkdir(save_loc)
    end
    % Saving network
    save(strcat(save_loc, net_name, ".mat"),'trained_net') 
  
end

%% Helper functions

function [] = rm_imgs(imInput, model_add, dice_thresh, union_thresh, remove_imgs)
% Removing images with poor labeling based on a threshold
if remove_imgs
    net = load(model_add);
    dataset = imInput; 
    reset(dataset)
    union_val = 0;
    intersection_val = 0;
    for ii=1:size(dataset.UnderlyingDatastores{1,1}.Files,1)
        input_label = dataset.read; % Reading input and label
        predict_test = predict(net.trained_net, input_label{1}, 'ExecutionEnvironment', 'gpu'); % Predicting labels
        Lebel_pred(ii) = sum(sum(predict_test));
        Lebel_true(ii) =  sum(sum(input_label{2}));
        predict_test_bin = predict_test>0.5;
        [union_temp, intersection_temp] = dice_fn( predict_test_bin, input_label{2});
        union_val = union_val + union_temp;
        intersection_val = intersection_val + intersection_temp;
        intersection_temp_file{ii} = intersection_temp;
        union_temp_file{ii} = union_temp;
        dice_per_file(ii)=(2*intersection_temp+1)/(union_temp);
        file_names{ii} = dataset.UnderlyingDatastores{1,1}.Files{ii};
    end

    groupData = array2table(zeros(5635,4));
    groupData.Properties.VariableNames = ["file_name"; "dice"; "intersection"; "union"];
    groupData.file_name= file_names'; groupData.dice= dice_per_file';
    groupData.intersection = intersection_temp_file'; groupData.union = union_temp_file';

    for ii=1:size(groupData,1)
        if ((groupData.dice(ii)<dice_thresh) && (groupData.union{ii}>union_thresh)) 
            source = groupData.file_name{ii,1};
            destination = '../data/train_features/train_satellite_removed';
            if ~exist(destination, 'dir')
            mkdir(destination)
            end
            movefile(source, destination)

            % Move labels
            outputFilenameParts =source;
            label_name = strcat(['../data/train_labels/train_kelp/', outputFilenameParts(end-21:end-14),'_kelp', '.tif']);
            source2 = label_name;
            destination2 = '../data/train_labels/train_kelp_removed/';
            if ~exist(destination2, 'dir')
                mkdir(destination2)
            end
            movefile(source2, destination2)
        end

    end

end

end

function return_images_func(return_images)
% Return images to main image folder after each iteration
if return_images && size(dir('../data/train_features/train_satellite_removed/*.tif'), 1)
    movefile('../data/train_features/train_satellite_removed/*.tif', '../data/train_features/train_satellite/');
    movefile('../data/train_labels/train_kelp_removed/*.tif', '../data/train_labels/train_kelp/');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [imdsTrain, imdsVal, outputTrain, outputVal] = partitionData(imds,pxds, i_rand, partitiontype, last)
% Partition data to training and validation sets
    
if partitiontype == 1

    numFiles = numel(imds.Files);
    if i_rand>=0
        rng(i_rand); 
        shuffledIndices = randperm(numFiles);
    else
        shuffledIndices = 1:(numFiles);
    end


    if last
        % Last 10%  for validation
        % Use 90% of the images for training
        numTrain = round(0.90 * numFiles);
        trainingIdx = shuffledIndices(1:numTrain);

  
        valIdx = shuffledIndices(numTrain+1:end); 
    else
        % First 10% for validation

        numVal = round(0.1 * numFiles);
        valIdx = shuffledIndices(1:numVal);
        % Using 90% of the images for training
        trainingIdx = shuffledIndices(numVal+1:end);
    end

elseif partitiontype == 2 % kfolds
    nn=i_rand; % Divide the files, nn: number of the folds 
    numFiles = numel(imds.Files);
    shuffledIndices = 1:numFiles;
    st_val = (nn-1)*round((1-0.9037) * numFiles)+1;
    end_val = (nn)*round((1-0.9037) * numFiles);
    valIdx = shuffledIndices(st_val:end_val);
    trainingIdx = setdiff(shuffledIndices,st_val:end_val,'stable');

elseif partitiontype == 3 % For trainedNetwork_unet_rand_8
    rng(i_rand);
    numFiles = numel(imds.Files);

    shuffledIndices = randperm(numFiles);
    numTrain = round(0.9037 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);
    valIdx = shuffledIndices(numTrain+1:end); 

elseif partitiontype == 4 % For trainedNetwork_unet_rand_15-25 
    rng(i_rand); 
    numFiles = numel(imds.Files);
    shuffledIndices_constant = 1:numFiles;
    shuffledIndices = randperm(numFiles);
    numTrain = round(0.9037 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);
    valIdx = shuffledIndices_constant(numTrain+1:end); 

elseif partitiontype == 5 % For trainedNetwork_unet_rand_52 
    rng(i_rand); 
    numFiles = numel(imds.Files);
    shuffledIndices_constant = 1:(numFiles);
    shuffledIndices = randperm(numFiles-(round((1-0.9037) * numFiles)));
    numTrain = round(0.9037 * numFiles);
    trainingIdx = shuffledIndices(1:round(numTrain*0.900684932));
    valIdx = shuffledIndices_constant(numTrain+1:end);

end

% Creating image datastores for training
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);


imdsTrain = imageDatastore(trainingImages, 'ReadFcn', @(filename)readTrainingSatelliteData(filename),...
    'LabelSource', 'foldernames');
imdsVal = imageDatastore(valImages, 'ReadFcn', @(filename)readTrainingSatelliteData(filename),...
    'LabelSource', 'foldernames');


outputTrain = pxds.Files(trainingIdx);
outputVal = pxds.Files(valIdx);

outputTrain = imageDatastore(outputTrain, 'ReadFcn', @(filename)readlabel(filename),...
    'LabelSource', 'foldernames'); 

outputVal = imageDatastore(outputVal, 'ReadFcn', @(filename)readlabel(filename),...
    'LabelSource', 'foldernames');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = readTrainingSatelliteData(outputFilename)
% Reading training data
    fullfilename = outputFilename;
    im = imread(fullfilename);
    im = double(im);
    
    output = im;

end


function label_output = readlabel(outputFilename)
% Reading labels
    outputFilenameParts = outputFilename;
    correspondingLabels= dir(['../data/train_labels/train_kelp/', outputFilenameParts(end-21:end-14),'_kelp', '.tif']);

    label = imread(strcat(correspondingLabels.folder,'/' ,correspondingLabels.name));
    label = double(label);
    label_output = label;

end


function [union, intersection] = dice_fn(pred, traget)
% Dice coefficient  
    A= reshape(pred,1,[]);
    B = reshape(traget,1,[]);
    intersection = sum(A .* B);
    union = sum(A) + sum(B);
end



function output = readTrainingSatelliteData2(outputFilename) 
% Reading training data
    fullfilename = outputFilename;
    im = imread(fullfilename);
    im = double(im);
    im(im>43636 ) = 43636 ;
    im(im<0 ) = 0 ;
    output = im;

end

