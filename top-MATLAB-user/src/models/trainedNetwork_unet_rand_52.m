function net=trainedNetwork_unet_rand_52()

net = dlnetwork;
% Add Layer Branches
% Add branches to the dlnetwork. Each branch is a linear array of layers.

tempNet = [
    imageInputLayer([350 350 7],"Name","encoderImageInputLayer","Normalization","zscore")
    resize2dLayer("Name","layer","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[352 352])
    convolution2dLayer([3 3],16,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],16,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
net = addLayers(net,tempNet);

tempNet = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],16,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
net = addLayers(net,tempNet);

tempNet = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],32,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
net = addLayers(net,tempNet);

tempNet = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Encoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-2")
    dropoutLayer(0.05,"Name","Encoder-Stage-4-DropOut")];
net = addLayers(net,tempNet);

tempNet = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-4-MaxPool","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","LatentNetwork-Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","LatentNetwork-Bridge-ReLU-1")
    convolution2dLayer([3 3],128,"Name","LatentNetwork-Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","LatentNetwork-Bridge-ReLU-2")
    dropoutLayer(0.05,"Name","LatentNetwork-Bridge-DropOut")
    transposedConv2dLayer([2 2],64,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-UpReLU")];
net = addLayers(net,tempNet);

tempNet = crop2dLayer("centercrop","Name","encoderDecoderSkipConnectionCrop4");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","encoderDecoderSkipConnectionFeatureMerge4")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],64,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv2dLayer([2 2],32,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU")];
net = addLayers(net,tempNet);

tempNet = crop2dLayer("centercrop","Name","encoderDecoderSkipConnectionCrop3");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","encoderDecoderSkipConnectionFeatureMerge3")
    convolution2dLayer([3 3],32,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],32,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv2dLayer([2 2],16,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-UpReLU")];
net = addLayers(net,tempNet);

tempNet = crop2dLayer("centercrop","Name","encoderDecoderSkipConnectionCrop2");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","encoderDecoderSkipConnectionFeatureMerge2")
    convolution2dLayer([3 3],16,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],16,"Name","Decoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    transposedConv2dLayer([2 2],8,"Name","Decoder-Stage-4-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-UpReLU")];
net = addLayers(net,tempNet);

tempNet = crop2dLayer("centercrop","Name","encoderDecoderSkipConnectionCrop1");
net = addLayers(net,tempNet);

tempNet = [
    concatenationLayer(3,2,"Name","encoderDecoderSkipConnectionFeatureMerge1")
    convolution2dLayer([3 3],8,"Name","Decoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],8,"Name","Decoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-2")
    convolution2dLayer([1 1],1,"Name","encoderDecoderFinalConvLayer")
    resize2dLayer("Name","layer_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[350 350])
    sigmoidLayer("Name","NetworksoftmaxLayer-Layer")];
net = addLayers(net,tempNet);

% clean up helper variable
clear tempNet;
% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

net = connectLayers(net,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
net = connectLayers(net,"Encoder-Stage-1-ReLU-2","encoderDecoderSkipConnectionCrop1/in");
net = connectLayers(net,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
net = connectLayers(net,"Encoder-Stage-2-ReLU-2","encoderDecoderSkipConnectionCrop2/in");
net = connectLayers(net,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-MaxPool");
net = connectLayers(net,"Encoder-Stage-3-ReLU-2","encoderDecoderSkipConnectionCrop3/in");
net = connectLayers(net,"Encoder-Stage-4-DropOut","Encoder-Stage-4-MaxPool");
net = connectLayers(net,"Encoder-Stage-4-DropOut","encoderDecoderSkipConnectionCrop4/in");
net = connectLayers(net,"Decoder-Stage-1-UpReLU","encoderDecoderSkipConnectionCrop4/ref");
net = connectLayers(net,"Decoder-Stage-1-UpReLU","encoderDecoderSkipConnectionFeatureMerge4/in2");
net = connectLayers(net,"encoderDecoderSkipConnectionCrop4","encoderDecoderSkipConnectionFeatureMerge4/in1");
net = connectLayers(net,"Decoder-Stage-2-UpReLU","encoderDecoderSkipConnectionCrop3/ref");
net = connectLayers(net,"Decoder-Stage-2-UpReLU","encoderDecoderSkipConnectionFeatureMerge3/in2");
net = connectLayers(net,"encoderDecoderSkipConnectionCrop3","encoderDecoderSkipConnectionFeatureMerge3/in1");
net = connectLayers(net,"Decoder-Stage-3-UpReLU","encoderDecoderSkipConnectionCrop2/ref");
net = connectLayers(net,"Decoder-Stage-3-UpReLU","encoderDecoderSkipConnectionFeatureMerge2/in2");
net = connectLayers(net,"encoderDecoderSkipConnectionCrop2","encoderDecoderSkipConnectionFeatureMerge2/in1");
net = connectLayers(net,"Decoder-Stage-4-UpReLU","encoderDecoderSkipConnectionCrop1/ref");
net = connectLayers(net,"Decoder-Stage-4-UpReLU","encoderDecoderSkipConnectionFeatureMerge1/in2");
net = connectLayers(net,"encoderDecoderSkipConnectionCrop1","encoderDecoderSkipConnectionFeatureMerge1/in1");
net = initialize(net);
