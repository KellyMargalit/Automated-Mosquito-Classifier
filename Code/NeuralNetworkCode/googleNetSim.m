clc;
close all;

%% Call googlenet
net = googlenet;

%% Create a Image Data Store
rootFolder = 'DatasetPupae';
categories = {'Female','Male', 'Unknown'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%% The first element of the Layers property of the network is the image input layer. 
% For a GoogLeNet network, this layer requires input images of size 224-by-224-by-3, 
% where 3 is the number of color channels. 
% Other networks can require input images with different sizes.
inputSize = net.Layers(1).InputSize;
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9);

numClasses = numel(categories(imdsTrain.Labels));

%% Replace final layers
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% Find the names of the two layers to replace. You can do this manually or 
% you can use the supporting function findLayersToReplace to find these layers automatically.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(3, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,3, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%% Train network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 5;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

%% Test network
rootFolder = 'GrayRGBTest';
testDS = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testDS.ReadFcn = @readFunctionTrainGoogle;

[labels,err_test] = classify(net, testDS);

confMat = confusionmat(testDS.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

% [YPred,probs] = classify(net,augimdsValidation);
% accuracy = mean(YPred == imdsValidation.Labels)

% displaay four random images
idx = randperm(numel(testDS.Files),10);
figure
for i = 1:10
    subplot(2,5,i)
    I = readimage(testDS,idx(i));
    imshow(I)
    label = labels(idx(i));
    title(string(label) + ", " + num2str(100*max(err_test(idx(i),:)),3) + "%");
end