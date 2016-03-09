%% Initialize data
close all; clear all;
load('ImgAvgs.mat', 'ImgAvgs');
threshold = .4;
DiscreteImgs = ImgAvgs > threshold;

disp('Data Loaded...')
%% Create A FlatNet
FlatNet; %This gives us 'neurons' and 'weights' and trains them with ImgAvgs
disp('Net Initialized...');
%% LoadTestingData
testDataSize = 100;
[Imgs_Test, Labels_Test] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-labels.idx1-ubyte',testDataSize, 0);
disp('Test Data Loaded...')
%% Compares input digits to output
close all;
itrCount = 6000;
testDataSize = 20;
netNum = 35;
Imgs_Test = Imgs_Test(:, :,randperm(size(Imgs_Test,3)));
for itr=1:testDataSize
    image = 2*(Imgs_Test(:,:,itr) > threshold)-1;
    neurons(:,1,netNum) = reshape(image, [400 1]);
    for iterations =1:itrCount
        neuronNum = ceil(400*rand());
        neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
    end
    subplot(2, testDataSize,itr), subimage(image);
    subplot(2, testDataSize,itr + testDataSize), subimage(reshape((neurons(:,1,netNum)),[20 20]));  
%     disp('done');
end

%% Focusses on a single digit

close all;

itrCount = 2000;
period = 40;
rows = 6;
imageCount = 1;
imageVector = reshape(Imgs_Test(:,:,1) > threshold,[400 1]);

neurons(:,1,1) = imageVector;

for itr =1:itrCount
    if mod(itr,period) == 0
        subplot(rows,ceil(itrCount/(rows*period)),imageCount), subimage(reshape((neurons(:,1,1)),[20 20]));
        imageCount = imageCount + 1;
    end
    neuronNum = ceil(400*rand());
    neurons(neuronNum, 1, 1) = biasFunc(weights(:, neuronNum, 1)'*neurons(:,1,1));
end

%% Digit Detector Using Bracket Method
correct = 0;
total = 0;
itrCount = 2000;
for imageItr=1:testDataSize
    %First Bracket Level    
%     neurons(:,1,[1 18 31 40 45]) = repmat(reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]),1,1,5);
    neurons(:,1,1) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    neurons(:,1,18) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    neurons(:,1,31) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    neurons(:,1,40) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    neurons(:,1,45) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    for iterations = 1:itrCount
        neuronNum = ceil(400*rand());
        neurons(neuronNum, 1, 1) = biasFunc(weights(:, neuronNum, 1)'*neurons(:,1,1));
        neurons(neuronNum, 1, 18) = biasFunc(weights(:, neuronNum, 18)'*neurons(:,1,18));
        neurons(neuronNum, 1, 31) = biasFunc(weights(:, neuronNum, 31)'*neurons(:,1,31));
        neurons(neuronNum, 1, 40) = biasFunc(weights(:, neuronNum, 40)'*neurons(:,1,40));
        neurons(neuronNum, 1, 45) = biasFunc(weights(:, neuronNum, 45)'*neurons(:,1,45));
    end
    
    %Determine Winners
    results = zeros(1,5); 
    for resultsItr =[1,3,5,7,9;1 18 31 40 45]
        output = neurons(:,1,resultsItr(2));
        [M,I] = min([sum(abs(output - DiscreteImgs(:,resultsItr(1))));
            sum(abs(output - DiscreteImgs(:,resultsItr(1) + 1)));
            sum(abs(-1*(output-1) - DiscreteImgs(:,resultsItr(1))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,resultsItr(1) + 1)))]);
        results(ceil(resultsItr(1)/2)) = resultsItr(1)*(I == 1) + resultsItr(1)*(I==3) + (resultsItr(1) + 1)*(I==2) + (resultsItr(1) + 1)*(I==4); 
    end
    disp(sprintf('Results in from first brack for image: %d', imageItr));
    disp(sprintf('\t\t[%d %d %d %d %d]', results));
end

























