%% Initialize data
DEBUG = 1 && exist('weights');
close all; 

if DEBUG
    clearvars -except weights DEBUG
else
    clearvars -except DEBUG;
end

load('ImgAvgs.mat', 'ImgAvgs');
threshold = .4;
DiscreteImgs = ImgAvgs > threshold;

disp('Data Loaded...')
%% Create A FlatNet
LookupTables; %generates tables and vectors for going back and forth between
              %neural net indexes and digit combos
% if DEBUG
%     disp('DEBUG MODE: SKIPPED NET INITIALIZATION')
% else
    FlatNet; %This gives us 'neurons' and 'weights' and trains them with ImgAvgs
% end
disp('Net Initialized...');
%% LoadTestingData
testDataSize = 100;
[Imgs_Test, Labels_Test] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-labels.idx1-ubyte',testDataSize, 0);
disp('Test Data Loaded...')
%% Compares input digits to output
close all;
itrCount = 6000;
testDataSize = 20;
netNum = 5;
% shuffled = randperm(size(Imgs_Test,3));
% Imgs_Test = Imgs_Test(:, :,shuffled);
disp(shuffled(1:10));
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
imageNum = 100;
netNum = 39;

imageVector = reshape(Imgs_Test(:,:,imageNum) > threshold,[400 1]);
neurons(:,1,netNum) = imageVector;

for itr =1:itrCount
    if mod(itr,period) == 0
        subplot(rows,ceil(itrCount/(rows*period)),imageCount), subimage(reshape((neurons(:,1,netNum)),[20 20]));
        imageCount = imageCount + 1;
    end
    neuronNum = ceil(400*rand());
    neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
end

subplot(rows,ceil(itrCount/(rows*period)),imageCount),imshow(Imgs_Test(:,:,imageNum) > threshold);

%% Digit Detector Using Bracket Method

correct = 0;
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
    disp(sprintf('Results in from first round for image: %d', imageItr));
    disp(sprintf('\t\t[%d %d %d %d %d]', results));
    
    %Second Bracket Level (01)vs(23) and (45)vs(67) with (89) getting a by.
    netNum1 = 2*((results(1) == 1) && (results(2) == 3)) + ...
        3*((results(1) == 1) && (results(2) == 4)) + ...
        10*((results(1) == 2) && (results(2) == 3)) + ...
        11*((results(1) == 2) && (results(2) == 4));
    netNum2 = 32*((results(3) == 5) && (results(4) == 7)) + ...
        33*((results(3) == 5) && (results(4) == 8)) + ...
        36*((results(3) == 6) && (results(4) == 7)) + ...
        37*((results(3) == 6) && (results(4) == 8));
    neurons(:,1,netNum1) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    neurons(:,1,netNum2) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    for iterations = 1:itrCount
        neuronNum = ceil(400*rand());
        neurons(neuronNum, 1, netNum1) = biasFunc(weights(:, neuronNum, netNum1)'*neurons(:,1,netNum1));
        neurons(neuronNum, 1, netNum2) = biasFunc(weights(:, neuronNum, netNum2)'*neurons(:,1,netNum2));
    end
    got_A_by = results(5);
    %Determine Winners
%     results = zeros(1,2); 
    
    results = [];
    for resultsItr =[netNum1 netNum2]
        output = neurons(:,1,resultsItr);
        digits = combVec(resultsItr,:);
        [M,I] = min([sum(abs(output - DiscreteImgs(:,digits(1))));
            sum(abs(output - DiscreteImgs(:,digits(2))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(1))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(2))))]);
        results = [results,...
            (digits(1)*(I == 1 || I == 3) +...
            digits(2)*(I == 2 || I == 4))]; 
    end
    disp(sprintf('Results in from second round for image: %d', imageItr));
    disp(sprintf('\t\t[%d %d] with %d on a by', results, got_A_by));
    
    %Third Bracket Level (457)vs(89) with (0123) getting a by.
    netNum = combTable(results(2), got_A_by);
    neurons(:,1,netNum) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    for iterations = 1:itrCount
        neuronNum = ceil(400*rand());
        neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
    end 
    
    %Determine Winners
    got_A_by = results(1);
    
    digits = combVec(netNum,:);
    [M,I] = min([sum(abs(output - DiscreteImgs(:,digits(1))));
            sum(abs(output - DiscreteImgs(:,digits(2))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(1))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(2))))]);
    results = digits(1)*(I == 1 || I == 3) + digits(2)*(I == 2 || I == 4);
    
    disp(sprintf('Results in from third round for image: %d', imageItr));
    disp(sprintf('\t\t%d with %d on a by', results, got_A_by));
        
    %Fourth Bracket Level (0123)vs(45789)
    netNum = combTable(got_A_by, results);
    neurons(:,1,netNum) = reshape(Imgs_Test(:,:,imageItr) > threshold,[400 1]);
    for iterations = 1:itrCount
        neuronNum = ceil(400*rand());
        neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
    end 
    
    %Determine Winners
    got_A_by = results(1);
    
    digits = combVec(netNum,:);
    [M,I] = min([sum(abs(output - DiscreteImgs(:,digits(1))));
            sum(abs(output - DiscreteImgs(:,digits(2))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(1))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(2))))]);
    results = digits(1)*(I == 1 || I == 3) + digits(2)*(I == 2 || I == 4);
    
    disp(sprintf('Final Result for image: %d', imageItr));
    disp(sprintf('\t\t%d', results));
    if Labels_Test(imageItr) == (results - 1)
        disp('Evaluated Correctly!');
        correct = correct + 1;
    else
        disp('Evaluated Incorrectly!');
        disp(sprintf('Correct Answer: %d', Labels_Test(imageItr) + 1));
    end
end

disp(sprintf('Accuracy: %d%%', 100*correct/testDataSize));

























