close all;
% [imgs, labels] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-labels.idx1-ubyte',60000, 0);
load('ImgAvgs.mat', 'ImgAvgs');
ImgAvgs = zeros(20*20,10);
ImgCounts = zeros(1,10);
for itr=1:60000
    num = labels(itr) + 1;
    temp = reshape(imgs(:,:,itr), [20*20,1]);
    ImgAvgs(:, num) = ImgAvgs(:, num) + temp;
    ImgCounts(1,num) = ImgCounts(1, num) + 1;
end
ImgAvgs = ImgAvgs./(ones(400,1)*ImgCounts);
save('ImgAvgs.mat', 'ImgAvgs');

OverallAvg = ImgAvgs*ones(10,1)./10;

%%

close all;
load('ImgAvgs.mat', 'ImgAvgs');


threshold = .4;
% for itr=1:10
%     subplot(2,6,itr), subimage(reshape((ImgAvgs(:,itr)>threshold),[20 20]));    
% end
for itr=1:10
    subplot(2,6,itr), subimage(reshape(((abs(OverallAvg - ImgAvgs(:,itr)))>.2),[20 20]));    
end
subplot(2,6,11), subimage(reshape(OverallAvg > threshold, [20 20])); 


%Create Net
net = HopfieldNet(20*20);

%Learn Average Patterns
% for itr=1:10
%     net = net.learnPattern((abs(OverallAvg - ImgAvgs(:,itr)))>.2);
% end
net = net.learnPattern((ImgAvgs(:,1))>threshold);
net = net.learnPattern((ImgAvgs(:,8))>threshold);
disp('done')
%%
%LoadTestingData
testDataSize = 20;
[Imgs_Test, Labels_Test] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-labels.idx1-ubyte',testDataSize, 0);

%% Compares input digits to output
close all;
itrCount = 6000;
for itr=1:testDataSize
    image = 2*(Imgs_Test(:,:,itr) > threshold) - 1;
    net = net.input(reshape(image,[400 1]));
    for iterations =1:itrCount
        net = net.singleIteration(ceil(400*rand()));
    end
    subplot(2, testDataSize,itr), subimage(image);
    subplot(2, testDataSize,itr + testDataSize), subimage(reshape((net.neurons),[20 20]));   
end

%% Focusses on a single digit

close all;

itrCount = 4000;
period = 40;
rows = 6;
imageCount = 1;
imageVector = reshape(Imgs_Test(:,:,1),[400 1]);
diff = abs(OverallAvg - imageVector);
net = net.input(imageVector > threshold);

for itr =1:itrCount
    if mod(itr,period) == 0
        subplot(rows,ceil(itrCount/(rows*period)),imageCount), subimage(reshape((net.neurons),[20 20]));
        imageCount = imageCount + 1;
    end
    net = net.singleIteration(ceil(400*rand()));
end

%%
correct = 0;
total = 0;
numberOfFullIterations = 10;

for itr=1:testDataSize
    net = net.input(2*reshape(Imgs_Test(:,:,itr)>threshold, [400 1]) - 1);
    for fullItr=1:numberOfFullIterations
        order = randperm(20*20);
        net = net.fullIteration(order);
%         disp('debug');
    end
    error = ImgAvgs - net.neurons*[1:10];
    [M, I] = min(sum(error));
    if I == Labels_Test(itr)
        correct = correct + 1;
    end
    total = total + 1;
end

disp(sprintf('Accuracy: %d', correct/total));