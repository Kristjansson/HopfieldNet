%% Initialize data
clearvars -except weights 

load('ImgAvgs.mat', 'ImgAvgs');
threshold = .4;
DiscreteImgs = (ImgAvgs > threshold) - (ImgAvgs <= threshold);

disp('Data Loaded...')
%% Create A "Flat" Hopfield Net
LookupTables; %generates tables and vectors for going back and forth between
              %neural net indexes and digit combos

FlatNet; %This gives us 'neurons' and 'weights' and trains them with DiscreteImgs
disp('Net Initialized...');
%% LoadTestingData
testDataSize = 100;
[Imgs_Test, Labels_Test] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\t10k-labels.idx1-ubyte',testDataSize, 0);
Imgs_Test = reshape(Imgs_Test, [400, 1, testDataSize]);
DiscreteImgs_Test = (Imgs_Test > threshold) - (Imgs_Test <= threshold);
disp('Test Data Loaded...')
%% Digit Detector Using Bracket Method

correct = 0;
itrCount = 2000;
for imageItr=1:testDataSize
    disp(sprintf('Results for image: %d', imageItr));
    
    %First Bracket Level    
    image = DiscreteImgs_Test(:,1,imageItr);
    netNums = [1 18 31 40 45];
    processInput;
    determineResults;

    disp(sprintf('\t\t1st Round: [%d %d %d %d %d]', results));
    
    %Second Bracket Level (01)vs(23) and (45)vs(67) with (89) getting a by.
    netNums = [combTable(results(1), results(2))...
        combTable(results(3), results(4))];
    processInput;
    got_A_by = results(5);
    determineResults;
    
    disp(sprintf('\t\t2nd Round: [%d %d] with %d on a by', results, got_A_by));
    
    %Third Bracket Level (457)vs(89) with (0123) getting a by.
    netNums = [combTable(results(2), got_A_by)];
    processInput;
    got_A_by = results(1);
    determineResults;
    
    disp(sprintf('\t\t3rd Round: [%d] with %d on a by', results, got_A_by));
        
    %Fourth Bracket Level (0123)vs(45789)
    netNums = [combTable(got_A_by, results)];
    processInput;
    determineResults;
    
    disp(sprintf('\t\t4th Round:%d', results));
    if Labels_Test(imageItr) == (results - 1)
        disp('Correct!');
        correct = correct + 1;
    else
        disp(sprintf('Incorrect. \t Answer: %d', Labels_Test(imageItr) + 1));
    end
    disp('----------------------------------------');
end

disp(sprintf('Accuracy: %d%%', 100*correct/testDataSize));
