TestsToRun = [2];

if any(1==TestsToRun)
    %% Compares input digits to output
    close all;
    itrCount = 6000;
    testDataSize = 20;
    netNum = 35;
    % shuffled = randperm(size(Imgs_Test,3));
    % Imgs_Test = Imgs_Test(:, :,shuffled);
    % disp(shuffled(1:10));
    for itr=1:testDataSize
        image = DiscreteImgs_Test(:,1,itr);
        neurons(:,1,netNum) = image;
        for iterations =1:itrCount
            neuronNum = ceil(400*rand());
            neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
        end
        subplot(2, testDataSize,itr), subimage(reshape(image,[20 20]));
        subplot(2, testDataSize,itr + testDataSize), subimage(reshape((neurons(:,1,netNum)),[20 20]));  
    end
end

if any(2==TestsToRun)
    %% Focusses on a single digit

    close all;

    itrCount = 2000;
    period = 40;
    rows = 6;
    imageCount = 1;
    imageNum = 598;
    netNum = 1;

    % imageVector = reshape(Imgs_Test(:,:,imageNum) > threshold,[400 1]);
    % neurons(:,1,netNum) = imageVector;

    image = DiscreteImgs_Test(:,1,imageNum);
    neurons(:,1,netNum) = image;

    for itr =1:itrCount
        if mod(itr,period) == 0
            subplot(rows,ceil(itrCount/(rows*period)),imageCount), subimage(reshape((neurons(:,1,netNum)),[20 20]));
            imageCount = imageCount + 1;
        end
        neuronNum = ceil(400*rand());
        neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
    end

    subplot(rows,ceil(itrCount/(rows*period)),imageCount),imshow(reshape(image, [20 20]));
end

if any(3==TestsToRun)
    %% Cheating Test

    itrCount = 2000;
    correct = 0;
    for imageItr=1:testDataSize
        image = DiscreteImgs_Test(:,1,imageItr);

        num = Labels_Test(imageItr) + 1;
        otherNum = (mod(num + floor(9*rand()),10)) + 1;

        netNum = combTable(num, otherNum);

        neurons(:,1,netNum) = image;
        for iterations = 1:itrCount
            neuronNum = ceil(400*rand());
            neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
        end 
        output = neurons(:, 1, netNum);
        digits = [num, otherNum];
        [M,I] = min([sum(abs(output - DiscreteImgs(:,digits(1))));
            sum(abs(output - DiscreteImgs(:,digits(2))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(1))));
            sum(abs(-1*(output-1) - DiscreteImgs(:,digits(2))))]);
        result = digits(1)*(I == 1 || I == 3) + digits(2)*(I == 2 || I == 4);

        if result == num
            disp('correct');
            correct = correct + 1;
        else
            disp('fail');
        end
    end

    fprintf('Accuracy: %d', correct/testDataSize);

end