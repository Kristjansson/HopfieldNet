% Subroutine - For a set image and pre-initialized nets, updates the
% neurons for a set amount of iterations, aftet which they have effectively
% converged. 

for netNum = netNums
    neurons(:,1,netNum) = image;
end

for iterations = 1:itrCount
    neuronNum = ceil(400*rand());
    for netNum = netNums
        neurons(neuronNum, 1, netNum) = biasFunc(weights(:, neuronNum, netNum)'*neurons(:,1,netNum));
    end
end