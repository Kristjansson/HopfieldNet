%% Parameters
pPerNet = 2; %Patterns Per Net
inputPairs = combnk([1:10], 2);
numNets = size(inputPairs,1);
nodes = 400;

%Create Flat Net (3D Array)
weights = zeros(nodes,nodes,numNets);
neurons = zeros(nodes, 1, numNets);

%% learnPatterns
for netItr=1:numNets
    patterns = DiscreteImgs(:,inputPairs(netItr, :));
    for outItr=[1:nodes] 
        for inItr=[outItr:nodes]
            if outItr ~= inItr
                val = sum((2*patterns(inItr, :)-1).*(2*patterns(outItr, :)-1));
                weights(outItr, inItr, netItr) = val;
                weights(inItr, outItr, netItr) = val;
            else
                weights(outItr, inItr, netItr) = 0;
            end
        end
    end
end
