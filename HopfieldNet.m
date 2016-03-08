classdef HopfieldNet
    properties
        size
        weights
        neurons
    end
    methods
        function net = HopfieldNet(size)
            net.size = size;
            net.weights = zeros(size, size);
            net.neurons = zeros(size, 1);
        end
        
        function self = input(self,pattern)
            if isequal(size(pattern), size(self.neurons))
                self.neurons = pattern;            
            else
                disp('Error');
            end
        end
        
        function val = biasFunc(self,updateVal)
            val = updateVal >= 0;
        end

    %     check that an input call has been made first
        function self = singleIteration(self, neuronNum)
            self.neurons(neuronNum) = self.biasFunc(self.weights(:, neuronNum)'*self.neurons);
        end
        
        function self = fullIteration(self, order)
%             temp = randperm(self.size);
            for idx=order
                self = self.singleIteration(self, idx);
            end
        end

        function self = learnPattern (self, pattern)
            for outerIndex=[1:self.size]
                for innerIndex=[outerIndex:self.size]
                    if outerIndex ~= innerIndex
                        val = (2*pattern(outerIndex)-1)*(2*pattern(innerIndex)-1);
                        self.weights(outerIndex, innerIndex) = val + self.weights(outerIndex, innerIndex);
                        self.weights(innerIndex, outerIndex) = val + self.weights(innerIndex, outerIndex);
                    else
                        self.weights(outerIndex, innerIndex) = 0;
                    end
                end
            end
        end
    end
end