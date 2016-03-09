% Subroutine - After neurons have converged, figures out with digit average
% they converged to. Stores the decisions in a reults vector. 

results = []; 

for resultsItr =netNums %netNums
    output = neurons(:,1,resultsItr);
    digits = combVec(resultsItr,:);

    [M,I] = min([sum(abs(output - DiscreteImgs(:,digits(1))));
        sum(abs(output - DiscreteImgs(:,digits(2))));
        sum(abs(-1*(output) - DiscreteImgs(:,digits(1))));
        sum(abs(-1*(output) - DiscreteImgs(:,digits(2))))]);
    results = [results,...
        (digits(1)*(I == 1 || I == 3) +...
        digits(2)*(I == 2 || I == 4))]; 
end