[imgs, labels] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-labels.idx1-ubyte',10, 0);

% net = HopfieldNet(20*20);
net = HopfieldNet(5);
% pattern = reshape(imgs(:,:,1), [20*20,1]);
pattern = [0 1 1 0 1];
net = net.learnPattern(pattern);
net = net.learnPattern([1 0 1 0 1]');
net = net.input([1 1 1 1 1]');
net = net.singleIteration(3);
% net = net.input(rand(400,1)>.5);

% 
% for itr=1:10
%     net = net.singleIteration(floor(20*20*rand()));
% end
% imshow(reshape(net.neurons,[20 20]));
 
