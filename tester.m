close all;

[imgs, labels] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-labels.idx1-ubyte',10, 0);

threshold = .5;

net = HopfieldNet(20*20);
net = net.learnPattern(reshape(imgs(:,:,1)>threshold, [400 1]));

noisy = (imgs(:,:,1)>threshold + rand(20,20)>.2) >=1 ;

net = net.input(reshape(noisy, [400 1]));

itrCount = 10000;
period = 400;
rows = 6;
imageCount = 1;
for itr =1:itrCount
    if mod(itr,period) == 0
        subplot(rows,ceil(itrCount/(rows*period)),imageCount), subimage(reshape((net.neurons),[20 20]));
        imageCount = imageCount + 1;
    end
    net = net.singleIteration(ceil(400*rand()));
end