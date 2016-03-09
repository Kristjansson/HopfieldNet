%%Calculate the averages for eacb digit
[imgs, labels] = readMNIST('C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-images.idx3-ubyte','C:\Users\Joseph\Documents\AMATH 383\FinalProject\train-labels.idx1-ubyte',60000, 0);
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


%% Display the average pictures
close all;

threshold = .4;
for itr=1:10
    subplot(2,5,itr), subimage(reshape((ImgAvgs(:,itr)>threshold),[20 20])), axis off;    
end
% for itr=1:10
% %     subplot(2,6,itr), subimage(reshape(((abs(OverallAvg - ImgAvgs(:,itr)))>.2),[20 20]));    
% end
% subplot(2,6,11), subimage(reshape(OverallAvg > threshold, [20 20])); 

% print('Averages', '-djpeg');

