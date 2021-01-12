clc;clear;close all;

nn = '15';

path = ['/storage/filament/result/pic/0224_v3/'];

label = zeros(80,80,80);
predict = zeros(80,80,80);
img = zeros(80,80,80);
predict_post = zeros(80,80,80);

nx = 80;

for i=0:nx-1
    label(:,:,i+1) = rgb2gray(imread([path 'label/' nn  '/' num2str(i) '.png']));
    predict(:,:,i+1) = rgb2gray(imread([path 'predict/' nn '/' num2str(i) '.png']));
    img(:,:,i+1) = rgb2gray(imread([path 'img/' nn '/' num2str(i) '.png']));
end



for i = 1:nx
    for j = 1:nx
        for k = 1:nx
            if predict(i,j,k) >= 50
                predict_post(i,j,k) = 1;
            else
                continue
            end
        end
    end
end

CC = bwconncomp(predict_post,26);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
predict_post(CC.PixelIdxList{idx}) = 2;

for i = 1:nx
    for j = 1:nx
        for k = 1:nx
            if predict_post(i,j,k) >= 50
                predict_post(i,j,k) = 1;
            else
                continue
            end
        end
    end
end

volumeViewer
volumeViewer