clc;clear;close all;

nn = '2';

path = ['/storage/filament/result/pic/0224/'];

label = zeros(80,80,80);
predict = zeros(80,80,80);
img = zeros(80,80,80);

nx = 80;

for i=0:nx-1
    label(:,:,i+1) = rgb2gray(imread([path 'label/' nn  '/' num2str(i) '.png']));
    predict(:,:,i+1) = rgb2gray(imread([path 'predict/' nn '/' num2str(i) '.png']));
    img(:,:,i+1) = rgb2gray(imread([path 'img/' nn '/' num2str(i) '.png']));
end

volumeViewer
volumeViewer