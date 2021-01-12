clc;clear;
% 
addpath('/storage/filament/codes/matlab/VolumetricDataExplorer/VolumetricDataExplorer/');
addpath('/storage/filament/codes/matlab/imtool3D_td-master/');

data_path = '/storage/filament/result/cluster_3d/40Mpc/density_log_plot/label_conn/box01/subbox01/0';
data_path_img = '/storage/filament/result/cluster_3d/40Mpc/density_log_plot/box01/subbox01/0';

tmp_tmp_label = zeros(96,96,96);
tmp_label = zeros(96,96,96);
tmp_tmp_img= zeros(100,100,96);
tmp_img= zeros(96,96,96);


for i=0:95
    tmp_tmp_label(:,:,i+1) = flipud((imread([data_path '/' num2str(i) '.png'])));
    tmp_tmp_img(:,:,i+1) = rgb2gray(imread([data_path_img '/' num2str(i) '.png']));
end

tmp_img = tmp_tmp_img(3:98,3:98,:);


%%

for i = 1:96
    for j =1:96
        for k =1:96
            if tmp_img(i,j,k) < 64
                tmp_label(i,j,k) = 0;
            else
                tmp_label(i,j,k) = 1;
            end
        end
    end
end

%%


CC = bwconncomp(tmp_label,26);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
tmp_label(CC.PixelIdxList{idx}) = 2;


volumeViewer

%%
% img_bwcon = bwmorph3(tmp,'majority');
% img_bwcon = double(img_bwcon);


%%
for i = 1:96
    for j =1:96
        for k =1:96
            if tmp(i,j,k)-BW3(i,j,k) == 255
                tmp_trans(i,j,k) = 1;
            else
                tmp_trans(i,j,k) = 0;
            end
        end
    end
end
%%
BW5 = tmp-BW3;
figure
%volshow(BW5);

%%

tmp_tmp = tmp;

CC = bwconncomp(tmp_tmp,8);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
tmp_tmp(CC.PixelIdxList{idx}) = 2;

volumeViewer
volumeViewer