clc;clear;close all;

% 
nx = 100;

%dens_smooth_dir = '/storage/filament/result/cluster_3d/40Mpc/density_smoothing_plot_5_v2/box02_add/';
%label_dir = '/storage/filament/result/cluster_3d/40Mpc/label/box02_add/';
dens_smooth_dir = '/storage/filament/result/cluster_3d/40Mpc/density_log_plot/box02_add/';
boxname = 'box02sub02C03';

dens_smooth_3d = zeros(nx,nx,nx-4);
%label_3d = zeros(nx-4,nx-4,nx-4);


for i=0:nx-5
    dens_smooth_3d(:,:,i+1) = flipud(rgb2gray(imread([dens_smooth_dir boxname '/' num2str(i) '.png'])));
    %label_3d(:,:,i+1) = flipud(rgb2gray(imread([label_dir boxname '/' num2str(i) '.png'])));
    
end

test_mat = zeros(100,100,95);

for i=1:nx
    for j=1:nx
        for k=1:nx-4
            if (dens_smooth_3d(i,j,k) > 80)
                test_mat(i,j,k) = 1;
            else
                test_mat(i,j,k) = 0;
            end
        end
    end
end




CC = bwconncomp(test_mat,26);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
test_mat(CC.PixelIdxList{idx}) = 2;

% 
% numPixels = cellfun(@numel,CC.PixelIdxList);
% [biggest,idx] = max(numPixels);
% test_mat(CC.PixelIdxList{idx}) = 2;

volumeViewer