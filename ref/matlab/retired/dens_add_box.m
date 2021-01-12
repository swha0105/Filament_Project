clc;clear;close all;

% 
nx = 100;

dens_smooth_dir = '/storage/filament/result/cluster_3d/40Mpc/density_smoothing_plot_5_v2/box02_add/';
label_dir = '/storage/filament/result/cluster_3d/40Mpc/label/box02_add/';
boxname = 'box02sub05C06';

dens_smooth_3d = zeros(nx-4,nx-4,nx-4);
label_3d = zeros(nx-4,nx-4,nx-4);


for i=0:nx-5
    dens_smooth_3d(:,:,i+1) = flipud(rgb2gray(imread([dens_smooth_dir boxname '/' num2str(i) '.png'])));
    label_3d(:,:,i+1) = flipud(rgb2gray(imread([label_dir boxname '/' num2str(i) '.png'])));
    
end

volumeViewer 
volumeViewer