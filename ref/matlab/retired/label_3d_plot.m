clc;clear;

nx = 101 ;

dens_dir = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box02_add/box02sub01C15/dens';
label_dir = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box02_add/box02sub01C15/label/';
%predic_dir = '/storage/filament/result/cluster_3d/40Mpc/models/0122/predict_result';

img_3d = zeros(nx,nx,nx);

%predic_3d = zeros(nx,nx,nx);

for i=0:nx-1
    img_3d(:,:,i+1) = flipud(rgb2gray(imread([dens_dir '/' num2str(i) '.png'])));

end


fileID = fopen([label_dir '0'] ,'r');
label = fread(fileID,'integer*4');
fclose(fileID);        

volumeViewer