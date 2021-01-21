%% test for window os 
clc; clear;
addpath('C:\Users\hsw63\OneDrive\바탕 화면\git\Filament_Project\working\npy-matlab-master\npy-matlab') 

%%
cluster_num = '1.npy';
array_size = 98;

%dens = readNPY(['C:\Users\hsw63\OneDrive\바탕 화면\git\Filament_Project\pyramid\' cluster_num]);
dens = readNPY(['C:\Users\hsw63\OneDrive\바탕 화면\git\Filament_Project\save\signature\filament\' cluster_num]);
dens = dens(1:array_size ,1:array_size ,1:array_size );



transparency = zeros(256,1);

for i = 1:256
    if i < 128
        transparency(i,1) = 0;
    else 
        transparency(i,1) = i/256;
    end
end

figure
%volshow(dens,'Background',[0.3,0.3,0.3],'CameraPosition',[-2,0,5],'CameraViewAngle',20,'Alphamap',transparency)