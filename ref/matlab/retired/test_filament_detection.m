clc;clear;

nx = 101 ;
box_name = 'box02';
subbox_name = 'subbox02';

peak_name = '0';

filament_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name '/' subbox_name '/filament/' peak_name];
label_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name '/' subbox_name '/label/26/' peak_name];
dens_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name '/' subbox_name '/dens/' peak_name];

dens = zeros(nx,nx,nx);
label = zeros(nx,nx,nx);
filament = zeros(nx,nx,nx);
xray = zeros(nx,nx,nx);

for i=0:nx-1
    dens(:,:,i+1) = flipud(rgb2gray(imread([dens_path '/' num2str(i) '.png'])));
end


for i = 1:101
    for j =1:101
        for k =1:101
            if dens(i,j,k) < 64
                label(i,j,k) = 0;
            else
                label(i,j,k) = 1;
            end
        end
    end
end


CC = bwconncomp(label,26);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
label(CC.PixelIdxList{idx}) = 2;


for i = 1:101
    for j =1:101
        for k =1:101
            if label(i,j,k) == 2
                label(i,j,k) = 1;
            else
                label(i,j,k) = 0;
            end
        end
    end
end


label = double(bwmorph3(label,'clean'));
label = double(bwskel(logical(label),'MinBranchLength',8));


struc = dir(filament_path);
num_filament = struc(~ismember({struc.name},{'.','..'}));


for i = 1:length(num_filament)
    coords = load( [filament_path '/' num2str(i-1)]);
    
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        filament( coords(j,3)+1,coords(j,2)+1,coords(j,1)+1) = i;
    end    
    
end

%filament = filament+label;

xray_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name '/' subbox_name '/xray/' peak_name];
xray_coords = load([xray_path '/peak_coords']);


for i = 2:length(xray_coords)
    xray(xray_coords(i,1)+1,xray_coords(i,2)+1,xray_coords(i,3)+1 ) = 1;
end
xray = flipud(xray);



xray_group = load([xray_path '/xray_group']);
xray_group = reshape(xray_group,[101,101,101]);
xray_group = permute(xray_group,[3 2 1]);

xray_group = flipud(xray_group);
volumeViewer

% 52    74    89
% 88    50    76
% 100    10    17
% 44    65    31