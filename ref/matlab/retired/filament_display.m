%%
clear;clc;

box_size = '200Mpc';
res = '1024'
box_num = '1';

num = '1'

index = 201;
% 
% if res == 1024
%     index = 201;
% end
% 
% if res == 512
%     index = 101;
% end
    

filament_path = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num  '/filament/'];
sorted_filament_path = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num  '/filament_info/coords/'];
post_filament_path = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num  '/filament_post_2/filament_info/coords/'];
dens_path = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num '/dens_img/'];
label_path_bw = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num '/label/bw/'];
label_path_major = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num '/label/major/'];

dens = zeros(index,index,index);
filament = zeros(index,index,index);
post= zeros(index,index,index);
label = zeros(index,index,index);
sort_filament = zeros(index,index,index);
post_filament = zeros(index,index,index);



for i=0:index-1
    dens(i+1,:,:) = rot90(rgb2gray(imread([dens_path  num2str(i)])),4);
end



for i=1:index
    for j=1:index
        for k=1:index
            
            if dens(i,j,k) > 64
                label(i,j,k) = 1;
            end
            
        end
    end
end


CC = bwconncomp(label,26);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
post(CC.PixelIdxList{idx}) = 2;


for i = 1:index
    for j =1:index
        for k =1:index
            if post(i,j,k) == 2
                post(i,j,k) = 1;
            else
                post(i,j,k) = 0;
            end
        end
    end
end


post= double(bwmorph3(post,'clean'));
post_tmp = double(bwskel(logical(post),'MinBranchLength',10));



struc = dir(filament_path);
num_filament = struc(~ismember({struc.name},{'.','..'}));


for i = 1:length(num_filament)


    coords = load( [filament_path '/' num2str(i)],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = i;
    end
        
end


struc = dir(post_filament_path);
num_filament = struc(~ismember({struc.name},{'.','..'}));

for i = 1:length(num_filament)
%for i = 3:3

    coords = load( [post_filament_path '/' num2str(i)],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        post_filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = i;
    end
        
end



struc = dir(sorted_filament_path);
num_filament = struc(~ismember({struc.name},{'.','..'}));


for i = 1:length(num_filament)

    coords = load( [sorted_filament_path '/' num2str(i)],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        sort_filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = i;
    end
        
end



bw_id = fopen([label_path_bw 'bw'],'r');
bw = fread(bw_id,'integer*4');
bw = reshape(bw,[index,index,index]);


major_id = fopen([label_path_major 'major'],'r');
major = fread(major_id,'integer*4');
major = reshape(major,[index,index,index]);


volumeViewer
%%