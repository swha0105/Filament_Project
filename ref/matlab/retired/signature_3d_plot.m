clear;clc;

%path = '/storage/filament/works_v3/nexus/200Mpc_1_1024/1/';
path = '/storage/filament/works_v4/data/200Mpc_1/signature/post/';
addpath('/storage/Codes/git/npy-matlab/npy-matlab')  

signature = readNPY([path '1.npy']);

%%
box_size = '200Mpc';

box_num = '1';

num = '1'


dens_path = ['/storage/filament/works_v4/data/200Mpc_1/clusters/'];

index = 401;
%%
dens = readNPY([dens_path '2.npy']);

dens = dens(1:index-3,1:index-3,1:index-3);
%%
threshold_signature = zeros(index-3,index-3,index-3);


signature_mean = sum(sum(sum(signature))) ./ sum(sum(sum(signature~=0)));


for i = 1:1:index-3
    for j = 1:1:index-3
        for k = 1:1:index-3
                           
            if signature(i,j,k) >= signature_mean 
               threshold_signature(i,j,k) = 1;
            else  
                threshold_signature(i,j,k) = 0;
            end
        end
    end
end

majority = double(bwmorph3(threshold_signature  ,'majority'));
skeleton = double(bwskel(logical(majority)));


cut_brunch = double(bwskel(logical(skeleton),'MinBranchLength',30));


writeNPY(cut_brunch , ['/storage/filament/works_v4/data/200Mpc_1/filament/skeletonized/' '2'])
 

volumeViewer

%%
filament_path = '/storage/filament/works_v4/data/200Mpc_1/filament/segmented/';
post_filament = zeros(index-3,index-3,index-3);
struc = dir([filament_path '1']);
num_filament = struc(~ismember({struc.name},{'.','..'}));

for i = 1:length(num_filament)
%for i = 3:3

    coords = load( [filament_path '1' '/' num2str(i)],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        post_filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = i;
    end
        
end