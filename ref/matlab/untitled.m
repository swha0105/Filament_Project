clc; clear;
addpath('/storage/Codes/git/npy-matlab/npy-matlab') 

cluster_num = '22';
ref = readNPY(['/storage/filament/works_v7/300Mpc_1/cluster_box/xray/' cluster_num '.npy']);
label = single(readNPY(['/storage/filament/works_v7/300Mpc_1/label/upsampling/' cluster_num '.npy']));
skel = single(readNPY(['/storage/filament/works_v7/300Mpc_1/label/skeleton/' cluster_num '.npy']));
dens= readNPY(['/storage/filament/works_v7/300Mpc_1/cluster_box/dens/' cluster_num '.npy']);

label_post = single(readNPY(['/storage/filament/works_v7/300Mpc_1/label/upsampling_post/' cluster_num '.npy']));
%post = single(readNPY(['/storage/filament/works_v7/300Mpc_1/label/post/' cluster_num '.npy']));
index=300;
post_filament = zeros(index,index,index);
fitting_filament = zeros(index,index,index);
test_filament = zeros(index,index,index);


struc = dir(['/storage/filament/works_v7/300Mpc_1/filament/sorted/' cluster_num '/']);
num_filament = struc(~ismember({struc.name},{'.','..'}));


for i = 1:length(num_filament)

    coords = load( ['/storage/filament/works_v7/300Mpc_1/filament/sorted/' cluster_num '/'  num_filament(i).name],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        post_filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = i;
    end
      

    coords = load( ['/storage/filament/works_v7/300Mpc_1/filament/fitting/' cluster_num '/'  num_filament(i).name],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        fitting_filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = i;
    end

    
end
post_filament = single(post_filament);
fitting_filament = single(fitting_filament);

% 
whole_filament = zeros(index,index,index);

struc = dir(['/storage/filament/works_v7/300Mpc_1/filament/whole/' cluster_num '/']);
num_filament = struc(~ismember({struc.name},{'.','..'}));




whole_filament = zeros(index,index,index);

%for i = 1:length(num_filament)
%i = 4
for n = 1:i
    %n=12
    coords = load( ['/storage/filament/works_v7/300Mpc_1/filament/whole/' cluster_num '/'  num2str(n)],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        whole_filament( coords(j,1)+1,coords(j,2)+1,coords(j,3)+1) = n;
    end
      
    
end
whole_filament = single(whole_filament);

%test_filament = ref;
% 
% 
% skel_test= int32(bwmorph3(logical(skel),'fill')); %3Mpc
% skel_test= int32(bwskel(logical(skel_test))); %3Mpc
% skel_post = skel_test;
% for i = 6:300-5
%     for j = 6:300-5
%         for k = 6:300-5
%                     
%                 %if  length( find(label_post_2(i-5:i+5,j-5:j+5,k-5:k+5) == 1) ) > 512 && label_post_2(i,j,k) == 0
%                 %if    length( find(label_post_2(i-1:i+1,j-1:j+1,k-1:k+1) == 1)) > 14 && label_post_2(i,j,k) == 0 && ...
% %                 if  length( find(label_post_2(i-2:i+2,j-2:j+2,k-2:k+2) == 1) ) > 64 && label_post_2(i,j,k) == 0 
%                 if  length( find(skel_test(i-10:i+10,j-10:j+10,k-10:k+10) == 1)) > 100 && label_post_2(i,j,k) == 0
%                      %&& mean(mean(mean(xray(i-1:i+1,j-1:j+1,k-1:k+1)))) > threshold
%                     skel_post(i,j,k) = 1;
%                 end
%  
%              
%             end
%         end
%     end
%     



