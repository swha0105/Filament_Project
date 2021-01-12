clc;clear;


box_size = '200Mpc';
res = '1024'
box_num = '1';

num = '1'

dens_path = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num '/dens_img/'];

index = 201;

dens = zeros(index,index,index);
sort_filament = zeros(index-2,index-2,index-2);


sig_path = '/storage/filament/works_v3/nexus/200Mpc_1_1024/1/';

signature = load([sig_path 'normlized']);
signature = reshape(signature ,[199,199,199]);


for i=0:index-1
    dens(i+1,:,:) = rot90(rgb2gray(imread([dens_path  num2str(i)])),4);
end

dens = dens(1:199,1:199,1:199);


sorted_filament_path = ['/storage/filament/works_v2/result_filament/' box_size '_' box_num '_' res '/' num  '/filament_info/coords/'];

struc = dir(sorted_filament_path);
num_filament = struc(~ismember({struc.name},{'.','..'}));


for i = 1:length(num_filament)

    coords = load( [sorted_filament_path '/' num2str(i)],'r');
    
    tmp = size(coords);
    num = tmp(1);
    for j = 1:num
        sort_filament( coords(j,3)+1,coords(j,2)+1,coords(j,1)+1) = i;
    end
        
end


signature_label = zeros(199,199,199);



for i = 1:199
    for j =1:199
        for k =1:199
            if signature(i,j,k) > 0.1
                signature_label(i,j,k) = 1;
            else
                signature_label(i,j,k) = 0;
            end
        end
    end
end

% 
signature_label= double(bwmorph3(signature_label,'clean'));


filament = double(bwskel(logical(signature_label),'MinBranchLength',8));
filament= double(bwmorph3(filament,'clean'));



major= double(bwmorph3(signature_label,'majority'));


major_skel = double(bwskel(logical(major),'MinBranchLength',8));
major_skel= double(bwmorph3(major_skel,'clean'));


fileID = fopen(['/storage/filament/works_v3/nexus/200Mpc_1_1024/1/' 'major'],'w');
fwrite(fileID,major_skel,'integer*4');
fclose(fileID);


volumeViewer