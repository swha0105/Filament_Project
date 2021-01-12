clc; clear;
addpath('/storage/Codes/git/npy-matlab/npy-matlab') 

box_size = '300Mpc'
box_num = '1'

path = ['/storage/filament/works_v7/300Mpc_1/'];

struc = dir([path 'pyramid/xray/gaussian/2']);
num_cluster = struc(~ismember({struc.name},{'.','..'}));
array_size =300;

threshold_list = load([path  'label/threshold_list']);

for num = 1:length(num_cluster)
    %num = 9
    %num = 36
    %cluster_num = '36.npy'
    cluster_num = num_cluster(num).name
    
    threshold = threshold_list(find(threshold_list(:,1)==num),:);
    threshold = threshold(2);
    

    xray = readNPY([path 'cluster_box/xray/' cluster_num]);

    label_post_tmp= int32(readNPY([path  'label/upsampling/' cluster_num]));
    label_post_tmp = int32(bwmorph3(label_post_tmp,'majority'));
    
    label_spreading_conn = zeros(array_size,array_size,array_size);
    label_post = zeros(array_size,array_size,array_size);

    CC = bwconncomp(label_post_tmp,6);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    b = sort(numPixels);

    for order = 1:length(b)
        val = b(length(b)-order+1);
        idx = find(numPixels == val);
        idx = idx(1);
        coords_index = CC.PixelIdxList{idx};
        
        for i = 1:length(coords_index)
            if coords_index(i) == 150 + 300*150 + 300*300*150
                correct_order = order;
                break
                
            end
        end
    
    label_spreading_conn(CC.PixelIdxList{idx}) = order;
    index(order) = {CC.PixelIdxList{idx}};
    post_process_label(order) = {label_spreading_conn==order};

    end
                
    label_post_2 = (zeros(array_size,array_size,array_size));
    label_post_2 = (int32(post_process_label{correct_order}));

    label_post_3 = (int32(post_process_label{correct_order}));
    
   
    for i = 6:300-5
        for j = 6:300-5
            for k = 6:300-5
                    
                %if  length( find(label_post_2(i-5:i+5,j-5:j+5,k-5:k+5) == 1) ) > 512 && label_post_2(i,j,k) == 0
                
                if  length( find(label_post_2(i-3:i+3,j-3:j+3,k-3:k+3) == 1)) > 108 && label_post_2(i,j,k) == 0
                     %&& mean(mean(mean(xray(i-1:i+1,j-1:j+1,k-1:k+1)))) > threshold
                    label_post_3(i,j,k) = 1;
                end
 
               
                
                 if i < 15 || i > 300-15 || j < 35 || j > 300-15 || k < 15 || k > 300-15
                     label_post_3(i,j,k) = 0;
                 end
 
            end
        end
    end
    
    n = 1;

    
    while n < 10
    
    label_post_4 = int32(bwmorph3(logical(label_post_3),'fill')); %3Mpc
    
    if isequal(label_post_4,label_post_3) == 1
        break
    end
    label_post_3 = label_post_4;
    n = n+1;
    
    end
    
    
    skeleton = int32(bwskel(logical(label_post_3) )); %5Mpc    
    
        n = 1;
    while n < 10
    
    skeleton_1 = int32(bwskel(logical(skeleton),'MinBranchLength',30 )); %3Mpc
    
    if isequal(skeleton_1,skeleton) == 1
        break
    end
    skeleton = skeleton_1;
    n = n+1;
    
    end
    

    endpoint = int32(bwmorph3(logical(skeleton) ,'endpoints'));
    output = single(skeleton +endpoint);    

    
     writeNPY(label_post_3 , [path 'label/upsampling_post/' cluster_num])
     writeNPY(output, [path 'label/skeleton/' cluster_num])
end

%%