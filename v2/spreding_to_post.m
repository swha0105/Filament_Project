clc; clear;
addpath('/storage/Codes/git/npy-matlab/npy-matlab') 

array_size = 73;

path = ['/storage/filament/works_v7/300Mpc_1/label/'];

struc = dir([path 'spreading/']);
num_cluster = struc(~ismember({struc.name},{'.','..'}));

%%
for num = 1:length(num_cluster)
    
    cluster_num = num_cluster(num).name
    %cluster_num = '6.npy'
    label_spreading = single(readNPY([path  'spreading/' cluster_num]));
   
    xray = single(readNPY(['/storage/filament/works_v7/300Mpc_1/pyramid/xray/gaussian/2/' cluster_num]));

    %label_spreading_major = double(bwmorph3(label_spreading,'majority'));
    label_spreading_major = label_spreading;
    label_spreading_conn = zeros(array_size,array_size,array_size);
    CC = bwconncomp(label_spreading_major,6);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    b = sort(numPixels);

%     for order = 1:5

%     
    for order = 1:5
        val = b(length(b)-order+1);
        idx = find(numPixels == val);
        idx = idx(1);
        coords_index = CC.PixelIdxList{idx};
        
        for i = 1:length(coords_index)
            if abs(coords_index(i) - (36 + 73*36 + 73*73*36)) < 10
                correct_order = order;
                break
                
            end
        end
    end
    

    for order = 1:5
        val = b(length(b)-order+1);
        idx = find(numPixels == val);
        label_spreading_conn(CC.PixelIdxList{idx}) = order;
        index(order) = {CC.PixelIdxList{idx}};
        post_process_label(order) = {label_spreading_conn==order};
    end
    
 
    
    label = zeros(array_size,array_size,array_size);
    label = post_process_label{correct_order};

    %label = int32(readNPY([path cluster_num ]));    

    skeleton = int32(bwskel(logical(label),'MinBranchLength',5)); %3Mpc
    skeleton_2 = int32(bwskel(logical(skeleton)));

    endpoint = int32(bwmorph3(logical(skeleton_2  ) ,'endpoints'));
    output = single(skeleton+endpoint);    

    label = int32(label) + endpoint;


%     
      writeNPY(label , [path 'post/' cluster_num])
      writeNPY(output, [path 'skeleton/' cluster_num])

  end
