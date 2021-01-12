clc; clear;
addpath('/storage/Codes/git/npy-matlab/npy-matlab') 

box_size = '300Mpc'
box_num = '1'
cluster_num = '1.npy'
array_size = 73;

path = ['/storage/filament/works_v4/data/' box_size '_' box_num '/DL/smoothing/'];
path_ref = ['/storage/filament/works_v4/data/' box_size '_' box_num '/DL/raw_data/'];

dens_ref = readNPY([path_ref 'den/' cluster_num]);
dens = readNPY([path  'dens/' cluster_num]);
dens = dens(1:array_size ,1:array_size ,1:array_size );

label_post= readNPY([path  'label/' cluster_num]);
label_raw = readNPY([path 'label_raw/' cluster_num]);

label_spreading= readNPY([path  'label_spreading/' cluster_num]);

label_sum = label_spreading;

for i = 1:array_size
    for j = 1:array_size
        for k = 1:array_size
            if label_raw(i,j,k) == 1
                label_sum(i,j,k) = 1;
            end
        end
    end
end



label_spreading_major = double(bwmorph3(label_sum,'majority'));

label_spreading_conn = zeros(array_size,array_size,array_size);
CC = bwconncomp(label_spreading_major,6);
numPixels = cellfun(@numel,CC.PixelIdxList);
b = sort(numPixels);

for order = 1:10
    val = b(length(b)-order+1);
    idx = find(numPixels == val);
    label_spreading_conn(CC.PixelIdxList{idx}) = order;
    index(order) = {CC.PixelIdxList{idx}};
    post_process_label(order) = {label_spreading_conn==order};
end

%volumeViewer

%%
%labelvolshow(label_post,dens,'Background',[0.3,0.3,0.3],'CameraPosition',[5,5,5],'CameraTarget',[3,3,3],'CameraViewAngle',10)
figure;
labelvolshow(label_post,dens,'Background',[0.3,0.3,0.3],'CameraPosition',[-2,0,5],'CameraViewAngle',20,'VolumeOpacity',0.2,'VolumeThreshold',0.45)
saveas(gcf,'test','epsc');
%%
figure;
labelvolshow(label_spreading_conn,dens,'Background',[0.3,0.3,0.3],'CameraPosition',[-2,0,5],'CameraViewAngle',20,'VolumeOpacity',0.2,'VolumeThreshold',0.45)
%%
transparency = zeros(256,1);
for i = 1:256
    if i < 128
        transparency(i,1) = 0;
    else 
        transparency(i,1) = i/256;
    end
end
volshow(dens,'Background',[0.3,0.3,0.3],'CameraPosition',[-2,0,5],'CameraViewAngle',20,'Alphamap',transparency)
%%


function filament_connected_array = filament_connetion(label_spreading,x,y,z)
    filament_connected_array = zeros(5,5,5);
    
    for ix = 1:5
        for iy = 1:5
            for iz = 1:5
                if label_spreading(x+ix-3,y+iy-3,z+iz-3) == 1 
                    filament_connected_array(ix,iy,iz) = 1;
                end
            end
        end
    end
                    
end


function closest_coords = find_cloest_points(indices,ref_number,other_number,array_size)
    ref_coords = indices{ref_number};
    ref_z = floor(ref_coords/array_size^2);
    ref_a = mod(ref_coords,array_size^2);
    ref_y = floor(ref_a/array_size);
    ref_x = mod(ref_a,array_size);
    
    if ref_x == 0
        ref_y = ref_y-1;
        ref_x = array_size-1;
    end

    com_coords = indices{other_number};
    com_z = floor(com_coords/array_size^2);
    com_a = mod(com_coords,array_size^2);
    com_y = floor(com_a/array_size);
    com_x = mod(com_a,array_size);
    
    if com_x == 0
        com_y = com_y-1;
        com_x = array_size-1;
    end
    
    previous_distance = 100000;
    for i = 1:length(ref_coords)
        
        for j = 1:length(com_coords)
            distance = sqrt( (ref_x(i)-com_x(j))^2 + (ref_y(i)-com_y(j))^2 + (ref_z(i)-com_z(j))^2);

            if distance < previous_distance
                min_x_1 = ref_x(i);
                min_y_1 = ref_y(i);
                min_z_1 = ref_z(i);

                min_x_2 = com_x(j);
                min_y_2 = com_y(j);
                min_z_2 = com_z(j);
                
                previous_distance = distance;
            end
        end
        
    end

        
    closest_coords = zeros(3,2);
    closest_coords(1,1) = min_x_2;
    closest_coords(2,1) = min_y_2;
    closest_coords(3,1) = min_z_2;
    closest_coords(1,2) = min_x_1;
    closest_coords(2,2) = min_y_1;
    closest_coords(3,2) = min_z_1;
    
end


function post_process_label = major_conn(label,order,array_size)

    label_spreading_major = double(bwmorph3(label,'majority'));

    label_spreading_conn = zeros(array_size,array_size,array_size);
    CC = bwconncomp(label_spreading_major,6);

    numPixels = cellfun(@numel,CC.PixelIdxList);
    b = sort(numPixels);
    val = b(length(b)-order);
    idx = find(numPixels == val);
    label_spreading_conn(CC.PixelIdxList{idx}) = 1;
    
    post_process_label = label_spreading_conn;

end
   

function post_process_label = conn_major(label,order,array_size)
    
    label_spreading_conn = zeros(array_size,array_size,array_size);
    CC = bwconncomp(label,6);

    numPixels = cellfun(@numel,CC.PixelIdxList);
    b = sort(numPixels);
    val = b(length(b)-order);
    idx = find(numPixels == val);
    label_spreading_conn(CC.PixelIdxList{idx}) = 1;
    

    post_process_label = label_spreading_conn ;
end