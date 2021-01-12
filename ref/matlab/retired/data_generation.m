clc;clear;

dens=zeros(101,101,101);
label=zeros(101,101,101);
filament=zeros(101,101,101);


box_list = {['box01'],['box02']};
subbox_list = {['subbox01'],['subbox02'],['subbox03'],['subbox04'],['subbox05'],['subbox06'],['subbox07'],['subbox08']};


ref_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/';

for i = 1:2
    box_name = box_list{i};
    for j = 1:8
        subbox_name = subbox_list{j};

        path = [ref_path box_name '/' subbox_name '/dens/'];
        
        struc = dir(path);
        struc = struc(~ismember({struc.name},{'.','..'}));

        for k = 1:length(struc)
            peak_num = k-1;

            dens_path = [path num2str(peak_num) '/'];


            for i=0:100
                dens(:,:,i+1) = flipud(rgb2gray(imread([dens_path '/' num2str(i) '.png'])));
            end
            
            filament = find_connected_comp(dens);
            
            save_path = [ref_path box_name '/' subbox_name '/label/'];
            if ~exist(save_path, 'dir')
               mkdir(save_path)
            end
            
            fileID = fopen([save_path num2str(peak_num)],'w');
            fwrite(fileID,filament,'integer*4');
            fclose(fileID);

        end
    end
end
        
    
%% add_box
clc;clear;

dens=zeros(101,101,101);
label=zeros(101,101,101);
filament=zeros(101,101,101);

ref_path = '/storage/filament/result/cluster_3d/40Mpc/density_temp/';
box_list = {['box01_add'],['box02_add']};

for j = 1:2
    path = [ref_path box_list{j}];

    struc = dir(path);
    struc = struc(~ismember({struc.name},{'.','..'}));

    for i = 1:length(struc)
        subbox_name = struc(i).name;
        
        save_path = [path '/' subbox_name '/label/'];
        dens_path = [path '/' subbox_name '/dens/'];

        if ~exist(save_path, 'dir')
           mkdir(save_path)
        end    

        for i=0:100
            dens(:,:,i+1) = flipud(rgb2gray(imread([dens_path num2str(i) '.png'])));
        end

        filament = find_connected_comp(dens);

        fileID = fopen([save_path '0'] ,'w');
        fwrite(fileID,filament,'integer*4');
        fclose(fileID);        

    end
end
%subbox_list = struc(~ismember({struc.name},{'.','..'}));


%% 연결선 찾기

function filament = find_connected_comp(dens)

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

    
    CC = bwconncomp(label,6);

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


label= double(bwmorph3(label,'clean'));

label= double(bwmorph3(label,'clean'));
filament = double(bwskel(logical(label),'MinBranchLength',1));

end

