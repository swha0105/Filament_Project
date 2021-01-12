clc;clear;

dens=zeros(101,101,101);
label=zeros(101,101,101);
filament=zeros(101,101,101);


box_list = {['box01'],['box02']};
subbox_list = {['subbox01'],['subbox02'],['subbox03'],['subbox04'],['subbox05'],['subbox06'],['subbox07'],['subbox08']};
% 

%box_list = {['box01']};
%subbox_list = {['subbox01']};
 

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
            
            filament = find_connected_comp_v1(dens);
            filament = reshape(filament.',1,[]);

            save_path = [ref_path box_name '/' subbox_name '/label/bw/'];
            if ~exist(save_path, 'dir')
               mkdir(save_path)
            end
            
            fileID = fopen([save_path num2str(peak_num)],'w');
            fwrite(fileID,filament,'integer*4');
            fclose(fileID);


            filament = find_connected_comp_v2(dens);
            filament = reshape(filament.',1,[]);

            save_path = [ref_path box_name '/' subbox_name '/label/major/'];
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
        
        dens_path = [path '/' subbox_name '/dens/'];


        for i=0:100
            dens(:,:,i+1) = flipud(rgb2gray(imread([dens_path num2str(i) '.png'])));
        end
        
        filament = find_connected_comp_v1(dens);
        
        save_path = [path '/' subbox_name '/label/bw/'];
        if ~exist(save_path, 'dir')
           mkdir(save_path)
        end

        fileID = fopen([save_path '0'],'w');
        fwrite(fileID,filament,'integer*4');
        fclose(fileID);


        filament = find_connected_comp_v2(dens);

        save_path = [path '/' subbox_name '/label/major/'];
        if ~exist(save_path, 'dir')
           mkdir(save_path)
        end

        fileID = fopen([save_path '0'],'w');
        fwrite(fileID,filament,'integer*4');
        fclose(fileID);

    end
end


%%

%post processing

clear;clc
box_list = {['box01'],['box02']};
subbox_list = {['subbox01'],['subbox02'],['subbox03'],['subbox04'],['subbox05'],['subbox06'],['subbox07'],['subbox08']};
% 
filament = zeros(101,101,101);

formatSpec = '%d';
sizeA = [3 Inf];

for i = 1:2
    box_name = box_list{i};
    for j = 1:8
        subbox_name = subbox_list{j};
        tmp_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name '/' subbox_name '/filament/'];
        struc = dir(tmp_path);
        struc = struc(~ismember({struc.name},{'.','..'}));
        
        for m = 1:length(struc)
        %for m = 1:1
            peak_name = struc(m).name;
            
            
            filament_path = [tmp_path peak_name '/'];
            
            filament_dir = dir(filament_path);
            filament_num = filament_dir(~ismember({filament_dir.name},{'.','..'}));

            count = 0;
            for n = 1:length(filament_num)
            %for n = 1:1
                filament_postp = zeros(101,101,101);
                filament_tmp = zeros(101,101,101);
        
                %[filament_path num2str(n-1)]
                fileID = fopen([filament_path num2str(n-1)],'r');
                tmp = fscanf(fileID,formatSpec,sizeA);
                fclose(fileID);
            
                
                if length(tmp) < 4
                    continue
                else
                    count = count +1; 

                    for k = 1:length(tmp)
                        filament_tmp(tmp(1,k)+1,tmp(2,k)+1,tmp(3,k)+1) = 1;
                    end

                    filament_postp = double(bwskel(logical(filament_tmp),'MinBranchLength',1));

                    save_tmp_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name '/' subbox_name];

                    save_path = [save_tmp_path '/post_proc/' peak_name '/' ];
                    if ~exist(save_path, 'dir')
                       mkdir(save_path)
                    end

                    count_coords = 1;
                    A = 0;
                    for q = 1:101
                        for w = 1:101
                            for e = 1:101

                                if filament_postp(q,w,e) == 1
                                    
                                    A(count_coords,1) = q;
                                    A(count_coords,2) = w;
                                    A(count_coords,3) = e;

                                    count_coords = count_coords + 1;
                                else 
                                    continue

                                end
                            end
                        end
                    end


                    writematrix(A,[save_path num2str(count)],'Delimiter','tab');

              end

                
                
            end
              
            
        end
    end
end

%%

%post processing

clear;clc
box_list = {['box01_add'],['box02_add']};

filament = zeros(101,101,101);

formatSpec = '%d';
sizeA = [3 Inf];

for i = 1:2
    box_name = box_list{i};
    ref_path = ['/storage/filament/result/cluster_3d/40Mpc/density_temp/' box_name];


    tmp_path = dir(ref_path);
    subbox_list= tmp_path(~ismember({tmp_path.name},{'.','..'}));

    
    for j = 1:length(subbox_list)
        
        subbox_name = subbox_list(j).name;
        subbox_path = [ref_path '/' subbox_name '/'];
        
        filament_path = [subbox_path 'filament/'];
        
        
        filament_tmp = dir(filament_path);
        filament_num = filament_tmp(~ismember({filament_tmp.name},{'.','..'}));
   
        count = 0;
        for n = 1:length(filament_num)
        %for n = 1:1
            filament_postp = zeros(101,101,101);
            filament_tmp = zeros(101,101,101);

            %[filament_path num2str(n-1)]
            fileID = fopen([filament_path num2str(n-1)],'r');
            tmp = fscanf(fileID,formatSpec,sizeA);
            fclose(fileID);


            if length(tmp) < 4
                continue
            else
                count = count +1; 

                for k = 1:length(tmp)
                    filament_tmp(tmp(1,k)+1,tmp(2,k)+1,tmp(3,k)+1) = 1;
                end

                filament_postp = double(bwskel(logical(filament_tmp),'MinBranchLength',1));

                save_path = [subbox_path 'post_proc/'];

                
                if ~exist(save_path, 'dir')
                   mkdir(save_path)
                end

                count_coords = 1;
                A = 0;
                for q = 1:101
                    for w = 1:101
                        for e = 1:101

                            if filament_postp(q,w,e) == 1

                                A(count_coords,1) = q;
                                A(count_coords,2) = w;
                                A(count_coords,3) = e;

                                count_coords = count_coords + 1;
                            else 
                                continue

                            end
                        end
                    end
                end
            end


                writematrix(A,[save_path num2str(count)],'Delimiter','tab');

          

            
        end
    end
end



%% 연결선 찾기
function filament = find_connected_comp_v1(dens)

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

filament = double(bwskel(logical(label),'MinBranchLength',8));

end
%%
function filament = find_connected_comp_v2(dens)

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
label= double(bwmorph3(label,'majority'));

filament = double(bwskel(logical(label),'MinBranchLength',8));

end
