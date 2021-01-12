%post processing
clear;clc

filament = zeros(101,101,101);

formatSpec = '%d';
sizeA = [3 Inf];

ref_path = ['/storage/filament/works_v2/result_filament/200Mpc_1'];


tmp_path = dir(ref_path);
subbox_list= tmp_path(~ismember({tmp_path.name},{'.','..'}));


%for j = 1:length(subbox_list)
for j = 2:2

    subbox_name = subbox_list(j).name;
    subbox_path = [ref_path '/' subbox_name '/'];

    filament_path = [subbox_path 'filament/'];
       
    save_path = [subbox_path 'post_filament/'];


    if ~exist(save_path, 'dir')
       mkdir(save_path)
    else
       rmdir(save_path,'s')
       mkdir(save_path)
    end

    filament_dir = dir(filament_path);
    filament_num = filament_dir(~ismember({filament_dir.name},{'.','..'}));

    count = 0;
    filament_tmp = zeros(101,101,101);

    for n = 1:length(filament_num)
    
    
        fileID = fopen([filament_path num2str(n)],'r');
        tmp = fscanf(fileID,formatSpec,sizeA);
        fclose(fileID);


        length_filament = size(tmp);
        length_filament = length_filament(2);
        if length_filament < 8
            continue
        else
            count = count +1; 
            


            for k = 1:length_filament
                filament_tmp(tmp(1,k)+1,tmp(2,k)+1,tmp(3,k)+1) = count;
            end


         end
        
        
    end

    post_count = 0;
    A = zeros(101,101,101);
    for nn = 1:count
        idx = find(filament_tmp == nn);
        [ix,iy,iz] = ind2sub(size(filament_tmp),idx);
        
        len = size(ix);
        
        if len(1) < 10
            continue
        else
            post_count = post_count + 1;
            
            
            for iix = 1:101
                for iiy = 1:101
                    for iiz = 1:101

                        for m = 1:len
                            if(iix == ix(m) && iiy == iy(m) && iiz == iz(m))
                                A(iix,iiy,iiz) = post_count;
                            else
                                continue
                            end
                        end
                    end
                end
            end
                        
        end
    end
    
    
    
    
    
    
    
end



volumeViewer

%%

idx = find(filament_tmp == 2);
[row,col,pag] = ind2sub(size(filament_tmp),idx);



%%
clear;clc

filament = zeros(101,101,101);

formatSpec = '%d';
sizeA = [3 Inf];

ref_path = ['/storage/filament/works_v2/result_filament/200Mpc_1'];


tmp_path = dir(ref_path);
subbox_list= tmp_path(~ismember({tmp_path.name},{'.','..'}));


%for j = 1:length(subbox_list)
for j = 1:1

    subbox_name = subbox_list(j).name;
    subbox_path = [ref_path '/' subbox_name '/'];

    filament_path = [subbox_path 'filament/'];
       
    save_path = [subbox_path 'post_filament/'];


    if ~exist(save_path, 'dir')
       mkdir(save_path)
    else
       rmdir(save_path,'s')
       mkdir(save_path)
    end

    filament_dir = dir(filament_path);
    filament_num = filament_dir(~ismember({filament_dir.name},{'.','..'}));

    count = 0;
    
    for n = 1:length(filament_num)
    
    
        fileID = fopen([fiXlament_path num2str(n)],'r');
        tmp = fscanf(fileID,formatSpec,sizeA);
        fclose(fileID);


        length_filament = size(tmp);
        length_filament = length_filament(2);
        if length_filament < 8
            continue
        else
            count = count +1; 
            size(tmp)

            filament_postp = zeros(101,101,101);
            filament_tmp = zeros(101,101,101);

            for k = 1:length_filament
                filament_tmp(tmp(1,k)+1,tmp(2,k)+1,tmp(3,k)+1) = 1;
            end


            filament_postp = double(bwskel(logical(filament_tmp),'MinBranchLength',1));




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








%%
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
        