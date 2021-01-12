clear;clc
addpath('/storage/filament/codes/matlab/imtool3D_td-master/');
ref_path = '/storage/filament/result/cluster_3d/40Mpc/density_log_plot/';

box_list = {'box01/','box02/'};
subbox_list = {'subbox01/','subbox02/','subbox03/','subbox04/','subbox05/','subbox06/','subbox07/','subbox08/'};

tmp = zeros(100,100,96);
pre_label = zeros(96,96,96);
for i=1:2
    box_num = box_list{i};
    box_path = [ref_path box_num];
    
    for j=1:8
        subbox_num = subbox_list{j};
        subbox_path = [box_path subbox_num];
        
        struc = dir(subbox_path);
        struc = struc(~ismember({struc.name},{'.','..'}));
        
        for k = 1:length(struc)-1
            peak_num = k-1;
            
            for i=0:95
               
            tmp(:,:,i+1) = flipud(rgb2gray(imread([subbox_path num2str(k-1) '/' num2str(i) '.png'])));
            
            end
            
            for i = 1:96
                for j = 1:96
                    for k = 1:96
                        if tmp(i,j,k) >= 64
                           pre_label(i,j,k) = 1;
                        else
                           pre_label(i,j,k) = 0;
                        end
                    end
                end
            end
            
            CC = bwconncomp(pre_label,6);
 
            numPixels = cellfun(@numel,CC.PixelIdxList);
            [biggest,idx] = max(numPixels);
            pre_label(CC.PixelIdxList{idx}) = 2;
     
            for i = 1:96
                for j = 1:96
                    for k = 1:96
                        if pre_label(i,j,k) == 2
                           pre_label(i,j,k) = 1;
                        else
                           pre_label(i,j,k) = 0;
                        end
                    end
                end
            end
            
            save_dir = [ref_path 'label_conn/' box_num subbox_num num2str(peak_num) '/'];
            mkdir(save_dir)
            
            for k = 0:95
                imwrite(pre_label(:,:,k+1),fullfile([save_dir num2str(k) '.png']))
            end
                
            
        end
        
        
    end
        
end

%%

addpath('/storage/filament/codes/matlab/imtool3D_td-master/');
ref_path = '/storage/filament/result/cluster_3d/40Mpc/density_log_plot/';

box_list = {'box01_add/','box02_add/'};


tmp = zeros(100,100,96);
pre_label = zeros(96,96,96);
for i=1:2
    box_num = box_list{i};
    box_path = [ref_path box_num];
    
    struc = dir(box_path);
    struc = struc(~ismember({struc.name},{'.','..'}));

    len = length(struc);
    for j = 1:len
        subbox_num = struc(j).name;
        subbox_path = [box_path subbox_num];
        
        for k=0:95

        tmp(:,:,k+1) = flipud(rgb2gray(imread([subbox_path '/' num2str(k) '.png'])));

        end
        
        for i = 1:96
            for j = 1:96
                for k = 1:96
                    if tmp(i,j,k) >= 64
                       pre_label(i,j,k) = 1;
                    else
                       pre_label(i,j,k) = 0;
                    end
                end
            end
        end
 
      CC = bwconncomp(pre_label,6);
 
      numPixels = cellfun(@numel,CC.PixelIdxList);
      [biggest,idx] = max(numPixels);
      pre_label(CC.PixelIdxList{idx}) = 2;
 
      for i = 1:96
          for j = 1:96
              for k = 1:96
                  if pre_label(i,j,k) == 2
                     pre_label(i,j,k) = 1;
                  else
                     pre_label(i,j,k) = 0;
                  end
              end
          end
      end
 
      save_dir = [ref_path 'label/' box_num subbox_num '/'];
      mkdir(save_dir)
 
      for k = 0:95
          imwrite(pre_label(:,:,k+1),fullfile([save_dir num2str(k) '.png']))
      end
        
    end
        

end
% 
%     for i = 1:96
%         for j = 1:96
%             for k = 1:96
%                 if tmp(i,j,k) >= 64
%                    pre_label(i,j,k) = 1;
%                 else
%                    pre_label(i,j,k) = 0;
%                 end
%             end
%         end
%     end
% 
%     CC = bwconncomp(pre_label,6);
% 
%     numPixels = cellfun(@numel,CC.PixelIdxList);
%     [biggest,idx] = max(numPixels);
%     pre_label(CC.PixelIdxList{idx}) = 2;
% 
%     for i = 1:96
%         for j = 1:96
%             for k = 1:96
%                 if pre_label(i,j,k) == 2
%                    pre_label(i,j,k) = 1;
%                 else
%                    pre_label(i,j,k) = 0;
%                 end
%             end
%         end
%     end
% 
%     save_dir = [ref_path 'label/' box_num  '/'];
%     mkdir(save_dir)
% 
%     for k = 0:95
%         imwrite(pre_label(:,:,k+1),fullfile([save_dir num2str(k) '.png']))
%     end
% 
% 
%            
% end
% nx = 95;
% 
% dens = zeros(nx,nx,nx);
% 
% test_mat = zeros(nx,nx,nx);
% 
% 
% for i=0:nx-1
%     tmp = flipud(rgb2gray(imread([dir box_name subbox_name peak_name num2str(i) '.png'])));
%     dens(:,:,i+1) = tmp(1:95,1:95);
% end
% 
% for i = 1:nx
%     for j = 1:nx
%         for k = 1:nx
%             if dens(i,j,k) >= 64
%                 test_mat(i,j,k) = 1;
%             else
%                 test_mat(i,j,k) = 0;
%             end
%         end
%     end
% end
% 
% 
% 
% CC = bwconncomp(test_mat,6);
% 
% numPixels = cellfun(@numel,CC.PixelIdxList);
% [biggest,idx] = max(numPixels);
% test_mat(CC.PixelIdxList{idx}) = 2;
% 
% for i = 1:nx
%     for j = 1:nx
%         for k = 1:nx
%             if test_mat(i,j,k) == 2
%                 test_mat(i,j,k) = 1;
%             else
%                 test_mat(i,j,k) = 0;
%             end
%         end
%     end
% end
% 
% 
% 
% volshow(test_mat)
% imtool3D(test_mat)