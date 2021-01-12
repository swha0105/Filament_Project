clc;clear;close all;

pic_dir = '/storage/filament/result/temp_cluster_3d/2d_pic/';
data_dir = '/storage/filament/result/temp_cluster_3d/raw_data/';   
%label_dir = '/storage/filament/data/';

nx = 74;

cluster_num = '3';

pic_3d = zeros(nx,nx,nx);
%label_3d = zeros(74,74,74);


for i=0:73
   pic_3d(:,:,i+1) = flipud(rgb2gray(imread([pic_dir cluster_num '/' num2str(i) '.png']))); 
%   label_3d(:,:,i+1) = flipud(rgb2gray(imread([label_dir cluster_num num2str(i) '.png'])));
end



filter = imgaussfilt3(pic_3d,0.5);

for i=0:73
   pic_3d(:,:,i+1) = flipud(rgb2gray(imread([pic_dir cluster_num '/' num2str(i) '.png']))); 
%   label_3d(:,:,i+1) = flipud(rgb2gray(imread([label_dir cluster_num num2str(i) '.png'])));
end


range = [32 64 96 128 160 192 224 256];



for iz = 1:nx
    for iy = 1:nx
        for ix = 1:nx
            if pic_3d(ix,iy,iz) >= 160
                pic_thres(ix,iy,iz) = 256;
            else
                pic_thres(ix,iy,iz) = 0;
            end
            
            if pic_3d(ix,iy,iz) >= 224
               pic_label(ix,iy,iz) = 1;
            else
               pic_label(ix,iy,iz) = 0;
            end
            
        end
    end
end

filter_thres = imgaussfilt3(pic_thres,0.5);
volumeViewer



% 
% data = load([data_dir cluster_num]);
% data_3d = reshape(data,[nx,nx,nx]);
% data_thres = zeros(nx,nx,nx);
% 
% 
% for ix = 1:nx
%     for iy = 1:nx
%         for iz = 1:nx
%             if data_3d(ix,iy,iz) >= 10^4 && data_3d(ix,iy,iz) < 10^5 
%                 data_thres(ix,iy,iz) = 128;
%             elseif data_3d(ix,iy,iz) >= 10^5 && data_3d(ix,iy,iz) <10^7
%                 data_thres(ix,iy,iz) = 256;
%             elseif data_3d(ix,iy,iz) >= 10^7
%                 data_thres(ix,iy,iz) = 256;
%             end
%         end
%     end
% end
