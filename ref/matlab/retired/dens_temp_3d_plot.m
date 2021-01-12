clc;clear;
data_path_dens = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box01/subbox01/dens/0';
data_path_temp = '/storage/filament/result/cluster_3d/40Mpc/density_temp/box01/subbox01/temp/0';

dens=zeros(101,101,101);
temp=zeros(101,101,101);
label=zeros(101,101,101);

for i=0:100
    dens(:,:,i+1) = flipud(rgb2gray(imread([data_path_dens '/' num2str(i) '.png'])));
    temp(:,:,i+1) = flipud(rgb2gray(imread([data_path_temp '/' num2str(i) '.png'])));
end



%% 밀도 너무 낮은 부분은 표시 안함.

for i = 1:101
    for j =1:101
        for k =1:101
            if dens(i,j,k) < 64
                
                %dens(i,j,k) = 0;
                label(i,j,k) = 0;
            else
                label(i,j,k) = 1;
            end
        end
    end
end



%% 연결선 찾기

CC = bwconncomp(label,6);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
label(CC.PixelIdxList{idx}) = 2;

%% 연결선만 뽑기 

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

%% isolated 된거 제외
label= double(bwmorph3(label,'clean'));
out = double(bwskel(logical(label),'MinBranchLength',1));


%% branch point 찾기.
out2= double(bwmorph3(out,'branchpoints'));

%%

%% branch point 와 label합침
filaments = out;
for i = 1:101
    for j = 1:101
        for k = 1:101
            if out2(i,j,k) == 1
                filaments(i,j,k) = 2;
            end
        end
    end
end

%%

img_bwcon = bwmorph3(label,'branchpoints');
img_bwcon = double(img_bwcon);

volumeViewer