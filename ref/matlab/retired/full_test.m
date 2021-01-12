clear;clc

dir = '/storage/filament/result/pic/pic_test/full';

nx=401;
dens = zeros(nx,nx,nx);
label = zeros(nx,nx,nx);


for i=0:nx-1
    dens(:,:,i+1) = flipud(rgb2gray(imread([dir '/' num2str(i) '.png'])));
end



for i = 1:nx
    for j =1:nx
        for k =1:nx
            if dens(i,j,k) < 64
                label(i,j,k) = 0;
            else
                label(i,j,k) = 1;
            end
        end
    end
end



CC = bwconncomp(label,26);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
label(CC.PixelIdxList{idx}) = 2;


for i = 1:nx
    for j =1:nx
        for k =1:nx
            if label(i,j,k) == 2
                label(i,j,k) = 1;
            else
                label(i,j,k) = 0;
            end
        end
    end
end


label= double(bwmorph3(label,'clean'));


filament = double(bwskel(logical(label),'MinBranchLength',1));

volumeViewer