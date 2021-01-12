box_name = 'box01/';
subbox_name = 'subbox01/';
peak_name = '0/';

img_dir = ['/storage/filament/result/cluster_3d/40Mpc/density_smoothing_plot_5_v2/' box_name subbox_name peak_name]  ;
label_dir = ['/storage/filament/result/cluster_3d/40Mpc/label_smoothing/' box_name subbox_name peak_name 'whole/0']  ;
smooth_dir = ['/storage/filament/result/cluster_3d/40Mpc/label_smoothing/' box_name subbox_name peak_name 'smoothing'];


nx =96;


img_3d = zeros(nx,nx,nx);
label_3d = zeros(nx,nx,nx);
smooth_3d = zeros(nx,nx,nx);

for i=0:nx-1
    img_3d(:,:,i+1) = flipud(rgb2gray(imread([img_dir '/' num2str(i) '.png'])));
    label_3d(:,:,i+1) = flipud(rgb2gray(imread([label_dir '/' num2str(i) '.png'])));
    smooth_3d(:,:,i+1) = flipud(rgb2gray(imread([smooth_dir '/' num2str(i) '.png'])));
end


volumeViewer
volumeViewer
volumeViewer
