%%
%path = '/storage/filament/works_v3/nexus/200Mpc_1_1024/1/';
path = '/storage/filament/works_v3/result/smoothing_test/refined/';
addpath('/storage/Codes/git/npy-matlab/npy-matlab')  ;

smoothing_list = [0.5,0.7,1.0,1.4,2.0,2.8,4.0];

%%

digits(2)
for i = 1:7
    scale = num2str(smoothing_list(i))
   


    signature = readNPY([path scale '_refined.npy']);


    intensity = [0 20 40 120 220 1024];
    alpha = [0 0.9 0.9 0.9 0.9 0.9];
    %color = ([0 0 0; 43 0 0; 103 37 20; 199 155 97; 216 213 201; 255 255 255]) ./ 255;
    queryPoints = linspace(min(intensity),max(intensity),256);
    alphamap = interp1(intensity,alpha,queryPoints)';
    %colormap = interp1(intensity,color,queryPoints);

    figure
    volshow(signature ,'Alphamap',alphamap,'BackgroundColor',[0.7 0.7 0.7]); 
    print(gcf,['/storage/filament/works_v3/result/smoothing_test/pics/' scale '_smoothing.eps'],'-deps2'); 
end