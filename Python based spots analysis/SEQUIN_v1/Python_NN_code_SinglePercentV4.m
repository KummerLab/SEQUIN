function Python_NN_code_MultiPercent_v3(fname,name,outfolder)
t0 = tic;
filelistAll_ch1 = dir([[fname '\ch1\'],'*.csv']);
filelistAll_ch2 = dir([[fname '\ch2\'],'*.csv']);
num_images = (size(filelistAll_ch1,1));% ./ 2)-1;
edges = [0	0.02	0.04	0.06	0.08	0.1	0.12	0.14	0.16	0.18	0.2	0.22	0.24	0.26	0.28	0.3	0.32	0.34	0.36	0.38	0.4	0.42	0.44	0.46	0.48	0.5	0.52	0.54	0.56	0.58	0.6	0.62	0.64	0.66	0.68	0.7	0.72	0.74	0.76	0.78	0.8	0.82	0.84	0.86	0.88	0.9	0.92	0.94	0.96	0.98	1	1.02	1.04	1.06	1.08	1.1	1.12	1.14	1.16	1.18	1.2	1.22	1.24	1.26	1.28	1.3	1.32	1.34	1.36	1.38	1.4	1.42	1.44	1.46	1.48	1.5	1.52	1.54	1.56	1.58	1.6	1.62	1.64	1.66	1.68	1.7	1.72	1.74	1.76	1.78	1.8	1.82	1.84	1.86	1.88	1.9	1.92	1.94	1.96	1.98	2	2.02	2.04	2.06	2.08	2.1	2.12	2.14	2.16	2.18	2.2	2.22	2.24	2.26	2.28	2.3	2.32	2.34	2.36	2.38	2.4	2.42	2.44	2.46	2.48	2.5	2.52	2.54	2.56	2.58	2.6	2.62	2.64	2.66	2.68	2.7	2.72	2.74	2.76	2.78	2.8	2.82	2.84	2.86	2.88	2.9	2.92	2.94	2.96	2.98	3];
%datax_correction = xlsread([XYcorrection '.xlsx'],'X');
%datay_correction = xlsread([XYcorrection '.xlsx'],'Y');
storage = cell(1,(2 * num_images));
storage2 = cell(1,(2 * num_images));
t = toc(t0);
disp(['load time: ' num2str(t)]); 
for i2=1:1
    alldata = cell(1,num_images);%zeros(1,27);
    percent = .2;
    %list = 1:2:10;%num_images;
    for i=1:num_images
    t2 = tic;   
    disp(i);
    ai1 = i;    
    ai2 = i+1;
    NNall = [];
    spots2all = [];
    spots1data = csvread([fname '\ch1\' filelistAll_ch1(i).name]);
    spots2data = csvread([fname '\ch2\' filelistAll_ch2(i).name]);
    spots1data = spots1data(:,1:10); %227
    spots2data = spots2data(:,1:10); %227
    filename = extractBetween(convertCharsToStrings(filelistAll_ch1(ai1).name),"",".csv");
    Tile = extractBetween(convertCharsToStrings(filelistAll_ch1(ai1).name),"(",")");
    %spots1data = spots1data(find(spots1data(:,2)>30),:);
    %spots2data = spots2data(find(spots2data(:,2)>30),:);
    
    spots1data = sortrows(spots1data,-3);
    spots2data = sortrows(spots2data,-3);
    spot1length = (size(spots1data,1) * percent);
    spot2length = (size(spots2data,1) * percent);
    spots1Raw = spots1data(1:spot1length,[6 5 4]);
    spots2Raw = spots2data(1:spot2length,[6 5 4]);
    spots1Subset = spots1data(1:spot1length,:);
    spots2Subset = spots2data(1:spot2length,:);
    spots1Coords = [spots1Raw(:,1)*.043,spots1Raw(:,2)*.043,spots1Raw(:,3)*.12];
    spots2Coords = [spots2Raw(:,1)*.043,spots2Raw(:,2)*.043,spots2Raw(:,3)*.12];

    Tile = str2num(Tile) +1;

        for i3=1:spot1length%%spotsloop
        spots1xyz = spots1Coords(i3 , 1:3); % Selection of XYZ position for current spots1 spot
        %% Calculation of Nearest Neighbor to current spot
        NN = pdist2(spots1xyz,spots2Coords, 'euclidean');
        NNrot = rot90(NN,3);
        NNrot = rmmissing(NNrot);
        
        spots2_partner = horzcat(NNrot,spots2Subset);
        NNmin = double(min(NN,[],2)); %min value for all columns
        spots2_partner = spots2_partner((spots2_partner(:,1) == NNmin),:);
        spots2_partner = spots2_partner(1,:);%%removes duplicatges
        NNall = vertcat (NNall, NNmin); %NN data only
        spots2all = vertcat (spots2all,spots2_partner); %spots2 detailed data
        %%flip analysis if set by user
        end
        %Freq
        % edges defines the current standard frequency distribution bin values.
        %spots2all(1,:)=[];
        spots1Subset = horzcat(spots1Subset,spots2all);
        writematrix(spots1Subset,[outfolder char(filename) '_' name 'AllData_20_Gauss_by_image.csv']);
        NNall = spots1Subset(:,11);
        NormalFreq = histcounts(NNall,edges);
        NormalFreq = rot90(NormalFreq,3);
        writematrix(NormalFreq,[outfolder char(filename) '_' name 'NormalFreq_20P_Gauss_by_image.xls']);
    
    storage{i} = NormalFreq;
    storage2{i} = spots1Subset;    
        
    t = toc(t2);
    disp(['image time: ' num2str(t)]);
    end


filename = [num2str(i2) '_' name '_Gauss.mat'];
%parsave(filename,NormalFreq);
%disp(['Interval: ' i2 ' processed']);    
end 
alldata = horzcat(storage{:});
alldata2 = vertcat(storage2{:});
NNall = alldata2(:,11);
NormalFreq = histcounts(NNall,edges);
NormalFreq = rot90(NormalFreq,3);
writematrix(alldata,[outfolder name 'Freq_by_image_20P_Gauss.csv']);
writematrix(alldata2,[outfolder name 'MetaData_20P_Gauss.csv']);
writematrix(NormalFreq,[outfolder name 'Freq_all_20P_Gauss.csv']);
t = toc(t0);
disp(['total time: ' num2str(t)]);
end
function parsave(filename, file)
save(filename,'file')
end