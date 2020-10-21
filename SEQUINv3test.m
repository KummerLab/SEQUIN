%README for SEQUINN analysis using Imaris 9.5+ CSV data output.
%Written by Andrew Sauerbeck, Mihika Gangoli, Keshav Kailash,and Lindsay
%Laws. Kummer Lab, Washington University St. Louis.
% This code can take up to three spots worth of data and perform a primary
% nearest neighbor colocalization analysis between 'spot 1' and 'spot 2'
% followed by a secondary colocalization with 'spot 3'.
% Input parameters:
% fname= ('X:\Active\P301s by ADS\Even-Piriform\G2normal_Statistics'); %%Location of input folder. This folder should contain a separate folder for each spot.
% % name = 'data output' %%Output filename
% spot1 = 'Spots_1_Statistics'; %%Name of spot 1 folder
% spot1ch = '1';                %% Channel # for spot 1 according to Imaris
% spot2 = 'Spots_2_Statistics'; %%Name of spot 2 folder
% spot2ch = '2';                %% Channel # for spot 2 according to Imaris
% spot3 = []; %%Name of spot 3 folder.
% spot3ch = [];                 %% Channel # for spot 3 according to Imaris
% spot1range = [80 100]; %%Intensity interval for spot selection. Requires min and max to be specified.
% spot2range = [80 100]; %%Intensity interval for spot selection. Requires min and max to be specified.
% spot3range = [];       %%Intensity interval for spot selection. Requires min and max to be specified.
% flip=[]; %% XY dimension of image. Assumes input image is square and coordinate space is reference to XYZ 0,0,0.
% spot1volume = []; %%Users selected volume cutoff for spots detection if RegionsGrowing was used in Imaris. Selects spots below cutoff.
% spot2volume = []; %%Users selected volume cutoff for spots detection if RegionsGrowing was used in Imaris. Selects spots below cutoff.
% spot3volume = []; %%Users selected volume cutoff for spots detection if RegionsGrowing was used in Imaris. Selects spots below cutoff.
% nnrange = [];     %% Nearest Neighbor range in nm to selected spot 1 and spot 2 subsets from to perform tricolocalization. Requires prior knowledge spot 1 vs spot2 frequency distribution to accurately set. 
% 
% 
% 
% spot1previousoutput = []; %% Reading CSV files can be time consuming. If code is to be rerun user can point to previously compiled data matrices to imporove performance.
% spot2previousoutput = ['X:\Active\P301s by ADS\Even-Piriform\G2normal_Statistics\80_100normal data output\New folder\Spots_1_Statisticsmatrix.csv'];
% spot3previousoutput = [];



%% Main Function 
function SEQUINv3(fname, name, spot1,spot1ch, spot2, spot2ch, spot1range, spot2range, nnrange, spot3, spot3ch,spot3range, flip, spot1volume,spot2volume, spot3volume,spot1previousoutput,spot2previousoutput,spot3previousoutput)

          
if(exist([fname '/data output'],'dir'))
else
     mkdir([fname '/data output']);
end
t0 = tic;
spot1ch = ['ch=' spot1ch];
spot2ch = ['ch=' spot2ch];
spot3ch = ['ch=' spot3ch];

filelist_spot1 = dir([fname '/' spot1 '/*']);
filelabel1 =  extractBetween(filelist_spot1(3).name,"","Area");
filelabel1 = char(filelabel1);


Spot1Position = readtable([fname '/' spot1 '/' filelabel1 'Position.csv'],'Delimiter', ',');
testingString = table2array(Spot1Position(:,13));
testingString = string(testingString);
datalabel = testingString;
datalabel2 = matlab.lang.makeValidName(datalabel);
animfld = unique(datalabel2); %%animfld is composed of all unique image names in the data set being processed.
animfld = transpose(animfld);




if(isempty(spot1previousoutput))

% The following portion reads in all metadata for Spot 1 followed by the metadata data for the channel for Spot 2 inside of Spot 1.
Spots1Area = readtable([fname '/' spot1 '/' filelabel1 'Area.csv'],'Delimiter', ',');
Spots1Area = Spots1Area(:,[1 6 10 ]);

Spots1IntensityMean1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Mean_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMean1 = Spots1IntensityMean1(:,[1 8 12]);

Spots1Position = readtable([fname '/' spot1 '/' filelabel1 'Position.csv'],'Delimiter', ',');
Spots1Position = Spots1Position(:,[1:3 9 12]);

Spots1IntensityMin1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Min_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMin1 = Spots1IntensityMin1(:,[1 8 12]);

Spots1IntensityMax1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Max_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMax1 = Spots1IntensityMax1(:,[1 8 12 ]);

Spots1IntensityCenter1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Center_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityCenter1 = Spots1IntensityCenter1(:,[1 8 12 ]);

Spots1IntensityMedian1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Median_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMedian1 = Spots1IntensityMedian1(:,[1 8 12]);

Spots1IntensityVolume = readtable([fname '/' spot1 '/' filelabel1 'Volume.csv'],'Delimiter', ',');
Spots1IntensityVolume = Spots1IntensityVolume(:,[1 6 10]);

Spots1IntensityStdDev1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_StdDev_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityStdDev1 = Spots1IntensityStdDev1(:,[1 8 12 ]);

Spots1IntensitySum1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Sum_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensitySum1 = Spots1IntensitySum1(:,[1 8 12]);

Spots1IntensitySum_of_Square1 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Sum_of_Square_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensitySum_of_Square1 = Spots1IntensitySum_of_Square1(:,[1 8 12 ]);

Spots1IntensityMean2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Mean_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMean2 = Spots1IntensityMean2(:,[1 8 12]);
Spots1IntensityMean2.Properties.VariableNames = {'IntensityMean2' 'ID' 'OriginalImageName' }; %%Variable for metadata from other channel remamed to be unique.

Spots1IntensityMin2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Min_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMin2 = Spots1IntensityMin2(:,[1 8 12]);
Spots1IntensityMin2.Properties.VariableNames = {'IntensityMin2' 'ID' 'OriginalImageName' };

Spots1IntensityMax2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Max_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMax2 = Spots1IntensityMax2(:,[1 8 12 ]);
Spots1IntensityMax2.Properties.VariableNames = {'IntensityMax2' 'ID' 'OriginalImageName' };

Spots1IntensityCenter2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Center_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityCenter2 = Spots1IntensityCenter2(:,[1 8 12 ]);
Spots1IntensityCenter2.Properties.VariableNames = {'IntensityCenter2' 'ID' 'OriginalImageName' };

Spots1IntensityMedian2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Median_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityMedian2 = Spots1IntensityMedian2(:,[1 8 12]);
Spots1IntensityMedian2.Properties.VariableNames = {'IntensityMedian2' 'ID' 'OriginalImageName' };

Spots1IntensityStdDev2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_StdDev_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensityStdDev2 = Spots1IntensityStdDev2(:,[1 8 12 ]);
Spots1IntensityStdDev2.Properties.VariableNames = {'IntensityStdDev2' 'ID' 'OriginalImageName' };

Spots1IntensitySum2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Sum_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensitySum2 = Spots1IntensitySum2(:,[1 8 12]);
Spots1IntensitySum2.Properties.VariableNames = {'IntensitySum2' 'ID' 'OriginalImageName' };

Spots1IntensitySum_of_Square2 = readtable([fname '/' spot1 '/' filelabel1 'Intensity_Sum_of_Square_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots1IntensitySum_of_Square2 = Spots1IntensitySum_of_Square2(:,[1 8 12 ]);
Spots1IntensitySum_of_Square2.Properties.VariableNames = {'IntensitySum_of_Square2' 'ID' 'OriginalImageName' };

%%Joining of all metadata tables. Critical Note!! Horzcat can not be used as Imaris does not output data across sheets in the same order of spots.
innerjoinedTable = innerjoin(Spots1Area, Spots1Position);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityVolume);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMin1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMax1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityCenter1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMean1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMedian1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityStdDev1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensitySum1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensitySum_of_Square1);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMin2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMax2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityCenter2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMean2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityMedian2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensityStdDev2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensitySum2);
innerjoinedTable = innerjoin(innerjoinedTable, Spots1IntensitySum_of_Square2);

%%Ordering of final data table for Spot 1
spots1table = [innerjoinedTable.IntensityMean innerjoinedTable.PositionX innerjoinedTable.PositionY innerjoinedTable.PositionZ innerjoinedTable.IntensityMax innerjoinedTable.IntensityCenter innerjoinedTable.Area innerjoinedTable.Volume innerjoinedTable.IntensityMin innerjoinedTable.IntensityMedian innerjoinedTable.IntensityStdDev innerjoinedTable.IntensitySum innerjoinedTable.IntensitySumOfSquare innerjoinedTable.IntensityMin2 innerjoinedTable.IntensityMax2 innerjoinedTable.IntensityCenter2 innerjoinedTable.IntensityMean2 innerjoinedTable.IntensityMedian2 innerjoinedTable.IntensityStdDev2 innerjoinedTable.IntensitySum2 innerjoinedTable.IntensitySum_of_Square2 innerjoinedTable.ID];

imageNames = string(innerjoinedTable.OriginalImageName);
imageNames = matlab.lang.makeValidName(imageNames);
spots1matrix = [spots1table imageNames];

 
filename = strcat(fname, ['/data output/' spot1 'matrix.csv']);
writematrix(spots1matrix,filename); %%Creation of combined output matirx for Spot 1 to speed up rerunning of code. This output data is used to populate the spot1previousoutput variable.
spots1matrix = array2table(spots1matrix);
 else
     spots1matrix = readtable(spot1previousoutput);

 
 end
 
if(isempty(spot2previousoutput))
% The following portion reads in all metadata for Spot 2 followed by the metadata data for the channel for Spot 1 inside of Spot 2.

      
filelist_spot2 = dir([fname '/' spot2 '/*']);
filelabel2 =  extractBetween(filelist_spot2(3).name,"","Area");
filelabel2 = char(filelabel2);

Spots2Area = readtable([fname '/' spot2 '/' filelabel2 'Area.csv'],'Delimiter', ',');
Spots2Area = Spots2Area(:,[1 6 10 ]);

Spots2IntensityMean1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Mean_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMean1 = Spots2IntensityMean1(:,[1 8 12]);

Spots2Position = readtable([fname '/' spot2 '/' filelabel2 'Position.csv'],'Delimiter', ',');
Spots2Position = Spots2Position(:,[1:3 9 12]);

Spots2IntensityMin1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Min_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMin1 = Spots2IntensityMin1(:,[1 8 12]);

Spots2IntensityMax1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Max_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMax1 = Spots2IntensityMax1(:,[1 8 12 ]);

Spots2IntensityCenter1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Center_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityCenter1 = Spots2IntensityCenter1(:,[1 8 12 ]);

Spots2IntensityMedian1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Median_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMedian1 = Spots2IntensityMedian1(:,[1 8 12]);

Spots2IntensityVolume = readtable([fname '/' spot2 '/' filelabel2 'Volume.csv'],'Delimiter', ',');
Spots2IntensityVolume = Spots2IntensityVolume(:,[1 6 10 ]);

Spots2IntensityStdDev1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_StdDev_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityStdDev1 = Spots2IntensityStdDev1(:,[1 8 12 ]);

Spots2IntensitySum1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Sum_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensitySum1 = Spots2IntensitySum1(:,[1 8 12]);

Spots2IntensitySum_of_Square1 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Sum_of_Square_' spot2ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensitySum_of_Square1 = Spots2IntensitySum_of_Square1(:,[1 8 12 ]);

Spots2IntensityMean2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Mean_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMean2 = Spots2IntensityMean2(:,[1 8 12]);
Spots2IntensityMean2.Properties.VariableNames = {'IntensityMean2' 'ID' 'OriginalImageName' }; 
Spots2IntensityMin2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Min_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMin2 = Spots2IntensityMin2(:,[1 8 12]);
Spots2IntensityMin2.Properties.VariableNames = {'IntensityMin2' 'ID' 'OriginalImageName' };

Spots2IntensityMax2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Max_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMax2 = Spots2IntensityMax2(:,[1 8 12 ]);
Spots2IntensityMax2.Properties.VariableNames = {'IntensityMax2' 'ID' 'OriginalImageName' };

Spots2IntensityCenter2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Center_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityCenter2 = Spots2IntensityCenter2(:,[1 8 12 ]);
Spots2IntensityCenter2.Properties.VariableNames = {'IntensityCenter2' 'ID' 'OriginalImageName' };

Spots2IntensityMedian2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Median_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityMedian2 = Spots2IntensityMedian2(:,[1 8 12]);
Spots2IntensityMedian2.Properties.VariableNames = {'IntensityMedian2' 'ID' 'OriginalImageName' };

Spots2IntensityStdDev2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_StdDev_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensityStdDev2 = Spots2IntensityStdDev2(:,[1 8 12 ]);
Spots2IntensityStdDev2.Properties.VariableNames = {'IntensityStdDev2' 'ID' 'OriginalImageName' };

Spots2IntensitySum2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Sum_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensitySum2 = Spots2IntensitySum2(:,[1 8 12]);
Spots2IntensitySum2.Properties.VariableNames = {'IntensitySum2' 'ID' 'OriginalImageName' };

Spots2IntensitySum_of_Square2 = readtable([fname '/' spot2 '/' filelabel2 'Intensity_Sum_of_Square_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots2IntensitySum_of_Square2 = Spots2IntensitySum_of_Square2(:,[1 8 12 ]);
Spots2IntensitySum_of_Square2.Properties.VariableNames = {'IntensitySum_of_Square2' 'ID' 'OriginalImageName' };


innerjoinedTable2 = innerjoin(Spots2Area, Spots2Position); 
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityVolume);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMin1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMax1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityCenter1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMean1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMedian1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityStdDev1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensitySum1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensitySum_of_Square1);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMin2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMax2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityCenter2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMean2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityMedian2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensityStdDev2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensitySum2);
innerjoinedTable2 = innerjoin(innerjoinedTable2, Spots2IntensitySum_of_Square2);

spots2table = [innerjoinedTable2.IntensityMean innerjoinedTable2.PositionX innerjoinedTable2.PositionY innerjoinedTable2.PositionZ innerjoinedTable2.IntensityMax innerjoinedTable2.IntensityCenter innerjoinedTable2.Area innerjoinedTable2.Volume innerjoinedTable2.IntensityMin innerjoinedTable2.IntensityMedian innerjoinedTable2.IntensityStdDev innerjoinedTable2.IntensitySum innerjoinedTable2.IntensitySumOfSquare innerjoinedTable2.IntensityMin2 innerjoinedTable2.IntensityMax2 innerjoinedTable2.IntensityCenter2 innerjoinedTable2.IntensityMean2 innerjoinedTable2.IntensityMedian2 innerjoinedTable2.IntensityStdDev2 innerjoinedTable2.IntensitySum2 innerjoinedTable2.IntensitySum_of_Square2 innerjoinedTable2.ID];
imageNames = string(innerjoinedTable2.OriginalImageName);
imageNames = matlab.lang.makeValidName(imageNames);
spots2matrix = [spots2table imageNames];
filename = strcat(fname, ['/data output/' spot2 'matrix.csv']);
writematrix(spots2matrix,filename);
spots2matrix = array2table(spots2matrix);
else
 spots2matrix = readtable(spot2previousoutput);
end


if(~isempty(spot3))
if(isempty(spot3previousoutput))
% The following portion reads in all metadata for Spot 3 followed by the metadata data for the channel for Spot 1 inside of Spot 3.


filelist_spot3 = dir([fname '/' spot3 '/*']);
filelabel3 =  extractBetween(filelist_spot3(3).name,"","Area");
filelabel3 = char(filelabel3);

Spots3Area = readtable([fname '/' spot3 '/' filelabel3 'Area.csv']);
Spots3Area = Spots3Area(:,[1 6 10 ]);

Spots3IntensityMean1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Mean_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMean1 = Spots3IntensityMean1(:,[1 8 12]);

Spots3Position = readtable([fname '/' spot3 '/' filelabel3 'Position.csv'],'Delimiter', ',');
Spots3Position = Spots3Position(:,[1:3 9 12]);

Spots3IntensityMin1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Min_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMin1 = Spots3IntensityMin1(:,[1 8 12]);

Spots3IntensityMax1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Max_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMax1 = Spots3IntensityMax1(:,[1 8 12 ]);

Spots3IntensityCenter1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Center_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityCenter1 = Spots3IntensityCenter1(:,[1 8 12 ]);

Spots3IntensityMedian1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Median_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMedian1 = Spots3IntensityMedian1(:,[1 8 12]);

Spots3IntensityVolume = readtable([fname '/' spot3 '/' filelabel3 'Volume.csv'],'Delimiter', ',');
Spots3IntensityVolume = Spots3IntensityVolume(:,[1 6 10 ]);

Spots3IntensityStdDev1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_StdDev_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityStdDev1 = Spots3IntensityStdDev1(:,[1 8 12 ]);

Spots3IntensitySum1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Sum_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensitySum1 = Spots3IntensitySum1(:,[1 8 12]);

Spots3IntensitySum_of_Square1 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Sum_of_Square_' spot3ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensitySum_of_Square1 = Spots3IntensitySum_of_Square1(:,[1 8 12 ]);

Spots3IntensityMean2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Mean_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMean2 = Spots3IntensityMean2(:,[1 8 12]);
Spots3IntensityMean2.Properties.VariableNames = {'IntensityMean2' 'ID' 'OriginalImageName' };

Spots3IntensityMin2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Min_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMin2 = Spots3IntensityMin2(:,[1 8 12]);
Spots3IntensityMin2.Properties.VariableNames = {'IntensityMin2' 'ID' 'OriginalImageName' };

Spots3IntensityMax2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Max_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMax2 = Spots3IntensityMax2(:,[1 8 12 ]);
Spots3IntensityMax2.Properties.VariableNames = {'IntensityMax2' 'ID' 'OriginalImageName' };

Spots3IntensityCenter2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Center_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityCenter2 = Spots3IntensityCenter2(:,[1 8 12 ]);
Spots3IntensityCenter2.Properties.VariableNames = {'IntensityCenter2' 'ID' 'OriginalImageName' };

Spots3IntensityMedian2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Median_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityMedian2 = Spots3IntensityMedian2(:,[1 8 12]);
Spots3IntensityMedian2.Properties.VariableNames = {'IntensityMedian2' 'ID' 'OriginalImageName' };

Spots3IntensityStdDev2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_StdDev_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensityStdDev2 = Spots3IntensityStdDev2(:,[1 8 12 ]);
Spots3IntensityStdDev2.Properties.VariableNames = {'IntensityStdDev2' 'ID' 'OriginalImageName' };

Spots3IntensitySum2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Sum_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensitySum2 = Spots3IntensitySum2(:,[1 8 12]);
Spots3IntensitySum2.Properties.VariableNames = {'IntensitySum2' 'ID' 'OriginalImageName' };

Spots3IntensitySum_of_Square2 = readtable([fname '/' spot3 '/' filelabel3 'Intensity_Sum_of_Square_' spot1ch '_Img=1.csv'],'Delimiter', ',');
Spots3IntensitySum_of_Square2 = Spots3IntensitySum_of_Square2(:,[1 8 12 ]);
Spots3IntensitySum_of_Square2.Properties.VariableNames = {'IntensitySum_of_Square2' 'ID' 'OriginalImageName' };

innerjoinedTable3 = innerjoin(Spots3Area, Spots3Position);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityVolume);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMin1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMax1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityCenter1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMean1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMedian1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityStdDev1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensitySum1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensitySum_of_Square1);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMin2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMax2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityCenter2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMean2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityMedian2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensityStdDev2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensitySum2);
innerjoinedTable3 = innerjoin(innerjoinedTable3, Spots3IntensitySum_of_Square2);

spots3table = [innerjoinedTable3.IntensityMean innerjoinedTable3.PositionX innerjoinedTable3.PositionY innerjoinedTable3.PositionZ innerjoinedTable3.IntensityMax innerjoinedTable3.IntensityCenter innerjoinedTable3.Area innerjoinedTable3.Volume innerjoinedTable3.IntensityMin innerjoinedTable3.IntensityMedian innerjoinedTable3.IntensityStdDev innerjoinedTable3.IntensitySum innerjoinedTable3.IntensitySumOfSquare innerjoinedTable3.IntensityMin2 innerjoinedTable3.IntensityMax2 innerjoinedTable3.IntensityCenter2 innerjoinedTable3.IntensityMean2 innerjoinedTable3.IntensityMedian2 innerjoinedTable3.IntensityStdDev2 innerjoinedTable3.IntensitySum2 innerjoinedTable3.IntensitySum_of_Square2 innerjoinedTable3.ID];
imageNames = string(innerjoinedTable3.OriginalImageName);
imageNames = matlab.lang.makeValidName(imageNames);
spots3matrix = [spots3table imageNames];
filename = strcat(fname, ['/data output/' spot3 'matrix.csv']);
writematrix(spots3matrix,filename);
spots3matrix = array2table(spots3matrix);
 else
     spots3matrix = readtable(spot3previousoutput);
 end
end


for i = 1:length(animfld) %% Loops through the number of unique image sets in the current data being processed.

    
animfld22 = char(animfld(i)) ;

spots1= spots1matrix(find(strcmp(char(animfld22),spots1matrix{:,23})),:); %select metadata for current image being processed.
spots2= spots2matrix(find(strcmp(char(animfld22),spots2matrix{:,23})),:); %select metadata for current image being processed.

if(~isempty(spot3))
spots3= spots3matrix(find(strcmp(char(animfld22),spots3matrix{:,23})),:); %select metadata for current image being processed.
end

animfld2 = animfld(i); %%Used for file naming.


spots1=spots1(:,1:end-1); %Removes image name from data being processed.
spots2=spots2(:,1:end-1);
if(~isempty(spot3))
spots3=spots3(:,1:end-1);
end

%Converts current data to table.
spots1 = table2array(spots1); 
spots2 = table2array(spots2);  
if(~isempty(spot3))
spots3 = table2array(spots3); 
end

%If first run of data, converts data to double.
if(isempty(spot2previousoutput)) 
spots1 = str2double(spots1);
spots2 = str2double(spots2); 
if(~isempty(spot3))
spots3 = str2double(spots3); 
end
end

% Stores data in Structure 'data'.
data.spots1 = spots1; 
data.spots2 = spots2;
if(~isempty(spot3))
data.spots3 = spots3;
end

% Performs volume filtering of spots if set by user.
if(~isempty(spot1volume)) 
    data.spots1 = data.spots1(find(data.spots1(:,8) < spot1volume),:);% Spot 1 volume filter
end
    
if(~isempty(spot2volume))
    data.spots2 = data.spots2(find(data.spots2(:,8) < spot2volume),:);% Spot 2 volume filter 
end
if(~isempty(spot3))
 if(~isempty(spot3volume))
    data.spots3 = data.spots3(find(data.spots3(:,8) < spot3volume),:);% Spot 3 volume filter
 end
end





% Sends current images data to function to perform Nearest Neighbors processing.
NearestNeighborCalculations(data,fname, name,spot1range,spot2range,flip,animfld2,spot3range,nnrange); 
end
end   





%% Function to perform Nearest Neighbors processing.
function NearestNeighborCalculations(data, dirname, fname,spots1range,spots2range,flip,animfld2,spot3range,nnrange)
                                                                      
 
 

%% data sorting by mean intensity of spot

    data.spots1sort = getsortedData(data.spots1);
    data.spots2sort = getsortedData(data.spots2); 

%% Gets data subsets set by user based upon mean intensity of spot
    data.spots1Subset = getsubsetData(data.spots1sort,spots1range);
    data.spots2Subset = getsubsetData(data.spots2sort,spots2range);


%% Calculation of Nearest Neighboring Spot 2 based upon Spot 1
    
    spots1Coords = NaN(max(length(data.spots2Subset),length(data.spots1Subset)),3);
    spots2Coords = NaN(max(length(data.spots2Subset),length(data.spots1Subset)),3);

    
    spots1Coords(1:length(data.spots1Subset),1:3)= data.spots1Subset(:,2:4);
    spots2Coords(1:length(data.spots2Subset),1:3) = data.spots2Subset(:,2:4);
    
    %% If flip value is set by user, generates Flipped Spot 2 positions
    if(~isempty(flip))
    Flippedspots2Coords = [(spots2Coords(:,1)),(flip-spots2Coords(:,2)),(spots2Coords(:,3))];
    end
    
    
    
    
    
    NNall = NaN(1,1); %establish NNall matrix
    NNflipall = NaN(1,1); %establish NNflipall matrix
    indexingSize = size(data.spots1Subset);
    indexingSize = indexingSize(2)+1;
    spots2all = NaN(1,indexingSize); %establish NNall matrix
    spots2allflip = NaN(1,indexingSize); %establish NNall matrix
    spots1loop = rmmissing(spots1Coords); % Number of 'Spot 1' spots in current image
    
    
    for ai5 =1:length(spots1loop) % Loops through number of 'Spot 1' spots
   
    spots1xyz = spots1Coords(ai5 , 1:3); % Selection of XYZ position for current spots1 spot
   
    %% Calculation of Nearest Neighbor to current spot
    NN = pdist2(spots1xyz,spots2Coords, 'euclidean');
    NNrot = rot90(NN,3);
    NNrot = rmmissing(NNrot);
    spots2_partner = horzcat(NNrot,data.spots2Subset);
    NNmin = double(min(NN,[],2)); %min value for all columns
    spots2_partner = spots2_partner((spots2_partner(:,1) == NNmin),:);
    spots2_partner = spots2_partner(1,:);%%removes duplicatges
    NNall = vertcat (NNall, NNmin); %NN data only
    spots2all = vertcat (spots2all,spots2_partner); %spots2 detailed data
    
    %%flip analysis if set by user
    if(~isempty(flip))
    NNflip = pdist2(spots1xyz,Flippedspots2Coords, 'euclidean');
    NNfliprot = rot90(NNflip,3);
    NNfliprot = rmmissing(NNfliprot);
    spots2_partnerflip = horzcat(NNfliprot,data.spots2Subset);
    NNflipmin = min(NNflip,[],2); %min value for all columns
    spots2_partnerflip = spots2_partnerflip(find(spots2_partnerflip(:,1) == NNflipmin),:);
    spots2_partnerflip = spots2_partnerflip(1,:); %%removes duplicatges
    NNflipall = vertcat (NNflipall, NNflipmin); %NN data only
    spots2allflip = vertcat (spots2allflip,spots2_partnerflip); %spots2apsin detailed data
    end
   
    
    end
    spots2all(1,:) = [];
    totaldata= horzcat (data.spots1Subset, spots2all); %total detailed data for current spot. Laybout: Spot1metadata--Nearest Neighbor distance--Spot2metadata
    
    
    if(~isempty(flip))
    spots2allflip(1,:) = [];
    totaldataflip= horzcat (data.spots1Subset, spots2allflip); %total detailed data for flip analysis
    end

    % edges defines the current standard frequency distribution bin values. 
    edges = [0	0.02	0.04	0.06	0.08	0.1	0.12	0.14	0.16	0.18	0.2	0.22	0.24	0.26	0.28	0.3	0.32	0.34	0.36	0.38	0.4	0.42	0.44	0.46	0.48	0.5	0.52	0.54	0.56	0.58	0.6	0.62	0.64	0.66	0.68	0.7	0.72	0.74	0.76	0.78	0.8	0.82	0.84	0.86	0.88	0.9	0.92	0.94	0.96	0.98	1	1.02	1.04	1.06	1.08	1.1	1.12	1.14	1.16	1.18	1.2	1.22	1.24	1.26	1.28	1.3	1.32	1.34	1.36	1.38	1.4	1.42	1.44	1.46	1.48	1.5	1.52	1.54	1.56	1.58	1.6	1.62	1.64	1.66	1.68	1.7	1.72	1.74	1.76	1.78	1.8	1.82	1.84	1.86	1.88	1.9	1.92	1.94	1.96	1.98	2	2.02	2.04	2.06	2.08	2.1	2.12	2.14	2.16	2.18	2.2	2.22	2.24	2.26	2.28	2.3	2.32	2.34	2.36	2.38	2.4	2.42	2.44	2.46	2.48	2.5	2.52	2.54	2.56	2.58	2.6	2.62	2.64	2.66	2.68	2.7	2.72	2.74	2.76	2.78	2.8	2.82	2.84	2.86	2.88	2.9	2.92	2.94	2.96	2.98	3];
     
    % Generates frequency distibution for normal analysis
    NormalFreq = histcounts(NNall,edges);
    NormalFreq = rot90(NormalFreq,3);
    
    % Generates frequency distibution for flip analysis
    if(~isempty(flip))
    FlippedFreq = histcounts(NNflipall,edges);
    FlippedFreq = rot90(FlippedFreq,3);
    % Performs flip subtraction to calculate number of above chance colocalizations within a given frequency bin.    
    Flipsubtraction = (NormalFreq - FlippedFreq);
    % Sets all negative values to zero since the question of interest is number above colocalizations above chance only.   
    FlipsubtractionZeros = max(Flipsubtraction,0);    
    end
    
    
    
    if(exist([dirname '/data output'],'dir'))
    else
     mkdir(['data output']);
    end
    
    
    

      filename = strcat(dirname, '/data output/', animfld2,'.xlsx');

    % Converts totaldata to table, renames columns, and stores data.
    totaldataTable = array2table(totaldata);
    %% The 'S#' in front of a variable indentifies the Spot used to get data from the image. If no # at the end then the channel is the one the spot was created with. 
    %%If a # is at the end it corresponds to the channel the meta data comes from.
    totaldataTable.Properties.VariableNames = {'S1IntensityMean' 'S1PositionX' 'S1PositionY' 'S1PositionZ' 'S1IntensityMax' 'S1IntensityCenter' 'S1Area' 'S1Volume' 'S1IntensityMin' 'S1IntensityMedian' 'S1IntensityStdDev' 'S1IntensitySum' 'S1IntensitySumOfSquare' 'S1IntensityMin2' 'S1IntensityMax2' 'S1IntensityCenter2' 'S1IntensityMean2' 'S1IntensityMedian2' 'S1IntensityStdDev2' 'S1IntensitySum2' 'S1IntensitySumOfSquare2' 'S1ID' 'NearestNeighbor' 'S2IntensityMean' 'S2PositionX' 'S2PositionY' 'S2PositionZ' 'S2IntensityMax' 'S2IntensityCenter' 'S2Area' 'S2Volume' 'S2IntensityMin' 'S2ntensityMedian' 'S2IntensityStdDev' 'S2IntensitySum' 'S2IntensitySumOfSquare' 'S2IntensityMin1' 'S2IntensityMax1' 'S2IntensityCenter1' 'S2IntensityMean1' 'S2IntensityMedian1' 'S2IntensityStdDev1' 'S2IntensitySum1' 'S2IntensitySumOfSquare1' 'SYNS1ID'};
    writetable(totaldataTable,filename,'Sheet','Normal Raw Data');
    
    % Converts NormalFreq to table and stores data.
    normalFreqTable = array2table(NormalFreq);
    writetable(normalFreqTable,filename,'Sheet','Normal Frequency')
    
    % Writes data for flip analysis if set by user.
    if(~isempty(flip))
        totaldataflipTable = array2table(totaldataflip);
        totaldataflipTable.Properties.VariableNames = {'S1IntensityMean' 'S1PositionX' 'S1PositionY' 'S1PositionZ' 'S1IntensityMax' 'S1IntensityCenter' 'S1Area' 'S1Volume' 'S1IntensityMin' 'S1IntensityMedian' 'S1IntensityStdDev' 'S1IntensitySum' 'S1IntensitySumOfSquare' 'S1IntensityMin2' 'S1IntensityMax2' 'S1IntensityCenter2' 'S1IntensityMean2' 'S1IntensityMedian2' 'S1IntensityStdDev2' 'S1IntensitySum2' 'S1IntensitySumOfSquare2' 'S1ID' 'NearestNeighbor' 'S2IntensityMean' 'S2PositionX' 'S2PositionY' 'S2PositionZ' 'S2IntensityMax' 'S2IntensityCenter' 'S2Area' 'S2Volume' 'S2IntensityMin' 'S2ntensityMedian' 'S2IntensityStdDev' 'S2IntensitySum' 'S2IntensitySumOfSquare' 'S2IntensityMin1' 'S2IntensityMax1' 'S2IntensityCenter1' 'S2IntensityMean1' 'S2IntensityMedian1' 'S2IntensityStdDev1' 'S2IntensitySum1' 'S2IntensitySumOfSquare1' 'S2ID'};
        writetable(totaldataflipTable,filename,'Sheet','Flipped Raw Data');
        FlippedFreqTable = array2table(FlippedFreq);
        writetable(FlippedFreqTable,filename,'Sheet','Flipped Frequency');
        FlipSubtractionTable = array2table(Flipsubtraction);
        writetable(FlipSubtractionTable,filename,'Sheet','Flip Subtraction Frequency');
        FlipsubtractionZerosTable = array2table(FlipsubtractionZeros);
        writetable(FlipsubtractionZerosTable,filename,'Sheet','Flip Sub Zeros Frequency');
    end
    
   % Performs secondary colocalization to Spot 3 if set by user. Required variables: nnrage, spot3, spot3ch  
   if(~isempty(nnrange))
    nnrange =nnrange/1000; % User specifies in nm, convert to microns to remain consistent with previous output
    subsetspots = totaldata((totaldata(:,23)>=nnrange(1) & totaldata(:,23)<nnrange(2)),:); % Selects Spot1-Spot2 data within nearest neighbor range set by nnrange.
    subsetspots1 = subsetspots(:,1:22); % Gets Spot 1 specific data
    subsetspots2 = subsetspots(:,24:45);% Gets Spot 2 specific data
    subsetspotsNN = subsetspots(:,23);  % Gets nearest neighbot data data
    
    % Sorts and selects Spot 3 based upon mean intensity
    data.spot3sort = getsortedData(data.spots3); 
    data.spot3Subset = getsubsetData(data.spot3sort,spot3range);
    
    % Gets spot coordinates specific to secondary colocalization of Spot 1 and Spot 2 puncta indentified as synaptic.
    spot1Coords2 = subsetspots(:,2:4);
    spot2Coords2 = subsetspots(:,25:27);
    spot3Coords = data.spot3Subset(:,2:4);
    
    % Creates flipped coordinates for Spot 1 and Spot2 if set by user.
     if(~isempty(flip))
     spot1Coords2flip = [(spot1Coords2(:,1)),(flip-spot1Coords2(:,2)),(spot1Coords2(:,3))];
     spot2Coords2flip = [(spot2Coords2(:,1)),(flip-spot2Coords2(:,2)),(spot2Coords2(:,3))];
     end

    
   %Creates matrices to store data
    NNall3 = NaN(1,1);
    spot3all1 = NaN(1,indexingSize);
    NNall4 = NaN(1,1);
    spot3all2 = NaN(1,indexingSize);
    
    NNall3flip = NaN(1,1);
    spot3all1flip = NaN(1,indexingSize);
    NNall4flip = NaN(1,1);
    spot3all2flip = NaN(1,indexingSize);
   
    totaldataspot31 = NaN(1,1);
    totaldataspot32 = NaN(1,1);
    
    totaldataspot31flip = NaN(1,1);
    totaldataspot32flip = NaN(1,1);
    
    
    for ai7 =1:length(spot1Coords2)
   % Performs Spot 1 vs Spot 3 Nearest Neighbor colocalization
    spot1xyz3 = spot1Coords2(ai7 , 1:3); % PSD coordinate fir #=ai
    NN3 = pdist2(spot1xyz3,spot3Coords, 'euclidean');
    NNrot3 = rot90(NN3,3);
    NNrot3 = rmmissing(NNrot3);
    spot3_partner1 = horzcat(NNrot3,data.spot3Subset);
    NNmin3 = double(min(NN3,[],2)); %min value for all columns
    spot3_partner1 = spot3_partner1(find(spot3_partner1(:,1) == NNmin3),:);
    spot3_partner1 = spot3_partner1(1,:);%%removes duplicatges
    NNall3 = vertcat (NNall3, NNmin3); %NN data only
    spot3all1 = vertcat (spot3all1,spot3_partner1); %synapsin detailed data
    
    % Performs Spot 2 vs Spot 3 Nearest Neighbor colocalization    
    spot2xyz3 = spot2Coords2(ai7 , 1:3); % PSD coordinate fir #=ai
    NN4 = pdist2(spot2xyz3,spot3Coords, 'euclidean');
    NNrot4 = rot90(NN4,3);
    NNrot4 = rmmissing(NNrot4);
    spot3_partner2 = horzcat(NNrot4,data.spot3Subset);
    NNmin4 = double(min(NN4,[],2)); %min value for all columns
    spot3_partner2 = spot3_partner2(find(spot3_partner2(:,1) == NNmin4),:);
    spot3_partner2 = spot3_partner2(1,:);%%removes duplicatges
    NNall4 = vertcat (NNall4, NNmin4); %NN data only
    spot3all2 = vertcat (spot3all2,spot3_partner2); %synapsin detailed data
    
    % Performs flip analysis if set by user.
    if(~isempty(flip))
      
    spot1xyz3flip = spot1Coords2flip(ai7 , 1:3); % PSD coordinate fir #=ai
    NN3flip = pdist2(spot1xyz3flip,spot3Coords, 'euclidean');
    NNrot3flip = rot90(NN3flip,3);
    NNrot3flip = rmmissing(NNrot3flip);
    spot3_partner1flip = horzcat(NNrot3flip,data.spot3Subset);
    NNmin3flip = double(min(NN3flip,[],2)); %min value for all columns
    spot3_partner1flip = spot3_partner1flip(find(spot3_partner1flip(:,1) == NNmin3flip),:);
    spot3_partner1flip = spot3_partner1flip(1,:);%%removes duplicatges
    NNall3flip = vertcat (NNall3flip, NNmin3flip); %NN data only
    spot3all1flip = vertcat (spot3all1flip,spot3_partner1flip); %synapsin detailed data
    
        
    spot2xyz3flip = spot2Coords2flip(ai7 , 1:3); % PSD coordinate fir #=ai
    NN4flip = pdist2(spot2xyz3flip,spot3Coords, 'euclidean');
    NNrot4flip = rot90(NN4flip,3);
    NNrot4flip = rmmissing(NNrot4flip);
    spot3_partner2flip = horzcat(NNrot4flip,data.spot3Subset);
    NNmin4flip = double(min(NN4flip,[],2)); %min value for all columns
    spot3_partner2flip = spot3_partner2flip(find(spot3_partner2flip(:,1) == NNmin4flip),:);
    spot3_partner2flip = spot3_partner2flip(1,:);%%removes duplicatges
    NNall4flip = vertcat (NNall4flip, NNmin4flip); %NN data only
    spot3all2flip = vertcat (spot3all2flip,spot3_partner2flip); %synapsin detailed data

    end
    
    end   
    
    % Creates frequency distribution of Spot 1 spots identified by nnrange versus Spot 3 spots.
    Spot3_1Freq = histcounts(NNall3,edges);
    Spot3_1Freq = rot90(Spot3_1Freq,3);
    % Creates frequency distribution of Spot 2 spots identified by nnrange versus Spot 3 spots.
    Spot3_2Freq = histcounts(NNall4,edges);
    Spot3_2Freq = rot90(Spot3_2Freq,3);

    spot3all1(1,:) = [];
    spot3all2(1,:) = [];
    % Creates metadata table with Spot1 or 2 data, followed by nearest neighbor distance to spot 3, followed for spot 3 metadata.
    totaldataspot31= horzcat (subsetspots1, subsetspotsNN);
    totaldataspot31= horzcat (subsetspots1, spot3all1);
    totaldataspot32= horzcat (subsetspots2, subsetspotsNN);
    totaldataspot32= horzcat (subsetspots2, spot3all2);
    
    % Renames column headings and write data
    totaldataspot31 = array2table(totaldataspot31);
    totaldataspot31.Properties.VariableNames = {'S1IntensityMeanTriC' 'S1PositionX' 'S1PositionY' 'S1PositionZ' 'S1IntensityMaxTriC' 'S1IntensityCenterTriC' 'S1AreaTriC' 'S1VolumeTriC' 'S1IntensityMinTriC' 'S1IntensityMedianTriC' 'S1IntensityStdDevTriC' 'S1IntensitySumTriC' 'S1IntensitySumOfSquareTriC' 'S1IntensityMin2TriC' 'S1IntensityMax2TriC' 'S1IntensityCenter2TriC' 'S1IntensityMean2TriC' 'S1IntensityMedian2TriC' 'S1IntensityStdDev2TriC' 'S1IntensitySum2TriC' 'S1IntensitySumOfSquare2TriC' 'S1ID' 'S1NearestNeighbor3TriC' 'ForS1_S3IntensityMeanTriC' 'ForS1_S3PositionXTriC' 'ForS1_S3PositionYTriC' 'ForS1_S3PositionZTriC' 'ForS1_S3IntensityMaxTriC' 'ForS1_S3IntensityCenterTriC' 'ForS1_S3AreaTriC' 'ForS1_S3VolumeTriC' 'ForS1_S3IntensityMinTriC' 'ForS1_S3ntensityMedianTriC' 'ForS1_S3IntensityStdDevTriC' 'ForS1_S3IntensitySumTriC' 'ForS1_S3IntensitySumOfSquareTriC' 'ForS1_S3IntensityMin1TriC' 'ForS1_S3IntensityMax1TriC' 'ForS1_S3IntensityCenter1TriC' 'ForS1_S3IntensityMean1TriC' 'ForS1_S3IntensityMedian1TriC' 'ForS1_S3IntensityStdDev1TriC' 'ForS1_S3IntensitySum1TriC' 'ForS1_S3IntensitySumOfSquare1TriC' 'ForS1_S3IDTriC'};
    writetable(totaldataspot31,filename,'Sheet','TriCSpot1_3');
    totaldataspot32 = array2table(totaldataspot32);
    totaldataspot32.Properties.VariableNames = {'S2IntensityMeanTriC' 'S2PositionX' 'S2PositionY' 'S2PositionZ' 'S2IntensityMaxTriC' 'S2IntensityCenterTriC' 'S2AreaTriC' 'S2VolumeTriC' 'S2IntensityMinTriC' 'S2IntensityMedianTriC' 'S2IntensityStdDevTriC' 'S2IntensitySumTriC' 'S2IntensitySumOfSquareTriC' 'S2IntensityMin1TriC' 'S2IntensityMax1TriC' 'S2IntensityCenter1TriC' 'S2IntensityMean1TriC' 'S2IntensityMedian1TriC' 'S2IntensityStdDev1TriC' 'S2IntensitySum1TriC' 'S2IntensitySumOfSquare1TriC' 'S2ID' 'S2NearestNeighbor3TriC' 'S3IntensityMeanTriC' 'S3PositionXTriC' 'S3PositionYTriC' 'S3PositionZTriC' 'S3IntensityMaxTriC' 'S3IntensityCenterTriC' 'S3AreaTriC' 'S3VolumeTriC' 'S3IntensityMinTriC' 'S3ntensityMedianTriC' 'S3IntensityStdDevTriC' 'S3IntensitySumTriC' 'S3IntensitySumOfSquareTriC' 'S3IntensityMin1TriC' 'S3IntensityMax1TriC' 'S3IntensityCenter1TriC' 'S3IntensityMean1TriC' 'S3IntensityMedian1TriC' 'S3IntensityStdDev1TriC' 'S3IntensitySum1TriC' 'S3IntensitySumOfSquare1TriC' 'S3IDTriC'};
    writetable(totaldataspot32,filename,'Sheet','TriCSpot2_3')
    
    % Create metadata table containing data from initial Spot 1 versus Spot 2 colocaization followed by meata data from secondary colocalization
    AllDataSpot1 = innerjoin ( totaldataTable,totaldataspot31);
    AllDataSpot1 = AllDataSpot1 (:,[1:45 64:86]);  %Old version (:,[1:45 64 56:63 65:86]);
    
    % Converts Spot 2 versus Spot 3 data to table
    totaldataspot32 = table2array(totaldataspot32);
    % Creates storage structure the size for Spot 1 data selected by nnrange
    storage = cell(1,size(AllDataSpot1,1));
    % Indetifies the Spot 2 partner that was paired with Spot 1 during during initial analysis. Stores the Spot 2 versus Spot 3 data for that partner in storage stucture.
    % Since a given Spot 2 could have been paired with multiple Spot 1 spots, this corrects for any table size and alignment mismatches.
    for i=1:size(AllDataSpot1,1)
        Spot2id = AllDataSpot1(i,45);
        Spot2id=table2array(Spot2id);
        Spot2metaDataRow = totaldataspot32(find(totaldataspot32(:,22)==Spot2id),:);
        Spot2metaDataRow = Spot2metaDataRow(1,:);
        storage{i} =  Spot2metaDataRow;
    end
    
    Spot2metaDataAll = vertcat(storage{:});
    Spot2metaDataAll = Spot2metaDataAll(:,24:45); %Old (:,14:22)
    Spot2metaDataAll = array2table(Spot2metaDataAll);
    % Renames columns headings
    Spot2metaDataAll.Properties.VariableNames = { 'ForS2_S3IntensityMeanTriC' 'ForS2_S3PositionXTriC' 'ForS2_S3PositionYTriC' 'ForS2_S3PositionZTriC' 'ForS2_S3IntensityMaxTriC' 'ForS2_S3IntensityCenterTriC' 'ForS2_S3AreaTriC' 'ForS2_S3VolumeTriC' 'ForS2_S3IntensityMinTriC' 'ForS2_S3ntensityMedianTriC' 'ForS2_S3IntensityStdDevTriC' 'ForS2_S3IntensitySumTriC' 'ForS2_S3IntensitySumOfSquareTriC' 'ForS2_S3IntensityMin1TriC' 'ForS2_S3IntensityMax1TriC' 'ForS2_S3IntensityCenter1TriC' 'ForS2_S3IntensityMean1TriC' 'ForS2_S3IntensityMedian1TriC' 'ForS2_S3IntensityStdDev1TriC' 'ForS2_S3IntensitySum1TriC' 'ForS2_S3IntensitySumOfSquare1TriC' 'ForS2_S3IDTriC'};
    % Combines with prior compiled metadata
    AllDataSpot1 = horzcat(AllDataSpot1, Spot2metaDataAll);
    % Writes output   
    writetable(AllDataSpot1,filename,'Sheet','AllData');
    
    
    % Writes Spot 1 vs Spot 3 data
    Spot3_1Freq = array2table(Spot3_1Freq);
    writetable(Spot3_1Freq,filename,'Sheet','Spot3_1Freq');
    % Writes Spot 2 vs Spot 3 data
    Spot3_2Freq = array2table(Spot3_2Freq);
    writetable(Spot3_2Freq,filename,'Sheet','Spot3_2Freq')
    
    % Performs flip analysis of Spot 1 and Spot 2 versus Spot 3 if specified by user.
    if(~isempty(flip))
    Spot3_1Freqflip = histcounts(NNall3flip,edges);
    Spot3_1Freqflip = rot90(Spot3_1Freqflip,3);
    
    
    Spot3_2Freqflip = histcounts(NNall4flip,edges);
    Spot3_2Freqflip = rot90(Spot3_2Freqflip,3);
    
    
    %Performs flip subtraction
    Spot3_1Freq = table2array (Spot3_1Freq);
    Spot3_1Freqflipsubtraction = (Spot3_1Freq - Spot3_1Freqflip);
    Spot3_1FreqflipsubtractionZeros = max(Spot3_1Freqflipsubtraction,0);   
    
    Spot3_2Freq = table2array(Spot3_2Freq);  
    Spot3_2Freqflipsubtraction = (Spot3_2Freq - Spot3_2Freqflip);
    Spot3_2FreqflipsubtractionZeros = max(Spot3_2Freqflipsubtraction,0);   
    
    
    
    % Writes output
    Spot3_1Freqflip = array2table(Spot3_1Freqflip);
    writetable(Spot3_1Freqflip,filename,'Sheet','Spot3_1Freqflip');
    Spot3_2Freqflip = array2table(Spot3_2Freqflip);
    writetable(Spot3_2Freqflip,filename,'Sheet','Spot3_2Freqflip')
    
    Spot3_1Freqflipsubtraction = array2table(Spot3_1Freqflipsubtraction);
    writetable(Spot3_1Freqflipsubtraction,filename,'Sheet','Spot3_1Freqflipsubtraction');
    Spot3_2Freqflipsubtraction = array2table(Spot3_2Freqflipsubtraction);
    writetable(Spot3_2Freqflipsubtraction,filename,'Sheet','Spot3_2Freqflipsubtraction')
    
    Spot3_1FreqflipsubtractionZeros = array2table(Spot3_1FreqflipsubtractionZeros);
    writetable(Spot3_1FreqflipsubtractionZeros,filename,'Sheet','Spot3_1FreqflipsubtractionZeros');
    Spot3_2FreqflipsubtractionZeros = array2table(Spot3_2FreqflipsubtractionZeros);
    writetable(Spot3_2FreqflipsubtractionZeros,filename,'Sheet','Spot3_2FreqflipsubtractionZeros')

    end
    end
     
     end

    

% Function to sort data by mean intensity
function sortdat= getsortedData(rawData)
    [~,idx] = sort(rawData(:,1),'ascend'); % sort just the first column
    sortdat = rawData(idx,:);   % sort the whole matrix using the sort indices
end
% Function to get user defined subset by mean intensity
function subsetdat = getsubsetData(data,interval)
    lims = length(data).* interval/100;
    subsetdat = data(ceil(lims(1))+1:ceil(lims(2)),:);
end
