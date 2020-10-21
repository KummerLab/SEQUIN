%% README
% How to use this function: call using compileExceldocs(pathname), where
% pathname is the path of the folder where all the files to be compiled
% together are located (ex:
% 'C:\Users\gangollim\Documents\Andrew\scripts\data output\test data to
% compile')
%data gets saved as an excel sheet named compiled NNdata in a subfolder
%called compileddata
function compileExceldocs_SEQUINv3(pathname,row1start,row1end,row2start,row2end,flip_exist)
    t0 = tic;
    if(isempty(pathname))
        disp('Specify the path where files are located')
    end
    
    files = dir([pathname '\*xlsx']);
    mkdir([pathname '\compileddata']);
    outputfname = [pathname '\compileddata\compiled_nearest_neighborFlipSub.xlsx'];
    labcol = 1;
    
    for fi = 1:length(files)
        disp(['Processing ' files(fi).name])
        [~,fname,~] = fileparts([pathname '\' files(fi).name]);
        
        
       NormalFreq = xlsread([pathname '\' files(fi).name],'Normal Frequency');
       
       
       
        
        labref = idx2A1(labcol);
        nameref = idx2A1(labcol);
        numref = idx2A1(labcol);
        % [datanum2, datalabel] = xlsread([pathname '\' files(fi).name],'AU2:AU2');
         datalabel =  extractBetween(files(fi).name,row1start,row1end);
         datalabel2 =  extractBetween(files(fi).name,row2start,row2end);
     
         
      xlswrite(outputfname,datalabel,'Normal Frequency',[labref num2str(1)]);
      xlswrite(outputfname,datalabel2,'Normal Frequency',[nameref num2str(2)]); 
      xlswrite(outputfname,NormalFreq(1:end,1:1),'Normal Frequency',[numref num2str(3)]);
       
      
      if flip_exist == 1
      FlippedFreq = xlsread([pathname '\' files(fi).name],'Flipped Frequency');
      Flipsubtraction = xlsread([pathname '\' files(fi).name],'Flip Subtraction Frequency');
      FlipsubtractionZeros = xlsread([pathname '\' files(fi).name],'Flip Sub Zeros Frequency');
        
      xlswrite(outputfname,datalabel,'Flipped Frequency',[nameref num2str(1)]); 
      xlswrite(outputfname,datalabel2,'Flipped Frequency',[nameref num2str(2)]); 
      xlswrite(outputfname,FlippedFreq(1:end,1:1),'Flipped Frequency',[numref num2str(3)]);
       
      xlswrite(outputfname,datalabel,'Flip Subtraction Frequency',[nameref num2str(1)]); 
      xlswrite(outputfname,datalabel2,'Flip Subtraction Frequency',[nameref num2str(2)]); 
      xlswrite(outputfname,Flipsubtraction(1:end,1:1),'Flip Subtraction Frequency',[numref num2str(3)]);
       
      xlswrite(outputfname,datalabel,'Flip Sub Zeros Frequency',[nameref num2str(1)]);
      xlswrite(outputfname,datalabel2,'Flip Sub Zeros Frequency',[nameref num2str(2)]); 
      xlswrite(outputfname,FlipsubtractionZeros(1:end,1:1),'Flip Sub Zeros Frequency',[numref num2str(3)]);
      end
% % %        
        labcol = labcol+ 1; 
    end
    t = toc(t0);
    disp(['Finished processing all files. Time elapsed: ' num2str(t) ' seconds'] )
end

function a1String = idx2A1(idx)

    if idx > 16384 
        warning('Column number is larger than Excel limit.'); 
    end

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

    a1String = ''; 
    while idx > 0 
        idx2 = rem(idx, 26); 
        if idx2 == 0 
            idx2 = 26; 
            idx = idx - 26; 
        end 
        a1String = [alphabet(idx2) a1String]; 
        idx = floor(idx / 26); 
    end
end