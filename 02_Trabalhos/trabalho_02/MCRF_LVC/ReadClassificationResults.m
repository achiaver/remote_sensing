%% Database Info
Database    = 'PaviaUniversity';% PaviaUniversity Indian_Pines
PredictionsDir = 'Predictions_MonoCRF';
Results_Dir = ['./Data/' Database '/Results'];  mkdir(Results_Dir);
RawData_Dir = ['./Data/' Database '/RawData'];
TrainMaskFileName  = [RawData_Dir '/Train.bmp'];
TestMaskFileName   = [RawData_Dir '/Test.bmp'];
ShowClassificationMaps = 1;
%% Read Label Image
[TrainMask, colormap_orig]  = imread(TrainMaskFileName);   TrainMask  = double(TrainMask);
[TestMask, ~]               = imread(TestMaskFileName);    TestMask   = double(TestMask);
if ShowClassificationMaps
    figure(),imshow(uint8(TestMask),colormap_orig);
end
LabelImage  = TrainMask + TestMask;     
NClasses = length(unique(LabelImage))-1;
LabelsTesting = LabelImage(:);
LabelsTesting(TestMask(:) == 0) = [];
%% Read Binary Files
PredictionsFolders = dir(PredictionsDir);
for idFolder = 3:length(PredictionsFolders)
    FolderName = PredictionsFolders(idFolder).name;
    MapFileName = [Results_Dir '/' FolderName '.png'];
    MapFileNameResize = [Results_Dir '/' FolderName '_Resize.png'];
    MetricsFileName = [Results_Dir '/' FolderName '.mat'];
    
    binFileNames = dir([PredictionsDir '/' FolderName '/*.bin']);
    binFileNames = {binFileNames.name};
    LabelsClassification = [];
    
    for idFile = 1:length(binFileNames)
        binFileName = [PredictionsDir '/' FolderName '/' cell2mat(binFileNames(idFile))];
        fileID = fopen(binFileName,'r');
        LabelsClassificationTile = fread(fileID,'float');
        LabelsClassificationTile = LabelsClassificationTile + 1;
        LabelsClassification = [LabelsClassification; LabelsClassificationTile];
        fclose(fileID);
    end
    
    % Compute Accuracy Metrics -------------------------------------------
    % Confusion Matrix
    Metrics.CM = confusionmat(LabelsTesting,LabelsClassification,'order',1:NClasses);
    
    Metrics.OA = 100*sum(diag(Metrics.CM))/sum(sum(Metrics.CM));% overall accuracy
    tp = diag(Metrics.CM)';% true positives
    fp = sum(Metrics.CM,1) - tp;% false positives
    fn = sum(Metrics.CM,2)' - tp;% false negatives
    
    AvgAcc = 100*diag(Metrics.CM)./sum(Metrics.CM,2);
    Metrics.AA = sum(AvgAcc(~isnan(AvgAcc)))/length(unique(LabelsTesting));% average class accuracy
    
    Metrics.Precision = 100*tp./(tp + fp);% precision
    Metrics.Recall = 100*tp./(tp + fn);% recall
    
    % F1-score and Average F1-score
    Metrics.F1_measure = 2*Metrics.Precision.*Metrics.Recall./(Metrics.Precision + Metrics.Recall);
    Metrics.F1_measure(isnan(Metrics.F1_measure)) = 0;
    Metrics.F1_measure_avg = sum(Metrics.F1_measure)/NClasses;
    
    save(MetricsFileName,'Metrics');
    %---------------------------------------------------------------------
    
    % Save Classification Map ---------------------------------------------
    ClassificationMap = TestMask;
    ClassificationMap(ClassificationMap ~= 0) = LabelsClassification;
    if ShowClassificationMaps
        figure(), imshow(uint8(ClassificationMap), colormap_orig);
    end
    imwrite(uint8(ClassificationMap),colormap_orig,MapFileName);
    imwrite(imresize(uint8(ClassificationMap),8,'nearest'),colormap_orig,MapFileNameResize);
    %----------------------------------------------------------------------    
end
