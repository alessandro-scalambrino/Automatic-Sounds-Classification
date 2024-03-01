clear; clc

% ––––––––––––––––––––––––––––– IASPROJECT –––––––––––––––––––––––––––––––
%----AUTHOR: ALESSANDRO-SCALAMBRINO-923216-------

% ---MANAGE PATH, DIRECTORIES, FILES--
addpath(genpath(pwd))
fprintf('Extracting features from the audio files...\n\n')

coughingFile = dir([pwd,'/Coughing/*.ogg']);
cryingFile = dir([pwd,'/Crying/*.ogg']);
snoringFile = dir([pwd,'/Snoring/*.ogg']);

F = [coughingFile; cryingFile; snoringFile];

% ---WINDOWS/STEP LENGHT---
windowLength = 0.025;
stepLength = 0.01;

% ---FEATURES EXTRACTION---
% ---INITIALIZING FEATURES VECTORS---
allFeatures = [];
coughingFeatures = [];
cryingFeatures = [];
snoringFeatures = [];

% ---EXTRACTION---
for i=1:3
    for j=1:40
        Features = stFeatureExtraction(F(i+j-1).name, windowLength, stepLength);
        allFeatures = [allFeatures Features];
        if i == 1; coughingFeatures = [coughingFeatures Features]; end
        if i == 2; cryingFeatures = [cryingFeatures Features]; end
        if i == 3; snoringFeatures = [snoringFeatures Features]; end
    end
end


% ---FEATURES NORMALIZATION----
mn = mean(allFeatures);
st = std(allFeatures);
allFeaturesNorm =  (allFeatures - repmat(mn,size(allFeatures,1),1))./repmat(st,size(allFeatures,1),1);


% ---PCA---
warning('off', 'stats:pca:ColRankDefX')

[coeff,score,latent,tsquared,explained] = pca(allFeaturesNorm');

disp('The following results are the values of the variance of each coefficient:')

explained

counter = 0;

for p=1:length(explained)
    if explained(p) > 80
        counter = counter + 1;
    end
end

disp(['The number of coefficients offering at least 80% of variance is ', mat2str(counter)])

fprintf('\n\n')

% ---PCA PLOTTING---
S=[]; % size of each point, empty for all equal
C=[repmat([1 0 0],length(coughingFeatures),1); repmat([0 1 0],length(cryingFeatures),1); repmat([0 0 1],length(snoringFeatures),1)];
scatter3(score(:,1),score(:,2),score(:,3),S,C,'filled')
axis equal
title('PCA')


% –––––––––––––––––––––––TRAIN/TEST DATASET –––––––––––––––––––––––

trainPerc = 0.70;
testPerc = 1 - trainPerc;

coughingTrain = coughingFile(1:length(coughingFile)*trainPerc);
cryingTrain = cryingFile(1:length(cryingFile)*trainPerc);
snoringTrain = snoringFile(1:length(snoringFile)*trainPerc);

FTR = [coughingTrain cryingTrain snoringTrain];

coughingTest = coughingFile(length(coughingFile)*trainPerc + 1:length(coughingFile));
cryingTest = cryingFile(length(cryingFile)*trainPerc + 1:length(cryingFile));
snoringTest = snoringFile(length(snoringFile)*trainPerc + 1:length(snoringFile));

FTE = [coughingTest cryingTest, snoringTest];

% ---INITIALIZING TIME/FREQ EMPTY VECTORS FOR TRAINING---

% ----TRAINING-----

coughingTrainTimeFeatures = [];
cryingTrainTimeFeatures = [];
snoringTrainTimeFeatures = [];

coughingTrainFreqFeatures = [];
cryingTrainFreqFeatures = [];
snoringTrainFreqFeatures = [];


% ----TRAINING-----
allTrainTimeFeatures = [];
allTrainFreqFeatures = [];

for a=1:3
    for b=1:28
        TrainFeatures = stFeatureExtraction(FTR(a+b-1).name, windowLength, stepLength);
        allTrainTimeFeatures = [allTrainTimeFeatures TrainFeatures(1:3, :)];
        allTrainFreqFeatures = [allTrainFreqFeatures TrainFeatures(4:21, :)];
        if a == 1
            coughingTrainTimeFeatures = [coughingTrainTimeFeatures TrainFeatures(1:3, :)];
            coughingTrainFreqFeatures = [coughingTrainFreqFeatures TrainFeatures(4:21, :)];
        end
        if a == 2
            cryingTrainTimeFeatures = [cryingTrainTimeFeatures TrainFeatures(1:3, :)];
            cryingTrainFreqFeatures = [cryingTrainFreqFeatures TrainFeatures(4:21, :)];
        end
        if a == 3
            snoringTrainTimeFeatures = [snoringTrainTimeFeatures TrainFeatures(1:3, :)];
            snoringTrainFreqFeatures = [snoringTrainFreqFeatures TrainFeatures(4:21, :)];
        end
    end
end

coughingTrainFeatures = [coughingTrainTimeFeatures; coughingTrainFreqFeatures];
cryingTrainFeatures = [cryingTrainTimeFeatures; cryingTrainFreqFeatures];
snoringTrainFeatures = [snoringTrainTimeFeatures; snoringTrainFreqFeatures];

allTrainFeatures = [coughingTrainFeatures cryingTrainFeatures snoringTrainFeatures];


% ---INITIALIZING TIME/FREQ EMPTY VECTORS FOR TESTING---

% ----TESTING-----

coughingTestTimeFeatures = [];
cryingTestTimeFeatures = [];
snoringTestTimeFeatures = [];

coughingTestFreqFeatures = [];
cryingTestFreqFeatures = [];
snoringTestFreqFeatures = [];

allTestTimeFeatures = [];
allTestFreqFeatures = [];

for x=1:3
    for y=1:28
        TestFeatures = stFeatureExtraction(FTE(x+y-1).name, windowLength, stepLength);
        allTestTimeFeatures = [allTestTimeFeatures TestFeatures(1:3, :)];
        allTestFreqFeatures = [allTestFreqFeatures TestFeatures(4:21, :)];
        if x == 1
            coughingTestTimeFeatures = [coughingTestTimeFeatures TestFeatures(1:3, :)];
            coughingTestFreqFeatures = [coughingTestFreqFeatures TestFeatures(4:21, :)];
        end
        if x == 2
            cryingTestTimeFeatures = [cryingTestTimeFeatures TestFeatures(1:3, :)];
            cryingTestFreqFeatures = [cryingTestFreqFeatures TestFeatures(4:21, :)];
        end
        if x == 3
            snoringTestTimeFeatures = [snoringTestTimeFeatures TestFeatures(1:3, :)];
            snoringTestFreqFeatures = [snoringTestFreqFeatures TestFeatures(4:21, :)];
        end
    end
end

coughingTestFeatures = [coughingTestTimeFeatures; coughingTestFreqFeatures];
cryingTestFeatures = [cryingTestTimeFeatures; cryingTestFreqFeatures];
snoringTestFeatures = [snoringTestTimeFeatures; snoringTestFreqFeatures];

allTestFeatures = [coughingTestFeatures cryingTestFeatures snoringTestFeatures];


% –––––––––––––––––––––– TRAIN DATASET NORMALISATION –––––––––––––––––––––

% normalisation in time domain of TRAIN data
allTrainTimeFeatures = allTrainTimeFeatures';
mnTime = mean(allTrainTimeFeatures);
stTime = std(allTrainTimeFeatures);
allTrainTimeFeatures =  (allTrainTimeFeatures - repmat(mnTime,size(allTrainTimeFeatures,1),1))./repmat(stTime,size(allTrainTimeFeatures,1),1);

% normalisation in frequency domain of TRAIN data
allTrainFreqFeatures = allTrainFreqFeatures';
mnFreq = mean(allTrainFreqFeatures);
stFreq = std(allTrainFreqFeatures);
allTrainFreqFeatures =  (allTrainFreqFeatures - repmat(mnFreq,size(allTrainFreqFeatures,1),1))./repmat(stFreq,size(allTrainFreqFeatures,1),1);

% normalisation of both time domain and frequency domain of TRAIN data
allTrainFeatures = allTrainFeatures';
mnAll = mean(allTrainFeatures);
stAll = std(allTrainFeatures);
allTrainFeatures =  (allTrainFeatures - repmat(mnAll,size(allTrainFeatures,1),1))./repmat(stAll,size(allTrainFeatures,1),1);

% ––––––––––––––––––––––– TEST DATASET NORMALISATION –––––––––––––––––––––

% normalisation in time domain of TEST data
allTestTimeFeatures = allTestTimeFeatures';
allTestTimeFeatures =  (allTestTimeFeatures - repmat(mnTime,size(allTestTimeFeatures,1),1))./repmat(stTime,size(allTestTimeFeatures,1),1);

% normalisation in frequency domain of TEST data
allTestFreqFeatures = allTestFreqFeatures';
allTestFreqFeatures =  (allTestFreqFeatures - repmat(mnFreq,size(allTestFreqFeatures,1),1))./repmat(stFreq,size(allTestFreqFeatures,1),1);

% normalisation of both time domain and frequency domain of TEST data
allTestFeatures = allTestFeatures';
allTestFeatures =  (allTestFeatures - repmat(mnAll,size(allTestFeatures,1),1))./repmat(stAll,size(allTestFeatures,1),1);


% –––––––––––––––––––––––––– TRAIN/TEST LABELS ––––––––––––––––––––––––––

% TRAIN
labelcoughingTime = repmat(1,length(coughingTrainTimeFeatures),1);
labelcryingTime = repmat(2,length(cryingTrainTimeFeatures),1);
labelsnoringTime = repmat(3, length(snoringTrainTimeFeatures),1);
allTimeLabels = [labelcoughingTime; labelcryingTime; labelsnoringTime];

labelcoughingFreq = repmat(1,length(coughingTrainFreqFeatures),1);
labelcryingFreq = repmat(2,length(cryingTrainFreqFeatures),1);
labelsnoringFreq = repmat(3, length(snoringTrainFreqFeatures),1);
allFreqLabels = [labelcoughingFreq; labelcryingFreq; labelsnoringFreq];

labelcoughingAll = repmat(1,length(coughingTrainFeatures),1);
labelcryingAll = repmat(2,length(cryingTrainFeatures),1);
labelsnoringAll = repmat(3, length(snoringTrainFeatures),1);
allLabels = [labelcoughingAll; labelcryingAll; labelsnoringAll];

% ––––––––––––––––––––––––––– APPLY TEST LABELS ––––––––––––––––––––––––––

testLabelcoughingTime = repmat(1,length(coughingTestTimeFeatures),1);
testLabelcryingTime = repmat(2,length(cryingTestTimeFeatures),1);
testLabelsnoringTime = repmat(3, length(snoringTestTimeFeatures),1);
groundTruthTime = [testLabelcoughingTime; testLabelcryingTime; testLabelsnoringTime];

testLabelcoughingFreq = repmat(1,length(coughingTestFreqFeatures),1);
testLabelcryingFreq = repmat(2,length(cryingTestFreqFeatures),1);
testLabelsnoringFreq = repmat(3, length(snoringTestFreqFeatures),1);
groundTruthFreq = [testLabelcoughingFreq; testLabelcryingFreq; testLabelsnoringFreq];

testLabelcoughingAll = repmat(1,length(coughingTestFeatures),1);
testLabelcryingAll = repmat(2,length(cryingTestFeatures),1);
testLabelsnoringAll = repmat(3, length(snoringTestFeatures),1);
allGroundTruth = [testLabelcoughingAll; testLabelcryingAll; testLabelsnoringAll];


% –––––––––––––––––––––––––––– KNN –––––––––––––––––––––––––––
fprintf('––––––––––––––––––––––––––– COMPUTING THE KNN ––––––––––––––––––––––––––––\n\n')

fprintf('Computing the recognition rate using the following values for k: 1, 2, 3, 5, 7, 8, 10, 15, 20, 50, 100, 200...\n\n')

%TIME
KNN_calculation(allTrainTimeFeatures, allTestTimeFeatures, allTimeLabels, groundTruthTime, testLabelcoughingTime, testLabelcryingTime, testLabelsnoringTime, 'TIME DOMAIN', '-pm')
%FREQ
KNN_calculation(allTrainFreqFeatures, allTestFreqFeatures, allFreqLabels, groundTruthFreq, testLabelcoughingFreq, testLabelcryingFreq, testLabelsnoringFreq, 'FREQUENCY DOMAIN', '-pg')
%ALL-TOGETHER
KNN_calculation(allTrainFeatures, allTestFeatures, allLabels, allGroundTruth, testLabelcoughingAll, testLabelcryingAll, testLabelsnoringAll, 'TIME AND FREQUENCY DOMAIN', '-pr')
