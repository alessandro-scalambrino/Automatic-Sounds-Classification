function KNN_calculation(allTrainFeatures, allTestFeatures, allLabels, groundTruth, testLabelcoughing, testLabelcrying, testLabelsnoring, ttl, clr)


k = [1 2 3 5 7 8 10 15 20 50 100 200];

% compute the recogniton rate with the use of k

disp(['–––––––––––––––––––––––––––––––– ', ttl, ' ––––––––––––––––––––––––––––––––'])
fprintf('\n')
rate = [];
for kk=1:length(k)
    disp(['Setting up the kNN with the following number of neigbors: ', mat2str(k(kk))])
    Mdl = fitcknn(allTrainFeatures, allLabels', 'NumNeighbors', k(kk));
    
    % test the kNN
    predicted_label = predict(Mdl,allTestFeatures);
    
    % measure the performance
    correct = 0;
    for i=1:length(predicted_label)
        if predicted_label(i)==groundTruth(i)
            correct=correct+1;            
        end
    end
    disp('Its relative recognition rate is:')
    rate(kk) = (correct/length(predicted_label))*100;
    disp(rate((kk)))
end

% plot the performances
figure
plot(k, rate, clr)
title(ttl)

xlabel('k')
ylabel('recognition rate (%)')
grid on

% find the maximum
[a,b]=max(rate);
disp('––––––––––––––––––––––––––––––––– RESULTS –––––––––––––––––––––––––––––––––')
fprintf('\n')
disp(['The maximum recognition rate is ', mat2str(a), ' and it is achieved with a number of ', mat2str(k(b)),' nearest neighbors.'])

% best kvalue
Mdl = fitcknn(allTrainFeatures,allLabels','NumNeighbors',(k(b)));
predicted_label = predict(Mdl,allTestFeatures);


% cough recognition rate
correctcough = find(predicted_label(1:length(testLabelcoughing))==1);
mrr = length(correctcough) / length(testLabelcoughing) * 100

% cry recognition rate
correctcry = find(predicted_label(length(testLabelcoughing)+1:length(testLabelcrying)+length(testLabelcoughing))==2);
wrr = length(correctcry) / length(testLabelcrying) * 100

% snore recognition rate
correctsnore = find(predicted_label(length(testLabelcoughing)+length(testLabelcrying)+1:end)==3);
srr = length(correctsnore) / length(testLabelsnoring) * 100

% cough incorrect recognition rate
incorrectcoughcry = find(predicted_label(1:length(testLabelcoughing))==2);
incorrectcoughsnore = find(predicted_label(1:length(testLabelcoughing))==3);
mxwrr = length(incorrectcoughcry) / length(testLabelcoughing) * 100;
mxsrr = length(incorrectcoughsnore) / length(testLabelcoughing) * 100;

% cry incorrect recognition rate
incorrectcrycough = find(predicted_label(length(testLabelcoughing)+1:length(testLabelcrying)+length(testLabelcoughing))==1);
incorrectcrysnore = find(predicted_label(length(testLabelcoughing)+1:length(testLabelcrying)+length(testLabelcoughing))==3);
wxmrr = length(incorrectcrycough) / length(testLabelcrying) * 100;
wxsrr = length(incorrectcrysnore) / length(testLabelcrying) * 100;

% snore incorrect recognition rate
incorrectsnorecough = find(predicted_label(length(testLabelcrying)+length(testLabelcoughing)+1:end)==1);
incorrectsnorecry = find(predicted_label(length(testLabelcrying)+length(testLabelcoughing)+1:end)==2);
sxmrr = length(incorrectsnorecough) / length(testLabelsnoring) * 100;
sxwrr = length(incorrectsnorecry) / length(testLabelsnoring) * 100;

confusion_matrix = [mrr mxwrr mxsrr; wxmrr wrr wxsrr; sxmrr sxwrr srr]

figure
C = confusionmat(groundTruth, predicted_label)
cm = confusionchart(C, {'cough' 'cry' 'snore'}, 'Title', 'Sound classification', 'RowSummary', 'row-normalized');