function Features = stFeatureExtraction(fileName, windowLength, stepLength)

% function Features = stFeatureExtraction(signal, fs, win, step)
%
% This function computes basic audio feature sequencies for an audio
% signal, on a short-term basis.
%
% ARGUMENTS:
%  - signal:    the audio signal
%  - fs:        the sampling frequency
%  - win:       short-term window size (in seconds)
%  - step:      short-term step (in seconds)
%
% RETURNS:
%  - Features: a [MxN] matrix, where M is the number of features and N is
%  the total number of short-term windows. Each line of the matrix
%  corresponds to a seperate feature sequence
%
% (c) 2014 T. Giannakopoulos, A. Pikrakis

[signal,fs] = audioread(fileName);

% if STEREO ...
if (size(signal,2)>1), signal = (sum(signal,2)/2); % convert to MONO
end

% convert window length and step from seconds to samples:
[M,numOfFrames] = windowize(signal, round(windowLength*fs), round(stepLength*fs));

% number of features to be computed:
numOfFeatures = 21;
Features = zeros(numOfFeatures, numOfFrames);
Ham = window(@hamming, round(windowLength*fs)); 
mfccParams = feature_mfccs_init(round(windowLength*fs), fs);

for i=1:numOfFrames % for each frame
    % get current frame:
    frame = M(:,i);
    frame  = frame .* Ham;
    frameFFT = getDFT(frame, fs);
    
    if (sum(abs(frame))>eps)
        % compute time-domain features:
        Features(1,i) = feature_zcr(frame);
        Features(2,i) = feature_energy(frame);
        Features(3,i) = feature_energy_entropy(frame, 10);

        % compute freq-domain features: 
        if (i==1) frameFFTPrev = frameFFT; end;
        [Features(4,i) Features(5,i)] = ...
            feature_spectral_centroid(frameFFT, fs);
        Features(6,i) = feature_spectral_entropy(frameFFT, 10);
        Features(7,i) = feature_spectral_flux(frameFFT, frameFFTPrev);
        Features(8,i) = feature_spectral_rolloff(frameFFT, 0.90);
        MFCCs = feature_mfccs(frameFFT, mfccParams);
        Features(9:21,i)  = MFCCs;
    else
        Features(:,i) = zeros(numOfFeatures, 1);
    end    
    frameFFTPrev = frameFFT;
end
Features(21, :) = medfilt1(Features(21, :), 3);
