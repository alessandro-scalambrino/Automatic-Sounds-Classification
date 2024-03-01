function mfccs = extractMFCCs(filename, windowLength, stepLength)
[y,fs]=audioread(filename);
if size(y,2)==2, y=sum(y,2)/2; end
[M,nf]=windowize(y,windowLength*fs,stepLength*fs);
Ham = window(@hamming, windowLength*fs);
mfccParams = feature_mfccs_init(windowLength*fs, fs);
mfccs = zeros(13, nf);
for i=1:nf
    frame = M(:,i);
    frame  = frame .* Ham;
    frameFFT = getDFT(frame, fs);
    mfccs(1:13,i) = feature_mfccs(frameFFT, mfccParams);
end