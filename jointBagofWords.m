% In the Name of GOD
%*******************

%
% ----------------------------------------------------------- 23 Jan 2014
% ------------- Bag of words technique
% ------------- joint spatio-temporal features 
% ------------- MSR dataset

nClust= 500;   % number of clusters (number of observations in HMM) 

% putting all features of all videos in "data"
% -------------------------------------------
% for all videos of all actions! WOW too many... ;)
listVideos= dir('./jointsFeat/*.mat');
numVideos= numel(listVideos);
    
data= [];      % the whole feature values for all videos of all actions
numFrames= []; % the number of frames per video
for i=1:numVideos
    load(['./jointsFeat/',listVideos(i).name]);
    numFrames= cat(1,numFrames,size(jFeats,2));
    
    data= cat(1,data,jFeats');
end           
% -------------------------------------------
% size(data): (number of frames of all videos)x(399)
size(data)


% clustering features with only with 39 values (1-755)
% ---------------------------------------------------
tempD= []; % features with 78 empty elements= 39 values
indx= [];  % their indexes
 
for i=1:size(data,1)
    if (sum(data(i,:)==1000)==78) % 117-78= 39 values
        tempD= cat(1,tempD,data(i,:));
        indx= cat(1,indx,i);
    end
end
[~,centers]= kmeans(tempD,nClust,'emptyaction','singleton','display','iter');

% calculate distance of each instance to all cluster centers
clustIDX= zeros(size(tempD,1),1);
for i=1:size(tempD,1)
    %fprintf('pixel:%i...\n', i);
    D= zeros(1,nClust);
    for j=1:nClust
        D(1,j) = sum((tempD(i,:)-centers(j,:)).^2);
    end
    [~,clustIDX(i,1)]= min(D);
end
clustIdFinal(indx)= clustIDX; 


% clustering features with 117 values (1-45)
% ----------------------------------------- 
tempD= []; % features with 0 empty elements= 18 values
indx= []; % their indexes
nClust= 45;
for i=1:size(data,1)
    if (sum(data(i,:)==1000)==0) % 117-0=117 values
        tempD= cat(1,tempD,data(i,:));
        indx= cat(1,indx,i);
    end
end
[~,centers]=kmeans(tempD,nClust,'emptyaction','singleton','display','iter');

% calculate distance of each instance to all cluster centers
clustIDX= zeros(size(tempD,1),1);
for i=1:size(tempD,1)
    %fprintf('pixel:%i...\n', i);
    D= zeros(1,nClust);
    for j=1:nClust
        D(1,j) = sum((tempD(i,:)-centers(j,:)).^2);
    end
    [~,clustIDX(i,1)]= min(D);
end
clustIdFinal(indx)= (clustIDX+755);
clear clustIDX indx centers tempD data

% -------------------------------------------------------------------------
numVidAct=[29,28,27,24,28,33,30,27,26,22,32]; % number of videos per action
k=0;
lastNF= 0; % number of frame for the last video
for j=1:length(numVidAct)  %j-th action
    data= cell(1,numVidAct(1,j));
    fprintf('action: %d\n',j);
    
    for i=1:numVidAct(1,j)  %i-th video
        data{i}= clustIdFinal(k+1:k+numFrames(lastNF+i));  % data: observations data
        % numFrames(i): number of frames of video i     
        fprintf('---number of frames: %d, %d-%d\n',numFrames(lastNF+i),k+1,k+numFrames(lastNF+i));
        k= k+numFrames(lastNF+i);
    end
    lastNF= lastNF+numVidAct(1,j);
    
    save(['./HMMfeatures/obs_test_core9_noTrack_',actionList(j,1:3)],'data');
end


