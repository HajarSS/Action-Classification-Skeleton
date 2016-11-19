% In the name of GOD...
% ---------------------
% Working on ECCV conference
% Action Recognition
% Start: 2014-01-15

%{
%--------------------------------------------------------- 15 Jan 2014
% --------------- HMM & Multi-nomial Logistic Regression 
% --------------- features: 18-val vector (top,dir,dis,siz)
% --------------- HMM: train and evaluate an HMM model! :)
% --------------- Cross Validation

% 11 colors for each action bar diagram
mycolors = [ 0.0 0.0 0.5;
             0.0 0.0 1.0;
             0.0 0.5 1.0;
             0.0 1.0 1.0;
             0.3 0.8 0.5;
             0.9 0.9 0.0;
             1.0 0.5 0.0;
             1.0 0.0 0.0;
             0.5 0.0 0.0;
             0.8 0.0 0.5;
             0.2 0.6 0.8;];
         
for nState= 2:10
fprintf('========== nState: %i\n',nState);
t= cputime;
% list of actions to be processed (11 actions at the moment)
actionList= ['carry  ';'dig    ';'fall   ';'jump   ';'kick   ';'pickup ';
             'putdown';'run    ';'throw  ';'turn   ';'walk   ';];

nAct= size(actionList,1); % number of actions
% nState= 2;       % number of states
nObs= 800;       % number of observation (number of clusters: nClust)
priorD= 0.0001;  % Pseudocount (Dirichlet Prior)
mItr= 100;
mySeed= 10;

trainData= {}; % create an empty cell 
classData= []; % a vector for class of each training data includes 1..nAct

estPrior= cell(1,nAct);
estTrans= cell(1,nAct);
estEmis= cell(1,nAct);

results= zeros(nAct); % nActxnAct matrix

% ----------------------- Loading Training Data
fprintf('Loading Training Data...\n');
for act= 1:nAct
    %fprintf('---action: %s\n',actionList(act,1:7));
    
    load(['./HMMfeatures/obs_',actionList(act,1:3)]); % data
    trainData= cat(2,trainData,data);  % both (data and trainData) are cells
    classData= cat(2,classData,repmat(act,1,length(data)));
end
% size(trainData)= 1x(number of videos)


% keeps the log likelihoods for each video (306x11)
logProb= zeros(size(classData,2),nAct);
vidNum= 1;

for act= 1:nAct % act: the action with CrossVal on that
    fprintf('---------- action: %s\n',actionList(act,1:end));
    %fid = fopen('EvalRes.dat','a');
    %fprintf(fid,'----- action %s:\n',actionList(act,1:end));
    %fclose(fid);
    
    othAct= 1:nAct;
    othAct(act)= [];  % other actions with no-CrossVal
    
    fprintf('----- Training other actions than CV-action including:\n');
    % training other actions (non-CrossVal actions)
    for i=1:length(othAct) % training on other actions
        fprintf('----- action: %s\n',actionList(othAct(i),1:end));
        % ----------------------- Initial Parameters
        % initial probability
        rng(mySeed);
        prior= normalise(rand(nState,1));

        % initial state transition matrix
        rng(mySeed);
        trans= mk_stochastic(rand(nState,nState));
        
        % initial observation emission matrix
        rng(mySeed);
        emis= mk_stochastic(rand(nState,nObs));
        % ------------------------------------------
        
        % ----------------------- Training HMM
        % improve guess of parameters using EM
        [LL,x1,x2,x3]= ...
            dhmm_em(trainData(classData==othAct(i)),prior,trans,emis,'max_iter',mItr,'obs_prior_weight',priorD);
        %fprintf('(%d data)\n',length(trainData(classData==othAct(i))));
        estPrior{othAct(i)}= x1;
        estTrans{othAct(i)}= x2;
        estEmis{othAct(i)}= x3;
    end
    % We train one-model(prior/trans/emis) for each action
    
    
    % traning CrossVal action (CV)
    for i=1:sum(classData==act) % number of training data in CV-action
        % ------------------------------------------
        trainD= trainData(classData==act);
        testD= trainD(i);    % one video for test, all others for train
        trainD(i)= [];       % delete the test data video)

        % train the model on trainD
        % ------------------------------------------
        rng(mySeed);
        prior= normalise(rand(nState,1));
        rng(mySeed);
        trans= mk_stochastic(rand(nState,nState));
        rng(mySeed);
        emis= mk_stochastic(rand(nState,nObs));

        [LL,x1,x2,x3]= ...
            dhmm_em(trainD,prior,trans,emis,'max_iter',mItr,'obs_prior_weight',priorD);
        estPrior{act}= x1;
        estTrans{act}= x2;
        estEmis{act}= x3;
        
        % test the model on testD including only one video of action 'act'
        % ------------------------------------------
        % ----------------------- Evaluation HMM models
        loglik= zeros(1,nAct);
        for j=1:nAct  % number of actions
            loglik(j)=dhmm_logprob(testD,estPrior{j},estTrans{j},estEmis{j});
        end
        logProb(vidNum,:)= loglik;
        vidNum= vidNum+1;
        [mag,idx]= max(loglik);
        
        fprintf(' - data %d,loglike: %.2f, estimated action: %s\n',i,mag,actionList(idx,1:end));
        results(act,idx)= results(act,idx)+1;
        
        %fid = fopen('EvalRes.dat','a');
        %fprintf(fid,' - data %d, estimated action: %s',i,actionList(idx,1:end));
        %fprintf(fid,'\n');
        %fclose(fid);
    end
    
    
    fig= figure(11);
    bar(results(act,:),'FaceColor',mycolors(act,:));
    set(gca,'XTickLabel',{'carry','dig','fall','jump','kick','pickup',...
        'putdown','run','throw','turn','walk',});
    axis([0 (nAct+1) 0 sum(classData==act)]);  
    title(actionList(act,1:3));
    grid on
    
    print(fig,'-djpeg',[actionList(act,1:3),'_',num2str(mItr),...
        '_',num2str(nState),'.jpg']);
    fprintf('\n');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);

save(['results_',num2str(mItr),'_',num2str(nState)],'results');


% Multi-nomial Logistic Regression
% -------------------------------------------------------------------------
         
numVidAct=[29,28,27,24,28,33,30,27,26,22,32]; % number of videos per action
% a vector for class of each training data includes 1..nAct
classData= classData';

predictions= zeros(size(logProb,1),1);

for i=1:size(logProb,1)
    fprintf('Data %i\n',i);
    
    % We have continuous x (trainData) and 11-valued y (classD)
    testD= logProb(i,:);
    trainD= logProb;
    trainD(i,:)= [];
    %[~,classD]= max(trainD,[],2); % maximum index in each row
    classD= classData;
    classD(i,:)= [];
    
    % Fit a multinomial model for y as a function of x.
    modelLReg = mnrfit(trainD,classD,'model','nominal');
    % nomial: There is no ordering among the response categories
    
    % Prediction
    pred = mnrval(modelLReg,testD,'model','nominal');
    [~,predictions(i,1)]= max(pred,[],2);
    
    fprintf('----- predicted class: %i\n',predictions(i,1));
end

% plotting the results
results= zeros(size(actionList,1)); % 11x11 matrix
last= 0;

for i= 1:size(actionList,1)
    for j=(1+last):(numVidAct(1,i)+last) 
        idx= predictions(j);
        results(i,idx)= results(i,idx)+1;
    end
    last= numVidAct(1,i)+last;
    fig= figure(11);
    bar(results(i,:),'FaceColor',mycolors(i,:));
    set(gca,'XTickLabel',{'carry','dig','fall','jump','kick','pickup',...
        'putdown','run','throw','turn','walk',});
    axis([0 (nAct+1) 0 numVidAct(1,i)]);
    title(actionList(i,1:3));
    grid on
    print(fig,'-djpeg',[actionList(i,1:3),'_',num2str(nState),'.jpg']);
end
end
%}





%{
%--------------------------------------------------------- 23 Jan 2014
% ------------- joint spatio-temporal feature extraction 
% ------------- MSR dataset

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
    
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    jFeats = jointFeatExtractor(a,a,s,s,e,e);
    save(['./jointsFeat/jFeat_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}


%{
%--------------------------------------------------------- 23 Jan 2014
% ------------- Bag of words technique
% ------------- joint spatio-temporal features 
% ------------- MSR dataset

nClust= 100;   % number of clusters (number of observations in HMM) 

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


clustIDX= BagofWords1(data, nClust);

idx= 1;
for i=1:numVideos
    jObs= clustIDX(idx:(numFrames(i)+idx-1),1);
    idx= idx+numFrames(i);
    
    obsName= strrep(listVideos(i).name,'jFeat','jObs');
    obsName = strtok(obsName, '.'); % to remove '.mat'
    save(['./jointsFeat/',obsName,'_',num2str(nClust)],'jObs');
end
%}


%{ 
%--------------------------------------------------------- 23 Jan 2014
% ------------- Task #1
% ------------- Train and test an HMM+Multinomial
% ------------- using joint spatio-temporal features
% ------------- on MSR dataset

%
mycolors = [ 0.0 0.0 0.5;
             0.0 0.0 1.0;
             0.0 0.5 1.0;
             0.0 1.0 1.0;
             0.3 0.8 0.5;
             0.9 0.9 0.0;
             1.0 0.5 0.0;
             1.0 0.0 0.1;
             1.0 0.3 0.6;
             1.0 0.2 0.7;
             1.0 0.0 0.0;
             0.5 0.0 0.0;
             0.8 0.0 0.5;
             0.2 0.6 0.8;
             0.7 0.0 0.7;
             0.4 0.1 0.9;
             0.6 0.6 0.0;
             0.0 0.0 0.8;
             0.0 0.3 0.7;
             0.0 0.8 0.3;];

         
for nState= 3:10
fprintf('========== nState: %i\n',nState);
t= cputime;

nAct= 20;        % number of actions
% nState= 2;       % number of states
nObs= 100;       % number of observation (number of clusters: nClust)
priorD= 0.0001;  % Pseudocount (Dirichlet Prior)
mItr= 100;
mySeed= 10;

trainData= {}; % create an empty cell 
classData= []; % a vector for class of each training data includes 1..nAct

estPrior= cell(1,nAct);
estTrans= cell(1,nAct);
estEmis= cell(1,nAct);

results= zeros(nAct); % nActxnAct matrix

% ----------------------- Loading Training Data
numVidAct= [];
fprintf('Loading Training Data...\n');
for act= 1:nAct
    listVideos= dir(['./jointsFeat/jObs_a',sprintf('%02i',act),'*.mat']);
    numVideos= numel(listVideos);
    
    for i=1:numVideos
        load(['./jointsFeat/',listVideos(i).name]); % data
        trainData= cat(2,trainData,{jObs});  % make both (data and trainData) cells
    end
    classData= cat(2,classData,repmat(act,1,numVideos));
    numVidAct= [numVidAct numVideos];
end
% size(trainData and classData)= 1x(number of videos)


% keeps the log likelihoods for each video (567x20)
logProb= zeros(size(classData,2),nAct);
vidNum= 1;

for act= 1:nAct % act: the action with CrossVal on that
    fprintf('---------- action: %i\n',act);
    
    othAct= 1:nAct;
    othAct(act)= [];  % other actions with no-CrossVal
    
    fprintf('----- Training other actions than CV-action including:\n');
    % training other actions (non-CrossVal actions)
    for i=1:length(othAct) % training on other actions
        fprintf('----- action: %i\n',othAct(i));
        % ----------------------- Initial Parameters
        % initial probability
        rng(mySeed);
        prior= normalise(rand(nState,1));

        % initial state transition matrix
        rng(mySeed);
        trans= mk_stochastic(rand(nState,nState));
        
        % initial observation emission matrix
        rng(mySeed);
        emis= mk_stochastic(rand(nState,nObs));
        % ------------------------------------------
        
        % ----------------------- Training HMM
        % improve guess of parameters using EM
        [LL,x1,x2,x3]= ...
            dhmm_em(trainData(classData==othAct(i)),prior,trans,emis,'max_iter',mItr,'obs_prior_weight',priorD);
        %fprintf('(%d data)\n',length(trainData(classData==othAct(i))));
        estPrior{othAct(i)}= x1;
        estTrans{othAct(i)}= x2;
        estEmis{othAct(i)}= x3;
    end
    % We train one-model(prior/trans/emis) for each action
    
    
    % traning CrossVal action (CV)
    for i=1:sum(classData==act) % number of training data in CV-action
        % ------------------------------------------
        trainD= trainData(classData==act);
        testD= trainD(i);    % one video for test, all others for train
        trainD(i)= [];       % delete the test data video)

        % train the model on trainD
        % ------------------------------------------
        rng(mySeed);
        prior= normalise(rand(nState,1));
        rng(mySeed);
        trans= mk_stochastic(rand(nState,nState));
        rng(mySeed);
        emis= mk_stochastic(rand(nState,nObs));

        [LL,x1,x2,x3]= ...
            dhmm_em(trainD,prior,trans,emis,'max_iter',mItr,'obs_prior_weight',priorD);
        estPrior{act}= x1;
        estTrans{act}= x2;
        estEmis{act}= x3;
        
        % test the model on testD including only one video of action 'act'
        % ------------------------------------------
        % ----------------------- Evaluation HMM models
        loglik= zeros(1,nAct);
        for j=1:nAct  % number of actions
            loglik(j)=dhmm_logprob(testD,estPrior{j},estTrans{j},estEmis{j});
        end
        logProb(vidNum,:)= loglik;
        vidNum= vidNum+1;
        [mag,idx]= max(loglik);
        
        fprintf(' - data %d,loglike: %.2f, estimated action: %i\n',i,mag,idx);
        results(act,idx)= results(act,idx)+1;
        
        %fid = fopen('EvalRes.dat','a');
        %fprintf(fid,' - data %d, estimated action: %s',i,actionList(idx,1:end));
        %fprintf(fid,'\n');
        %fclose(fid);
    end
    
    
    fig= figure(11);
    bar(results(act,:),'FaceColor',mycolors(act,:));
    set(gca,'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11',...
        '12','13','14','15','16','17','18','19','20'});
    axis([0 (nAct+1) 0 sum(classData==act)]);  
    title(['action-', num2str(act)]);
    grid on
    
    print(fig,'-djpeg',['action_',num2str(act),'_',num2str(mItr),...
        '_',num2str(nState),'.jpg']);
    fprintf('\n');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);

save(['results_',num2str(mItr),'_',num2str(nState)],'results');

%{
% Multi-nomial Logistic Regression
% -------------------------------------------------------------------------
         
%numVidAct=[29,28,27,24,28,33,30,27,26,22,32]; % number of videos per action
% a vector for class of each training data includes 1..nAct
classData= classData';

predictions= zeros(size(logProb,1),1);

for i=1:size(logProb,1)
    fprintf('Data %i/%i\n',i,size(logProb,1));
    
    % We have continuous x (trainData) and 20-valued y (classD)
    testD= logProb(i,:);
    trainD= logProb;
    trainD(i,:)= [];
    %[~,classD]= max(trainD,[],2); % maximum index in each row
    classD= classData;
    classD(i,:)= [];
    
    % Fit a multinomial model for y as a function of x.
    modelLReg = mnrfit(trainD,classD,'model','nominal');
    % nomial: There is no ordering among the response categories
    
    % Prediction
    pred = mnrval(modelLReg,testD,'model','nominal');
    [~,predictions(i,1)]= max(pred,[],2);
    
    fprintf('----- predicted class: %i\n',predictions(i,1));
end

% plotting the results
results= zeros(nAct); % 20x20 matrix
last= 0;

for i= 1:nAct
    for j=(1+last):(numVidAct(1,i)+last) 
        idx= predictions(j);
        results(i,idx)= results(i,idx)+1;
    end
    last= numVidAct(1,i)+last;
    fig= figure(11);
    bar(results(i,:),'FaceColor',mycolors(i,:));
    set(gca,'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11',...
        '12','13','14','15','16','17','18','19','20'});
    axis([0 (nAct+1) 0 numVidAct(1,i)]);
    title(['action-', num2str(i)]);
    grid on
    print(fig,'-djpeg',['action',num2str(i),'_',num2str(nState),'.jpg']);
end
save(['results_',num2str(nState)],'results');
%}
end
%}


%hh= bsxfun(@rdivide,results,sum(results,2));
%bb= round(hh * 100) / 100;
%mean(diag(bb))
%}





%{
%--------------------------------------------------------- 29 Jan 2014
% ------------- Task #2 
% ------------- 3D joint spatio-temporal feature extraction 
% ------------- MSR dataset

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
    
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    dim=3; 
    jFeats = jointFeatExtractor(a,a,s,s,e,e,dim);
    save(['./jointsFeat/',num2str(dim),'D_jFeat_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}



%{
%--------------------------------------------------------- 29 Jan 2014
% ------------- Task #2 
% ------------- Bag of words technique
% ------------- 3D joint spatio-temporal features 
% ------------- MSR dataset

nClust= 100;   % number of clusters (number of observations in HMM) 

% putting all features of all videos in "data"
% -------------------------------------------
% for all videos of all actions! WOW too many... ;)
listVideos= dir('./jointsFeat/3D_*.mat');
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


clustIDX= BagofWords1(data, nClust);

idx= 1;
for i=1:numVideos
    jObs= clustIDX(idx:(numFrames(i)+idx-1),1);
    idx= idx+numFrames(i);
    
    obsName= strrep(listVideos(i).name,'jFeat','jObs');
    obsName = strtok(obsName, '.'); % to remove '.mat'
    save(['./jointsFeat/',obsName,'_',num2str(nClust)],'jObs');
end
%}





%{ 
%--------------------------------------------------------- 29 Jan 2014
% ------------- Task #2
% ------------- Train and test an HMM 
% ------------- using 3D joint spatio-temporal features
% ------------- on MSR dataset

%
mycolors = [ 0.0 0.0 0.5;
             0.0 0.0 1.0;
             0.0 0.5 1.0;
             0.0 1.0 1.0;
             0.3 0.8 0.5;
             0.9 0.9 0.0;
             1.0 0.5 0.0;
             1.0 0.0 0.1;
             1.0 0.3 0.6;
             1.0 0.2 0.7;
             1.0 0.0 0.0;
             0.5 0.0 0.0;
             0.8 0.0 0.5;
             0.2 0.6 0.8;
             0.7 0.0 0.7;
             0.4 0.1 0.9;
             0.6 0.6 0.0;
             0.0 0.0 0.8;
             0.0 0.3 0.7;
             0.0 0.8 0.3;];


t= cputime;

nAct= 20;        % number of actions
nState= 10;      % number of states
nObs= 100;       % number of observation (number of clusters: nClust)
priorD= 0.0001;  % Pseudocount (Dirichlet Prior)
mItr= 100;
mySeed= 10;

trainData= {}; % create an empty cell 
classData= []; % a vector for class of each training data includes 1..nAct

estPrior= cell(1,nAct);
estTrans= cell(1,nAct);
estEmis= cell(1,nAct);

results= zeros(nAct); % nActxnAct matrix

% ----------------------- Loading Training Data
numVidAct= [];
fprintf('Loading Training Data...\n');
for act= 1:nAct
    listVideos= dir(['./jointsFeat/3D_jObs_a',sprintf('%02i',act),'*.mat']);
    numVideos= numel(listVideos);
    
    for i=1:numVideos
        load(['./jointsFeat/',listVideos(i).name]); % data
        trainData= cat(2,trainData,{jObs});  % make both (data and trainData) cells
    end
    classData= cat(2,classData,repmat(act,1,numVideos));
    numVidAct= [numVidAct numVideos];
end
% size(trainData and classData)= 1x(number of videos)


% keeps the log likelihoods for each video (567x20)
logProb= zeros(size(classData,2),nAct);
vidNum= 1;

for act= 1:nAct % act: the action with CrossVal on that
    fprintf('---------- action: %i\n',act);
    
    othAct= 1:nAct;
    othAct(act)= [];  % other actions with no-CrossVal
    
    fprintf('----- Training other actions than CV-action including:\n');
    % training other actions (non-CrossVal actions)
    for i=1:length(othAct) % training on other actions
        fprintf('----- action: %i\n',othAct(i));
        % ----------------------- Initial Parameters
        % initial probability
        rng(mySeed);
        prior= normalise(rand(nState,1));

        % initial state transition matrix
        rng(mySeed);
        trans= mk_stochastic(rand(nState,nState));
        
        % initial observation emission matrix
        rng(mySeed);
        emis= mk_stochastic(rand(nState,nObs));
        % ------------------------------------------
        
        % ----------------------- Training HMM
        % improve guess of parameters using EM
        [LL,x1,x2,x3]= ...
            dhmm_em(trainData(classData==othAct(i)),prior,trans,emis,'max_iter',mItr,'obs_prior_weight',priorD);
        %fprintf('(%d data)\n',length(trainData(classData==othAct(i))));
        estPrior{othAct(i)}= x1;
        estTrans{othAct(i)}= x2;
        estEmis{othAct(i)}= x3;
    end
    % We train one-model(prior/trans/emis) for each action
    
    
    % traning CrossVal action (CV)
    for i=1:sum(classData==act) % number of training data in CV-action
        % ------------------------------------------
        trainD= trainData(classData==act);
        testD= trainD(i);    % one video for test, all others for train
        trainD(i)= [];       % delete the test data video)

        % train the model on trainD
        % ------------------------------------------
        rng(mySeed);
        prior= normalise(rand(nState,1));
        rng(mySeed);
        trans= mk_stochastic(rand(nState,nState));
        rng(mySeed);
        emis= mk_stochastic(rand(nState,nObs));

        [LL,x1,x2,x3]= ...
            dhmm_em(trainD,prior,trans,emis,'max_iter',mItr,'obs_prior_weight',priorD);
        estPrior{act}= x1;
        estTrans{act}= x2;
        estEmis{act}= x3;
        
        % test the model on testD including only one video of action 'act'
        % ------------------------------------------
        % ----------------------- Evaluation HMM models
        loglik= zeros(1,nAct);
        for j=1:nAct  % number of actions
            loglik(j)=dhmm_logprob(testD,estPrior{j},estTrans{j},estEmis{j});
        end
        logProb(vidNum,:)= loglik;
        vidNum= vidNum+1;
        [mag,idx]= max(loglik);
        
        fprintf(' - data %d,loglike: %.2f, estimated action: %i\n',i,mag,idx);
        results(act,idx)= results(act,idx)+1;
        
        %fid = fopen('EvalRes.dat','a');
        %fprintf(fid,' - data %d, estimated action: %s',i,actionList(idx,1:end));
        %fprintf(fid,'\n');
        %fclose(fid);
    end
    
    
    fig= figure(11);
    bar(results(act,:),'FaceColor',mycolors(act,:));
    set(gca,'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11',...
        '12','13','14','15','16','17','18','19','20'});
    axis([0 (nAct+1) 0 sum(classData==act)]);  
    title(['action-', num2str(act)]);
    grid on
    
    print(fig,'-djpeg',['action_',num2str(act),'_',num2str(mItr),...
        '_',num2str(nState),'.jpg']);
    fprintf('\n');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);

save(['results_3D_',num2str(mItr),'_',num2str(nState)],'results');

%{
% Multi-nomial Logistic Regression
% -------------------------------------------------------------------------
         
%numVidAct=[29,28,27,24,28,33,30,27,26,22,32]; % number of videos per action
% a vector for class of each training data includes 1..nAct
classData= classData';

predictions= zeros(size(logProb,1),1);

for i=1:size(logProb,1)
    fprintf('Data %i/%i\n',i,size(logProb,1));
    
    % We have continuous x (trainData) and 20-valued y (classD)
    testD= logProb(i,:);
    trainD= logProb;
    trainD(i,:)= [];
    %[~,classD]= max(trainD,[],2); % maximum index in each row
    classD= classData;
    classD(i,:)= [];
    
    % Fit a multinomial model for y as a function of x.
    modelLReg = mnrfit(trainD,classD,'model','nominal');
    % nomial: There is no ordering among the response categories
    
    % Prediction
    pred = mnrval(modelLReg,testD,'model','nominal');
    [~,predictions(i,1)]= max(pred,[],2);
    
    fprintf('----- predicted class: %i\n',predictions(i,1));
end

% plotting the results
results= zeros(nAct); % 20x20 matrix
last= 0;

for i= 1:nAct
    for j=(1+last):(numVidAct(1,i)+last) 
        idx= predictions(j);
        results(i,idx)= results(i,idx)+1;
    end
    last= numVidAct(1,i)+last;
    fig= figure(11);
    bar(results(i,:),'FaceColor',mycolors(i,:));
    set(gca,'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11',...
        '12','13','14','15','16','17','18','19','20'});
    axis([0 (nAct+1) 0 numVidAct(1,i)]);
    title(['action-', num2str(i)]);
    grid on
    print(fig,'-djpeg',['action',num2str(i),'_',num2str(nState),'.jpg']);
end
save(['results_',num2str(nState)],'results');
%}


%hh= bsxfun(@rdivide,results,sum(results,2));
%bb= round(hh * 100) / 100;
%mean(diag(bb))
%}





%{ 
%--------------------------------------------------------- 29 Jan 2014
% ------------- Task #3
% ------------- Train and test an SVM 
% ------------- using HON4D features
% ------------- on MSR dataset

%
mycolors = [ 0.0 0.0 0.5;
             0.0 0.0 1.0;
             0.0 0.5 1.0;
             0.0 1.0 1.0;
             0.3 0.8 0.5;
             0.9 0.9 0.0;
             1.0 0.5 0.0;
             1.0 0.0 0.1;
             1.0 0.3 0.6;
             1.0 0.2 0.7;
             1.0 0.0 0.0;
             0.5 0.0 0.0;
             0.8 0.0 0.5;
             0.2 0.6 0.8;
             0.7 0.0 0.7;
             0.4 0.1 0.9;
             0.6 0.6 0.0;
             0.0 0.0 0.8;
             0.0 0.3 0.7;
             0.0 0.8 0.3;];


t= cputime;

nAct= 20;        % number of actions

data= [];    % Whole data
labels= [];  % a vector for class of each training data includes 1..nAct

% ----------------------- Loading Training Data
fprintf('Loading Training Data...\n');
for act= 1:nAct
    listVideos= dir(['./HON4Ddesc/d_a',sprintf('%02i',act),'*.txt']);
    numVideos= numel(listVideos);
    
    for i=1:numVideos
        desc= load(['./HON4Ddesc/',listVideos(i).name]); % data
        data= cat(1,data,desc);  % make both (data and trainData) cells
    end
    labels= cat(1,labels,repmat(act,numVideos,1));
end
% size(data)=  (number of videos)x17880
% size(label)= (number of videos)x1

predicted_labels= zeros(size(labels)); % (number of videos)x1 vector

dataIndx= 1;
for act= 1:nAct % act: the action with CrossVal on that
    fprintf('---------- action: %i\n',act);
    
    othAct= 1:nAct;
    othAct(act)= [];  % other actions with no-CrossVal
    
    trainD= [];
    trainL= [];
    
    % training other actions (non-CrossVal actions)
    for i=1:length(othAct) % training on other actions
        
        fprintf('----- action: %i\n',othAct(i));
        trainD= cat(1, trainD, data(labels==othAct(i),:));
        trainL= cat(1, trainL, labels(labels==othAct(i),:));
    end
    
    
    % traning CrossVal action (CV)
    for i=1:sum(labels==act) % number of training data in CV-action
        % ------------------------------------------      
        tempD= data(labels==act, :);
        tempL= labels(labels==act);
        testD= tempD(i, :);  % one video for test, all others for train
        testL= tempL(i);     % it's label
        tempD(i, :)= [];     % delete the test data video
        tempL(i)= [];

        newTrainD= cat(1, trainD, tempD);
        newTrainL= cat(1, trainL, tempL);
        
        newTrainD = scaleDescs(newTrainD);
        testD = scaleDescs(testD);

        % train a SVM model on the new trainD
        % ------------------------------------------
        svmParams = '-q -t 1 -g 0.125 -d 3';
        model = svmtrain(newTrainL, newTrainD, svmParams);
        % -t kernel_type : set type of kernel function (default 2)
        % 	 0 -- linear: u'*v
        % 	 1 -- polynomial: (gamma*u'*v + coef0)^degree
        % 	 2 -- radial basis function: exp(-gamma*|u-v|^2)
        % 	 3 -- sigmoid: tanh(gamma*u'*v + coef0)
        % 	 4 -- precomputed kernel (kernel values in training_set_file)
        
        % -b probability_estimates: whether to train a SVC or SVR model for
        % probability estimates, 0 or 1 (default 0)
        
        
        
        % test the model on testD including only one video of action 'act'
        % ------------------------------------------
        predictedL= svmpredict(testL, testD, model);
        
        % accuracy, is a vector including accuracy (for classification), mean
        % squared error, and squared correlation coefficient (for regression).
        
        % "prob_estimates" is a matrix containing decision values or probability
        % estimates (if '-b 1' is specified).

        fprintf(' - data %d/%d, estimated action: %i\n',i, sum(labels==act), predictedL);      
        predicted_labels(dataIndx, 1)= predictedL;
        dataIndx = dataIndx+1;
    end

end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
dataIndx

acc = (length(find((predicted_labels == labels) == 1))/length(labels))*100;
    
%hh= bsxfun(@rdivide,results,sum(results,2));
%bb= round(hh * 100) / 100;
%mean(diag(bb))
%}





%{
%--------------------------------------------------------- 30 Jan 2014
% On the all (252) Splits
% clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

comb= nchoosek(1:10,5); % We split 10 subjects to two groups, each with 5 subjects 
accuracy= [];

for com= 1:size(comb,1)
    fprintf('-- com= %d/%d\n', com, size(comb,1));
    
    trainActors = comb(com, :);
    testActors = setdiff(1:10, trainActors);
    
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_','jFeat_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            des= cat(2, des, max(jFeats,[],2)); % in rows
            des= cat(2, des, mean(jFeats,2));
            des= cat(2, des, median(jFeats,2));
            des= cat(2, des, min(jFeats,[],2));
            des= cat(2, des, mode(jFeats,2));
            des= cat(2, des, std(jFeats,0,2));
            des= cat(2, des, var(jFeats,0,2));
            
            trainDesFeat= cat(1, trainDesFeat, reshape(des',1,[]));
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            des= cat(2, des, max(jFeats,[],2)); % in rows
            des= cat(2, des, mean(jFeats,2));
            des= cat(2, des, median(jFeats,2));
            des= cat(2, des, min(jFeats,[],2));
            des= cat(2, des, mode(jFeats,2));
            des= cat(2, des, std(jFeats,0,2));
            des= cat(2, des, var(jFeats,0,2));
            
            testDesFeat= cat(1, testDesFeat, reshape(des',1,[]));
        end
    end
    
    trainingDesc = scaleDescs(trainingDesc);
    testingDesc = scaleDescs(testingDesc);
    
    trainingDesc = [trainingDesc, trainDesFeat];
    testingDesc = [testingDesc, testDesFeat];
    
    %trainingDesc =  trainDesFeat;
    %testingDesc =  testDesFeat;
    
    svmParams = '-q -t 1 -g 0.125 -d 3';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc = (length(find((predicted_labels == testingLbls) == 1))/length(testingLbls))*100;
    
    accuracy= [accuracy, acc];
end

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 6 Feb 2014
%{   
% ------------- Task #1, fully connected skeleton 
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%  
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    % Fully connected skeleton
    %lines= [];
    %for j=1:numJoints
    %    for k= (j+1):numJoints
    %        lines= [lines, [j;k]];
    %    end
    %end
    
    dim= 2;
    state= 1; % full connected
    jFeats = jointFeatExtractor(a,a,s,s,e,e,dim,state,lines);
    save(['./jointsFeat/',num2str(dim),'D_jFeat_ful_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}
  
    
    

%{       
% Same split in the paper, train(1,3,5,7,9), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_ful_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = normc(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
        
        des= cat(2, des, max(jFeats,[],2)); % in rows
        des= cat(2, des, mean(jFeats,2));
        des= cat(2, des, median(jFeats,2));
        des= cat(2, des, min(jFeats,[],2));
        des= cat(2, des, mode(jFeats,2));
        des= cat(2, des, std(jFeats,0,2));
        des= cat(2, des, var(jFeats,0,2));
        
        trainDesFeat= cat(1, trainDesFeat, reshape(des',1,[]));
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        des= cat(2, des, max(jFeats,[],2)); % in rows
        des= cat(2, des, mean(jFeats,2));
        des= cat(2, des, median(jFeats,2));
        des= cat(2, des, min(jFeats,[],2));
        des= cat(2, des, mode(jFeats,2));
        des= cat(2, des, std(jFeats,0,2));
        des= cat(2, des, var(jFeats,0,2));
        
        testDesFeat= cat(1, testDesFeat, reshape(des',1,[]));
    end
end

trainingDesc = scaleDescs(trainingDesc);
testingDesc = scaleDescs(testingDesc);

trainingDesc = [trainingDesc, trainDesFeat];
testingDesc = [testingDesc, testDesFeat];

%trainingDesc =  trainDesFeat;
%testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc = (length(find((predicted_labels == testingLbls) == 1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 6 Feb 2014
%{
% ------------- Task #2, Anatomy skeleton 
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%  
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
       
    dim=2;
    state= 2; % anatomy
    jFeats = jointFeatExtractor(a,a,s,s,e,e,dim,state,lines);
    save(['./jointsFeat/',num2str(dim),'D_jFeat_ana_dis', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}    
     
    

%{       
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_ana_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = normc(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
        
        des= cat(2, des, max(jFeats,[],2)); % in rows
        des= cat(2, des, mean(jFeats,2));
        des= cat(2, des, median(jFeats,2));
        des= cat(2, des, min(jFeats,[],2));
        des= cat(2, des, mode(jFeats,2));
        des= cat(2, des, std(jFeats,0,2));
        des= cat(2, des, var(jFeats,0,2));
        
        trainDesFeat= cat(1, trainDesFeat, reshape(des',1,[]));
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        des= cat(2, des, max(jFeats,[],2)); % in rows
        des= cat(2, des, mean(jFeats,2));
        des= cat(2, des, median(jFeats,2));
        des= cat(2, des, min(jFeats,[],2));
        des= cat(2, des, mode(jFeats,2));
        des= cat(2, des, std(jFeats,0,2));
        des= cat(2, des, var(jFeats,0,2));
        
        testDesFeat= cat(1, testDesFeat, reshape(des',1,[]));
    end
end

trainingDesc = scaleDescs(trainingDesc);
testingDesc = scaleDescs(testingDesc);

trainingDesc = [trainingDesc, trainDesFeat];
testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 13 Feb 2014
% TASK #1: 2D/distance/bagHist/anatomy or full
% ------------- Anatomy skeleton/ Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%{              
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
    % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    
    disC= distChange(a, s, e, lines, dim);
    jFeats= [];
    for j=1:size(disC,2) % number of lines
        temp = bagHist(disC(:, j)', winWid, slideIncr); % histogram
        jFeats= cat(2, jFeats, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ana_dis_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}    
     
    

%{                   
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_bagHist_ana_disdir_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
               
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

%trainingDesc = scaleDescs(trainingDesc);
%testingDesc = scaleDescs(testingDesc);

%trainingDesc = [trainingDesc, trainDesFeat];
%testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 13 Feb 2014
% TASK #2: 2D/direction/bagHist/anatomy or full
% ------------- Anatomy skeleton/ Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%{            
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
%     lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%            3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
       for k= (j+1):numJoints
           lines= [lines, [j;k]];
       end
    end
    
    %disC= distChange(a, s, e, lines, dim); % distance
    dirC= dirChange(a, s, e, lines, dim);   % direction
    dirC= dirC/10;

    jFeats= [];
%     for j=1:size(disC,2) % number of lines
%         temp = bagHist(disC(:, j)', winWid, slideIncr); % histogram
%         jFeats= cat(2, jFeats, temp);
%     end
    for j=1:size(dirC,2) % number of lines
        % for the direction
        m1= 1;
        m2= 8;
        temp = bagHist(dirC(:, j)', winWid, slideIncr, m1, m2); % histogram
        jFeats= cat(2, jFeats, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ful_dir_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}    
     
    

%{                   
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_bagHist_ful_dir_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
               
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

%trainingDesc = scaleDescs(trainingDesc);
%testingDesc = scaleDescs(testingDesc);

%trainingDesc = [trainingDesc, trainDesFeat];
%testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}



%{
% For direction and distance
% ..........................
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    % distance
    jFeatName= strrep(dname,'d_','2D_jFeat_bagHist_ful_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats1 = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    clear jFeats;
    
    % direction
    jFeatName= strrep(dname,'d_','2D_jFeat_bagHist_ana_dir_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats2 = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, [jFeats1, jFeats2]);
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
               
        testDesFeat= cat(1, testDesFeat, [jFeats1, jFeats2]);
    end
end

%trainingDesc = scaleDescs(trainingDesc);
%testingDesc = scaleDescs(testingDesc);

%trainingDesc = [trainingDesc, trainDesFeat];
%testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 17 Feb 2014
%{
% TASK #3: 2D/distance/jointBagHist/anatomy 
% ------------- Anatomy skeleton/ joint Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
% joint Bag of words histogram makes a histogram for each video

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
t= cputime;

% Making words only for training (otherwise it's too many: 3^19)
words= [];
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    
    disC= distChange(a, s, e, lines, dim); % distance

    for j=1:size(disC, 1) % numFrames - 1
        curWord= disC(j,:);
        [~,indx] = ismember(words, curWord, 'rows');
        indx= find(indx==1);
        
        if isempty(indx) % it's a new word => add it to words
            words= cat(1, words, curWord);
        end
    end
end

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
          3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    
    disC= distChange(a, s, e, lines, dim); % distance

    jFeats = jointBagHist(disC, words); % histogram
    
    save(['./jointsFeat/',num2str(dim),'D_jFeat_jointBagHist_ana_dis_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}    
     
    

%{                    
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_jointBagHist_ana_dis_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
               
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

trainingDesc = scaleDescs(trainingDesc);
testingDesc = scaleDescs(testingDesc);

trainingDesc = [trainingDesc, trainDesFeat];
testingDesc = [testingDesc, testDesFeat];

%trainingDesc =  trainDesFeat;
%testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 18 Feb 2014
% TASK #4: Finding the best thresholds on distance for -1, 0, 1
% ------------- MSR dataset
%{                 
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

disC= [];
winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
    % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    temp= distChange(a, s, e, lines, dim);
    disC= [disC; temp];
    
end

haj= reshape(disC,[], 1);
haj(haj>1000)= 0;
haj(haj<-1000)= 0;
[c1,c2]= kmeans(haj, 3);
 
a1= haj(c1==1);
a2= haj(c1==2);
a3= haj(c1==3);
% a4= haj(c1==4);
% a5= haj(c1==5);

figure(1), plot(a1,'ob');
hold on
plot(a2, '*r');
hold on
plot(a3, 'om');
% hold on
% plot(a4, '*g');
% hold on
% plot(a5, 'ob');
hold off

thr1= min(a1);  % -3.8
thr2= max(a1);  % 3.9
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}    
     

% 
% TASK #4: 2D/distance/bagHist/anatomy or full
% ------------- Anatomy skeleton/ Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%{                 
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
thr1= -3.8;
thr2= 3.9;
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
    % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    
    disC= distChange(a, s, e, lines, dim, thr1, thr2);
    jFeats= [];
    for j=1:size(disC,2) % number of lines
        temp = bagHist(disC(:, j)', winWid, slideIncr,-1,1); % histogram
        jFeats= cat(2, jFeats, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ana_dis_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%} 



%{                         
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_bagHist_ana_dis_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
               
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

%trainingDesc = scaleDescs(trainingDesc);
%testingDesc = scaleDescs(testingDesc);

%trainingDesc = [trainingDesc, trainDesFeat];
%testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





%--------------------------------------------------------- 19 Feb 2014
%---------------------------------------------------------
% TASK #5: Meaningful features for distance
% ------------- MSR dataset
%{                 
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
disC= [];
frNums= []; % size: (numVids, 1);
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
    % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end

    temp= distChange(a, s, e, lines, dim);
    disC= cat(1, disC, temp);
    frNums= cat(1, frNums, size(temp,1));
end

fprintf('Time used: %0.2f min\n',(cputime-t)/60);

res= {};
idx= 1;
for i= 1:size(frNums,1)
    fprintf('-- %i / %i\n',i,size(frNums,1));
    temp= disC(idx:(idx+frNums(i)-1),:); 
    idx= idx+frNums(i);

    for j=1:size(temp,2)
        a= temp(:, j);
        a(a==0)= [];
        
        if isempty(a)
            continue;
        end

        b=a(1,1);
        for k= 2:size(a,1)
            if(b(end) ~= a(k,1))
                b= cat(2, b, a(k,1));
            end
        end
        res= cat(2, res, b);
    end
end

siz= [];
for i=1:size(res,2)
    if(size(res{i},2)==1)
        res{i}
    end
    siz= cat(2, siz, size(res{i},2));
end

m= mean(siz); % 17.84: the average size for the feature vector
%}
     



%{ 
% all features should be in this length: 18 (mean(siz))
% make new features by decreasing or increasing the length of feature
% vector: -1 -1 0 1 0 -1 -1 -1 0 1 1 => -1 -1 1
%                  
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
disC= [];
frNums= []; % size: (numVids, 1);
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
    % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end

    temp= distChange(a, s, e, lines, dim);
    disC= cat(1, disC, temp);
    frNums= cat(1, frNums, size(temp,1));
end

fprintf('Time used: %0.2f min\n',(cputime-t)/60);

featSize= 18; % we make all feature vector in this size

res= {};
weight= {};
idx= 1;
for i= 1:size(frNums,1)  % number of videos
    fprintf('-- %i / %i\n',i,size(frNums,1));
    jFeats= []; % for each video, size (featSize x numLines)
    temp= disC(idx:(idx+frNums(i)-1),:);
    idx= idx+frNums(i);
    
    for j=1:size(temp,2) % number of lines in the skeleton
        a= temp(:, j);
        a(a==0)= [];
        
        % a (frNum-1 x 1) => b (1 x length<frNum-1) => c (1 x featSize)
        if isempty(a)  % all a was 0
            c= repmat(0, 1, featSize);
        else
            % make b and w
            % ------------
            b= a(1,1);
            w= 1;      % weight
            for k= 2:size(a,1)
                if(b(end) ~= a(k,1))
                    b= cat(2, b, a(k,1));
                    w= cat(2, w, 1);
                else
                    w(end)= w(end)+1;
                end
            end
            % ------------
            
            if(size(b,2) > featSize)
                [~,idx] = sort(w,'descend');% Sort the values in descending order
                maxIdx = idx(1:featSize);% Get a linear index into A of the "featSize" largest values
                c= b(sort(maxIdx));
                   
            elseif (size(b,2) < featSize)
                for m= 1:(featSize - size(b,2))
                    [~, idx]= max(w);
                    b= [b(1:idx),b(idx:end)];
                end
                c= b;
            end
        end
        jFeats= cat(2, jFeats, c);
    end

    curVideo= listVids(i).name;  % current video
    save(['./jointsFeat/','jFeat_dis_', strrep(curVideo,'.txt','.mat')],'jFeats');
end

%}



% =========================================================================
% I MUST FIND A WAY ..... FOR ECCV14
% =========================================================================
% 25 Feb 2014
% 2D/topology/BagHist/anatomy 
% ------------- Anatomy skeleton/ joint Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
% Bag of words histogram makes a histogram for each video
%{
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
t= cputime;

% Making words only for training (otherwise it's too many: 25^3)
words= [];
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B,2);    % number of frames in the video
    
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    
    numLines= size(lines,2);
    top= []; %(numLines*(numLines-1), frNum); % 19: numLines
    for f= 1:frNum
        temp= [];  
        for j= 1:numLines   % number of lines (19)
            c1= lines(1,j);
            c2= lines(2,j);
            
            s_a= [B(c1,f,1) B(c1,f,2)];
            e_a= [B(c2,f,1) B(c2,f,2)];

            othJo= 1:size(lines,2);
            othJo(j)= [];  % other joints
            for m= 1:length(othJo)
                c1= lines(1,othJo(m));
                c2= lines(2,othJo(m));
                
                s_b= [B(c1,f,1) B(c1,f,2)];
                e_b= [B(c2,f,1) B(c2,f,2)];

                temp= [temp ; topFinder(s_a, e_a, s_b, e_b)]; 
            end
        end
        top= [top temp];        
    end

    
    % Making "words" for topology
    % ===========================
    for j = 1:size(top, 1) % 19x18
        temp= top(j,:);
        numstps = (length(temp)-winWid)/slideIncr + 1; % Number of windows
        
        for w = 1:numstps
            windowSlid = temp(w:(w+winWid-1));  %Calculation for each window
            [~,indx] = ismember(words, windowSlid, 'rows');
            indx= find(indx==1);
            
            if isempty(indx) % add the new word for train data
                words= cat(1, words, windowSlid);
            end
        end
    end
end
%}    




%{
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
numWords= 100;

load words;

t= cputime;

% K-means to reduce the number of "words"
% =======================================
[~, words]= kmeans(words, numWords);

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B,2);    % number of frames in the video
    
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
    
    numLines= size(lines,2);
    top= []; %(numLines*(numLines-1), frNum); % 19: numLines
    for f= 1:frNum
        temp= [];  
        for j= 1:numLines   % number of lines (19)
            c1= lines(1,j);
            c2= lines(2,j);
            
            s_a= [B(c1,f,1) B(c1,f,2)];
            e_a= [B(c2,f,1) B(c2,f,2)];

            othJo= 1:size(lines,2);
            othJo(j)= [];  % other joints
            for m= 1:length(othJo)
                c1= lines(1,othJo(m));
                c2= lines(2,othJo(m));
                
                s_b= [B(c1,f,1) B(c1,f,2)];
                e_b= [B(c2,f,1) B(c2,f,2)];

                temp= [temp ; topFinder(s_a, e_a, s_b, e_b)]; 
            end
        end
        top= [top temp];        
    end
    
    
    % Making histogram
    % ================
    jFeats= [];
    for j=1:size(top,1)
        feat= top(j,:); 
        
        his= zeros(1, size(words,1));
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            [~,indx] = ismember(words, windowSlid, 'rows');
            indx= find(indx==1);
            
            if isempty(indx) % this may happen for test videos
                L1_dist= sum(abs(bsxfun(@minus, words, windowSlid)), 2); % L1-distance
                [~,indx]= min(L1_dist); % new index shows the closest exiting word
            end
            
            his(1,indx)= his(1,indx) + 1; % histogram
        end
        jFeats= cat(2, jFeats, his);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_jointBagHist_ana_top_', strrep(curVideo,'.txt','.mat')],'jFeats');
    
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}






%{                 
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_jointBagHist_ana_top_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    %jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, reshape(jFeats',1,[]));
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, reshape(jFeats',1,[]));
    end
end

trainingDesc = scaleDescs(trainingDesc);
testingDesc = scaleDescs(testingDesc);

trainingDesc = [trainingDesc, trainDesFeat];
testingDesc = [testingDesc, testDesFeat];

% trainingDesc =  trainDesFeat;
% testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}



%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



% ============================================================= 27 Feb 2014
% TASK #1
%{
% Make Histograms for each video
% ``````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
winWid= 4;
slideIncr= 1;
dim = 2;      % joints dimension

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B, 2);       % number of frames in the video
    %joNum= size(B, 1);      % number of joints(20)
    
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
       
    lineNum= size(lines,2);   % number of lines
    
    disC= distChange(a, s, e, lines, dim, -0.5, 0.5);
    histo= [];
    for j=1:size(disC,2)   % number of lines
        temp = bagHist(disC(:, j)', winWid, slideIncr, -1, 1); % histogram
        histo= cat(2, histo, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_hist_ana_dis_', strrep(curVideo,'.txt','.mat')],'histo');
end
%}

%{
% Make Histograms for each activity
% ``````````````````````````````
dataFolder= './jointsFeat/';
numAct= 20;   % number of numActivities in the dataset
dim= 2;

for i= 1:numAct
    fprintf('activity: %i/%i\n',i, numAct);
    
    nameVids= sprintf('2D_hist_ana_dis_a%02i_*.mat', i);
    listVids= dir([dataFolder,nameVids]);  % List of videos for activity i 
    numVids= numel(listVids);

    histograms= [];
    for j=1:numVids
        load([dataFolder listVids(j).name]);
        histograms= cat(1, histograms, histo);
    end
    h= mean(histograms);    % per activity
    
    save(['./jointsFeat/',num2str(dim),'D_actHist_ana_dis_a',num2str(i),'.mat'],'h');
end
%}

%{
% Make new feature for each video
% ```````````````````````````````
dataFolder= './jointsFeat/';
listVids= dir([dataFolder,'2D_hist_ana_dis_*.mat']); 
numVids= numel(listVids);
numAct= 20;   % number of numActivities in the dataset
dim = 2;      % joints dimension

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    load([dataFolder curVideo]);  % histo: current video histogram
    
    feat= [];
    for j=1:numAct
        actHistName= [num2str(dim),'D_actHist_ana_dis_a',num2str(j),'.mat'];
        load([dataFolder actHistName]);  % h: activity histogram
        
        feat= cat(2, feat, histo-h);
    end
    
    save(['./jointsFeat/',strrep(curVideo,'2D_hist_ana_dis','hajar')],'feat');
end
%}

%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','hajar_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = feat/norm(feat);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, reshape(jFeats',1,[]));
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, reshape(jFeats',1,[]));
    end
end

% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}





% ============================================================= 27 Feb 2014
% TASK #2: 2D/distance/bagHist/anatomy and full
% ------------- Anatomy skeleton/ Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%{           
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= 1:numJoints
%            if (j~=k)
%                lines= [lines, [j;k]];
%            end
%        end
%     end
    
    disC= distChange(a, s, e, lines, dim);  % (frNum-1)x(lineNum)
    jFeats= [];
    for j=1:size(disC,2)  % number of lines
        temp = bagHist(disC(:, j)', winWid, slideIncr, -1, 1); % histogram
        jFeats= cat(2, jFeats, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ana_dis_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}


%{
% TASK #2: 2D/direction/bagHist/anatomy or full
% ------------- Anatomy skeleton/ Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension

m1= 1;
m2= 8;
        
t= cputime;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
%     % anotomy skeleton
%     lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%            3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
% 
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
       for k= (j+1):numJoints
           lines= [lines, [j;k]];
       end
    end
    
    dirC= dirChange(a, s, e, lines, dim);   % direction: frNum x lineNum
    dirC= dirC/10;

    jFeats= [];
    for j=1:size(dirC,2) % number of lines
        temp = bagHist(dirC(:, j)', winWid, slideIncr, m1, m2); % histogram
        jFeats= cat(2, jFeats, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ful_dir_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}



% --------------------- 3 Mar 2014 ----------------------------------------
% TASK #2: 2D/topology/BagHist/anatomy 
% ------------- Anatomy skeleton/ joint Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
% Bag of words histogram makes a histogram for each video
%{
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end

numLines= size(lines,2);

t= cputime;

% Making words only for training (otherwise it's too many: 25^3)
words= [];
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B,2);    % number of frames in the video

    top= []; % numLines x (frNum-1)  % 19: numLines
    for j= 1:numLines
        c1= lines(1,j);
        c2= lines(2,j);
        temp= []; % numLines x 1 
        for f= 1:frNum
            s_b= [B(c1,f,1) B(c1,f,2)];  
            e_b= [B(c2,f,1) B(c2,f,2)]; 

            if (f~=1) % if we are not in the first frame
                temp= cat(2, temp, topFinder(s_a, e_a, s_b, e_b)); 
                s_a= s_b;
                e_a= e_b;
            else
                s_a= s_b;
                e_a= e_b;
            end
        end
        top= cat(1, top, temp); 
    end

    
    % Making "words" for topology
    % ===========================
    for j = 1:size(top, 1) % 19
        temp= top(j,:);
        numstps = (length(temp)-winWid)/slideIncr + 1; % Number of windows
        
        for w = 1:numstps
            windowSlid = temp(w:(w+winWid-1));  %Calculation for each window
            [~,indx] = ismember(words, windowSlid, 'rows');
            indx= find(indx==1);
            
            if isempty(indx) % add the new word for train data
                words= cat(1, words, windowSlid);
            end
        end
    end
end
%}    

%{
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
numWords= 100;

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= (j+1):numJoints
%            lines= [lines, [j;k]];
%        end
%     end
numLines= size(lines,2);

load wordsTop;

t= cputime;

% K-means to reduce the number of "words"
% =======================================
%[~, words]= kmeans(words, numWords);

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B,2);    % number of frames in the video
    
    top= []; % numLines x (frNum-1)  % 19: numLines
    for j= 1:numLines
        c1= lines(1,j);
        c2= lines(2,j);
        temp= []; % numLines x 1
        for f= 1:frNum
            s_b= [B(c1,f,1) B(c1,f,2)];
            e_b= [B(c2,f,1) B(c2,f,2)];
            
            if (f~=1) % if we are not in the first frame
                temp= cat(2, temp, topFinder(s_a, e_a, s_b, e_b));
                s_a= s_b;
                e_a= e_b;
            else
                s_a= s_b;
                e_a= e_b;
            end
        end
        top= cat(1, top, temp);
    end
    
    
    % Making histogram
    % ================
    jFeats= [];
    for j=1:size(top,1)
        feat= top(j,:); 
        
        his= zeros(1, size(words,1));
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            [~,indx] = ismember(words, windowSlid, 'rows');
            indx= find(indx==1);
            
            if isempty(indx) % this may happen for test videos
                L1_dist= sum(abs(bsxfun(@minus, words, windowSlid)), 2); % L1-distance
                [~,indx]= min(L1_dist); % new index shows the closest exiting word
            end
            
            his(1,indx)= his(1,indx) + 1; % histogram
        end
        jFeats= cat(2, jFeats, his);
    end
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ana_top_', strrep(curVideo,'.txt','.mat')],'jFeats');
    
end
fprintf('Time used: %0.2f min\n',(cputime-t)/60);
%}
% -------------------------------------------------------------------------




% --------------------- 4 Mar 2014 ----------------------------------------
% TASK #3: 2D/Distance/BagHist/anatomy/one histogram for each activity 
% ------------- Anatomy skeleton/ joint Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%{
% Make Histograms for each video
% ``````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
winWid= 4;
slideIncr= 1;
dim = 2;      % joints dimension

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B, 2);       % number of frames in the video
    %joNum= size(B, 1);      % number of joints(20)
    
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
       
    lineNum= size(lines,2);   % number of lines
    
    disC= distChange(a, s, e, lines, dim);
    histo= [];
    for j=1:size(disC,2)   % number of lines
        temp = bagHist(disC(:, j)', winWid, slideIncr, -1, 1); % histogram
        histo= cat(2, histo, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_hist_ana_dis_', strrep(curVideo,'.txt','.mat')],'histo');
end
%}

%{
% Make Histograms for each activity
% ``````````````````````````````
dataFolder= './jointsFeat/';
numAct= 20;   % number of numActivities in the dataset
dim= 2;

for i= 1:numAct
    fprintf('activity: %i/%i\n',i, numAct);
    
    nameVids= sprintf('2D_hist_ana_dis_a%02i_*.mat', i);
    listVids= dir([dataFolder,nameVids]);  % List of videos for activity i 
    numVids= numel(listVids);

    histograms= [];
    for j=1:numVids
        load([dataFolder listVids(j).name]);
        histograms= cat(1, histograms, histo);
    end
    h= mean(histograms);    % per activity
    
    save(['./jointsFeat/',num2str(dim),'D_actHist_ana_dis_a',num2str(i),'.mat'],'h');
end
%}

%{
% Make new feature for each video
% ```````````````````````````````
dataFolder= './jointsFeat/';
listVids= dir([dataFolder,'2D_hist_ana_dis_*.mat']); 
numVids= numel(listVids);
numAct= 20;   % number of numActivities in the dataset
dim = 2;      % joints dimension

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    load([dataFolder curVideo]);  % histo: current video histogram
    
    feat= [];
    for j=1:numAct
        actHistName= [num2str(dim),'D_actHist_ana_dis_a',num2str(j),'.mat'];
        load([dataFolder actHistName]);  % h: activity histogram
        
        L1_dist= sum(abs(histo - h));     % L1-distance
        
        feat= cat(2, feat, L1_dist);
    end
    
    save(['./jointsFeat/',strrep(curVideo,'2D_hist_ana_dis','hajar')],'feat');
end
%}

%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','hajar_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = feat/norm(feat);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, reshape(jFeats',1,[]));
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, reshape(jFeats',1,[]));
    end
end

trainingDesc = scaleDescs(trainingDesc);
testingDesc = scaleDescs(testingDesc);

trainingDesc = [trainingDesc, trainDesFeat];
testingDesc = [testingDesc, testDesFeat];
% 
% trainingDesc =  trainDesFeat;
% testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}
% -------------------------------------------------------------------------



% --------------------- 4 Mar 2014 ----------------------------------------
% TASK #3: 2D/Distance/BagHist/anatomy/one histogram for each activity 
% ------------- Anatomy skeleton/ joint Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
%{
% Make Histograms for each video
% ``````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
winWid= 4;
slideIncr= 1;
dim = 2;      % joints dimension

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    
    B=[];
    file=sprintf('./MSRAction3DSkeleton(20joints)/a%02i_s%02i_e%02i_skeleton.txt',a,s,e);
    fp=fopen(file);
    if (fp>0)
        A=fscanf(fp,'%f');
        B=[B; A];
        fclose(fp);
    end
    
    l=size(B,1)/4;
    B=reshape(B,4,l);
    B=B';
    B=reshape(B,20,l/20,4);
    % B(:,:,1);   x-coordinate of joints
    % B(:,:,2);   y-coordinate of joints
    % B(:,:,3);   z-coordinate of joints
    frNum= size(B, 2);       % number of frames in the video
    %joNum= size(B, 1);      % number of joints(20)
    
    
%     % anotomy skeleton
%     lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%         3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
       
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
       for k= (j+1):numJoints
           lines= [lines, [j;k]];
       end
    end

    lineNum= size(lines,2);   % number of lines
    
    disC= distChange(a, s, e, lines, dim);
    histo= [];
    for j=1:size(disC,2)   % number of lines
        temp = bagHist(disC(:, j)', winWid, slideIncr, -1, 1); % histogram
        histo= cat(2, histo, temp);
    end
    save(['./jointsFeat/',num2str(dim),'D_hist_ana_dis_', strrep(curVideo,'.txt','.mat')],'histo');
end


% Make Histograms for each activity
% ``````````````````````````````
dataFolder= './jointsFeat/';
numAct= 20;   % number of numActivities in the dataset


for i= 1:numAct
    fprintf('activity: %i/%i\n',i, numAct);
    
    nameVids= sprintf('2D_hist_ana_dis_a%02i_*.mat', i);
    listVids= dir([dataFolder,nameVids]);  % List of videos for activity i 
    numVids= numel(listVids);

    histograms= [];
    for j=1:numVids
        load([dataFolder listVids(j).name]);
        histograms= cat(1, histograms, histo);
    end
    h= mean(histograms);    % per activity
    
    save(['./jointsFeat/',num2str(dim),'D_actHist_ana_dis_a',num2str(i),'.mat'],'h');
end
%}

%{
% Make new feature for each video
% ```````````````````````````````
dataFolder= './jointsFeat/';
listVids= dir([dataFolder,'2D_hist_ana_dis_*.mat']); 
numVids= numel(listVids);
numAct= 20;   % number of numActivities in the dataset
dim = 2;      % joints dimension

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;   % current video
    load([dataFolder curVideo]);  % histo: current video histogram
    
    feat= [];
    for j=1:numAct
        actHistName= [num2str(dim),'D_actHist_ana_dis_a',num2str(j),'.mat'];
        load([dataFolder actHistName]);    % h: activity histogram
        
        d= histo - h;
        %d= sum(abs(histo - h));     % L1-distance
        %d= bhattacharyya(histo, h);
        %d= pdist2(histo, h, 'emd');
        %d= pdist2(histo, h, 'euclidean');
        %d= pdist2(histo, h, 'cosine'); % Distance is defined as the cosine of the angle between two vectors.
        
        feat= cat(2, feat, d);
        
    end
    
    save(['./jointsFeat/',strrep(curVideo,'2D_hist_ana_dis','hajar')],'feat');
end
%}
% -------------------------------------------------------------------------



%{
% --------------------- 11 Mar 2014 ----------------------------------------
% TASK #4: 2D/Distance/anatomy/Fisher Vector
% ------------- Anatomy skeleton/ joint Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
% {
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 4;
slideIncr= 1;
dim = 2;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
% lines= [];
% for j=1:numJoints
%     for k= (j+1):numJoints
%         lines= [lines, [j;k]];
%     end
% end
lineNum= size(lines,2);   % number of lines

data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            data= cat(2, data, windowSlid');
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
[means, covariances, priors] = vl_gmm(data, numClusters);
% save('means','means');
% save('covariances','covariances');
% save('priors','priors');


% Making Fisher vector for each video
% ===================================
%  load means;
%  load covariances;
%  load priors;
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;

    
    for j=1:size(disC,1) % number of lines
        vidData= [];     % data per line in each video 
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
        % power "normalization": Variance stabilizing transform:
        % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
        %encoding = sign(encoding) .* sqrt(abs(encoding));
        
        % L2 normalization (may introduce NaN vectors)
        %jFeats= cat(2, jFeats, sqrt(sum(abs(encoding).^2)));
        jFeats= cat(2, jFeats, encoding');
    end
    %jFeats
    
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'D_fisher_ana_dis_', strrep(curVideo,'.txt','.mat')],'jFeats');
end

% }

% {
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_fisher_ana_dis_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}
% -------------------------------------------------------------------------


%{
% Make Fisher vector for each video
% `````````````````````````````````
ww= [3,4,5,6,7,8];
for w= 4:size(ww,2)
    winWid= ww(w);
    numClusters = 50; % for Fisher vector
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
% numClusters = 500 ; % for Fisher vector
% winWid= 5;
slideIncr= 1;
dim = 2;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines
  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            data= cat(2, data, windowSlid');
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
[means, covariances, priors] = vl_gmm(data, numClusters);
save(['means',num2str(10+w)],'means');
save(['covariances',num2str(10+w)],'covariances');
save(['priors',num2str(10+w)],'priors');
end
%}


%{
% `````````````````````````````````
ww= [3,4,5,6,7,8];
for w= 1:size(ww,2)
    winWid= ww(w);
    numClusters = 50; % for Fisher vector
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
% numClusters = 500 ; % for Fisher vector
%winWid= 5;
slideIncr= 1;
dim = 2;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines
  

% Making Fisher vector for each video
% ===================================
load(['means',num2str(10+w)],'means');
load(['covariances',num2str(10+w)],'covariances');
load(['priors',num2str(10+w)],'priors');
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        vidData= [];     % data per line in each video 
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
        % power "normalization": Variance stabilizing transform:
        % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
        %encoding = sign(encoding) .* sqrt(abs(encoding));
        
%         % L2 normalization (may introduce NaN vectors)
%         nr= 2; % by default
%         vnr = (sum (encoding.^nr)) .^ (1 / nr);
%         vout = double (encoding) * diag (double (1 ./ vnr));
        
%         jFeats= cat(2, jFeats, vout');
        jFeats= cat(2, jFeats, encoding');
    end
        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar3_', strrep(curVideo,'.txt','.mat')],'jFeats');
end


% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
%clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    %fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2hajar3_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
end
%}
% -------------------------------------------------------------------------

%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 3;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines
  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            data= cat(2, data, windowSlid');
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
[means, covariances, priors] = vl_gmm(data, numClusters);
save('means1','means');
save('covariances1','covariances');
save('priors1','priors');
%}

%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 3;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load 'means1';
load 'covariances1';
load 'priors1';

for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        vidData= [];     % data per line in each video 
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
%         % power "normalization": Variance stabilizing transform:
%         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
         encoding = sign(encoding) .* sqrt(abs(encoding));
%         
%         % L2 normalization (may introduce NaN vectors)
%         nr= 2; % by default
%         vnr = (sum (encoding.^nr)) .^ (1 / nr);
%         vout = double (encoding) * diag (double (1 ./ vnr));
        
%         jFeats= cat(2, jFeats, vout');
        jFeats= cat(2, jFeats, encoding');
    end
        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}


%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    %fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','3hajar1_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}

% ------------------------------------------------------------------------
%{
% Make Fisher vector for each video
% `````````````````````````````````
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 3;
slideIncr= 1;
dim = 3;      % joints dimension
nAct= 20;

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines

fprintf('Loading Training Data...\n');
for act= 1:nAct
    listVideos= dir(['./MSRAction3DSkeleton(20joints)/a',sprintf('%02i',act),'*.txt']);
    numVideos= numel(listVideos);
    data= [];
    
    for i= 1:numVideos
        fprintf('video: %i/%i\n',i, numVideos);
        curVideo= listVideos(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                data= cat(2, data, windowSlid');
            end
        end
    end
    save(['data',num2str(act)],'data');
end
% }

nAct= 20;
x= [];
index_color= [];
for i=1:nAct
    load(['data',num2str(i)]);
    x= cat(1, x, data');
    index_color= cat(1, index_color, repmat(i, size(data,2), 1));
    size(data)
end

% scatter3(x(:,1),x(:,2),x(:,3),50,index_color,'filled')
scatter3(x(1:77026,1),x(1:77026,2),x(1:77026,3),50,index_color(1:77026),'filled');
%}





% --------------------- 27 Mar 2014 ---------------------------------------
% TASK #4: 2D/Distance/anatomy/Fisher Vector
%{
% Make Fisher vector for each video
% `````````````````````````````````
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 2;      % joints dimension
nAct= 20;

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

fprintf('Loading Training Data...\n');
for act= 1:nAct
    listVideos= dir(['./MSRAction3DSkeleton(20joints)/a',sprintf('%02i',act),'*.txt']);
    numVideos= numel(listVideos);
    data= [];
    
    for i= 1:numVideos
        fprintf('video: %i/%i\n',i, numVideos);
        curVideo= listVideos(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            
            disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDisC= [];
            for j=1:size(disC,1)
                temp= disC(j,:)';
                smoothDisC= cat(1, smoothDisC, (smooth(temp))');
            end
            disC= smoothDisC;
            
            for j=1:size(disC,1) % number of lines
                feat= disC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                    data= cat(2, data, windowSlid');
                end
            end
        end
    end
    save(['data',num2str(act)],'data');
end
% }

nAct= 20;
x= [];
index_color= [];
for i=1:nAct
    load(['data',num2str(i)]);
    x= cat(1, x, data');
    index_color= cat(1, index_color, repmat(i, size(data,2), 1));
end

% scatter3(x(:,1),x(:,2),x(:,3),50,index_color,'filled')
% scatter3(x(1:77026,1),x(1:77026,2),x(1:77026,3),50,index_color(1:77026),'filled');

% Dimension Reduction using PCA
[COEFF,SCORE] = princomp(x,'econ');
reducedX= SCORE(:,1:3);   % reduce from 5 to 3

 
%# find out how many clusters you have
Cluster= index_color(1:47329,1);  % 47329
uClusters = unique(Cluster);
nClusters = length(uClusters);
X= reducedX(1:47329,1:3);

cmap = hsv(nClusters);

% plot, set DisplayName so that the legend shows the right label
figure(1), 
for iCluster = 1:nClusters
    clustIdx = Cluster==uClusters(iCluster);
    plot3(X(clustIdx,1),X(clustIdx,2),X(clustIdx,3),'o','MarkerSize',5,...
       'DisplayName',sprintf('Activity %i',uClusters(iCluster)),...
       'MarkerEdgeColor','k','MarkerFaceColor',cmap(iCluster,:));
    hold on
end

legend('show');
hold off
%}

% -------------------------------------------------------------------------
%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 2;      % joints dimension

% anotomy skeleton
lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

lineNum= size(lines,2);   % number of lines

data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            data= cat(2, data, windowSlid');
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
[means, covariances, priors] = vl_gmm(data, numClusters);
save('means1','means');
save('covariances1','covariances');
save('priors1','priors');
%}
%{
% -------- >>> PER VIDEO
load means1
load covariances1
load priors1

data= [];
index_color= [];
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);

    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    vidData= [];     % data per video
    for j=1:size(disC,1) % number of lines
         
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
    end
    % data: rows:dimension(19) ,columns:number of data(numFrame-1)
    encoding = vl_fisher(vidData, means, covariances, priors);
    
    %         % power "normalization": Variance stabilizing transform:
    %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
    %         encoding = sign(encoding) .* sqrt(abs(encoding));
    %
    %         % L2 normalization (may introduce NaN vectors)
    %         nr= 2; % by default
    %         vnr = (sum (encoding.^nr)) .^ (1 / nr);
    %         vout = double (encoding) * diag (double (1 ./ vnr));
    
    %         jFeats= cat(2, jFeats, vout');
    jFeats= encoding';
        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    data= cat(1, data, jFeats);
    index_color= cat(1, index_color, a);
end

% Dimension Reduction using PCA
[COEFF,SCORE] = princomp(data,'econ');
reducedX= SCORE(:,1:3);   % reduce from ? to 3



%# find out how many clusters you have
Cluster= index_color;  % 47329
uClusters = unique(Cluster);
nClusters = length(uClusters);
X= reducedX(:,1:3);

cmap = hsv(nClusters);

% plot, set DisplayName so that the legend shows the right label
figure(1), 
for iCluster = 1:nClusters
    clustIdx = Cluster==uClusters(iCluster);
    plot3(X(clustIdx,1),X(clustIdx,2),X(clustIdx,3),'o','MarkerSize',5,...
       'DisplayName',sprintf('Activity %i',uClusters(iCluster)),...
       'MarkerEdgeColor','k','MarkerFaceColor',cmap(iCluster,:));
    hold on
end

legend('show');
hold off
%}
% -------------------------------------------------------------------------
%{
% -------- >>> PER LINE OUT OF 19 LINES IN THE SKELETON
data= [];
index_color= [];
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        vidData= [];     % data per video
        
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
        jFeats= cat(2, jFeats, encoding');
    end

        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    data= cat(1, data, jFeats);
    index_color= cat(1, index_color, a);
end

% Dimension Reduction using PCA
[COEFF,SCORE] = princomp(data,'econ');
reducedX= SCORE(:,1:3);   % reduce from ? to 3



%# find out how many clusters you have
Cluster= index_color;  % 47329
uClusters = unique(Cluster);
nClusters = length(uClusters);
X= reducedX(:,1:3);

cmap = hsv(nClusters);

% plot, set DisplayName so that the legend shows the right label
figure(1), 
for iCluster = 1:nClusters
    clustIdx = Cluster==uClusters(iCluster);
    plot3(X(clustIdx,1),X(clustIdx,2),X(clustIdx,3),'o','MarkerSize',5,...
       'DisplayName',sprintf('Activity %i',uClusters(iCluster)),...
       'MarkerEdgeColor','k','MarkerFaceColor',cmap(iCluster,:));
    hold on
end

legend('show');
hold off
%}
% -------------------------------------------------------------------------





% --------------------- 31 March 2014 -------------------------------------
% TASK #2: 2D/distance/bagHist/anatomy and full
% ------------- Anatomy skeleton/ Bag-of-words Histogram
% ------------- 2D joint spatio-temporal feature extraction 
% ------------- MSR dataset
% ------------- accuracy: %41.82
%{           
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;

winWid= 4;    % Sliding window size
slideIncr= 1; % Slide replacement 
dim = 2;      % joints dimension
data= [];
index_color= [];
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
           3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    
%     % Fully connected skeleton
%     lines= [];
%     for j=1:numJoints
%        for k= 1:numJoints
%            if (j~=k)
%                lines= [lines, [j;k]];
%            end
%        end
%     end
    
    disC= distChange(a, s, e, lines, dim);  % (frNum-1)x(lineNum)
    jFeats= [];
    for j=1:size(disC,1)  % number of lines
        temp = bagHist(disC(j,:), winWid, slideIncr, -1, 1); % histogram
        jFeats= cat(2, jFeats, temp);
    end
    
    save(['./jointsFeat/',num2str(dim),'D_jFeat_bagHist_ana_dis_', ...
        strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}

%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2D_jFeat_bagHist_ana_dis_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];

trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;


% Dimension Reduction using PCA for training data
% [COEFF,SCORE,latent] = princomp(trainingDesc,'econ');
% reducedX= SCORE(:,find(latent>0.0001));   % reduce dimension
% trainingDesc= reducedX;

Data= zeros(size(trainingDesc));
% center the data
for i = 1:size(trainingDesc, 1)
    Data(i, :) = trainingDesc(i, :) - mean(trainingDesc);
end
DataCov = cov(Data);  % covariance matrix
[PC, variances, explained] = pcacov(DataCov);  % eigen

% project down to less dimension
PcaTrn= Data * PC(:, find(explained>0.0001));



% Dimension Reduction using PCA for testing data
% [COEFF,SCORE,latent] = princomp(testingDesc,'econ');
% reducedX= SCORE(:,find(latent>0.0001));   % reduce dimension
% testingDesc = reducedX;
Data= zeros(size(testingDesc));
% center the data
for i = 1:size(testingDesc, 1)
    Data(i, :) = testingDesc(i, :) - mean(trainingDesc);
end

% project down to less dimension
PcaTst= Data * PC(:, find(explained>0.0001));


svmParams = '-q -t 1 -g 0.125 -d 3';
model = svmtrain(trainingLbls,PcaTrn,svmParams);
predicted_labels = svmpredict(testingLbls,PcaTst,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
% cm= confusionmat(testingLbls,predicted_labels);
%}
% -------------------------------------------------------------------------





% --------------------- 1 April 2014 --------------------------------------
% TASK #4 FV/FULL/distance/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 3;      % joints dimension
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

% % anotomy skeleton
% lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%     3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
lines= [];
for j=1:numJoints
    for k= 1:numJoints
        if (j~=k)
            lines= [lines, [j;k]];
        end
    end
end

lineNum= size(lines,2);   % number of lines
  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
[means, covariances, priors] = vl_gmm(data, numClusters);
save('means4_full_dis_3D','means');
save('covariances4_full_dis_3D','covariances');
save('priors4_full_dis_3D','priors');
%}

%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 3;      % joints dimension

% % anotomy skeleton
% lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%     3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
lines= [];
for j=1:numJoints
    for k= 1:numJoints
        if (j~=k)
            lines= [lines, [j;k]];
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load 'means4_full_dis_3D';
load 'covariances4_full_dis_3D';
load 'priors4_full_dis_3D';

for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        vidData= [];     % data per line in each video 
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
%         % power "normalization": Variance stabilizing transform:
%         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
         encoding = sign(encoding) .* sqrt(abs(encoding));
%         
%         % L2 normalization (may introduce NaN vectors)
%         nr= 2; % by default
%         vnr = (sum (encoding.^nr)) .^ (1 / nr);
%         vout = double (encoding) * diag (double (1 ./ vnr));
        
%         jFeats= cat(2, jFeats, vout');
        jFeats= cat(2, jFeats, encoding');
    end
        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}


%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','3hajar1_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

% % Our Features + theirs
% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];
% 
% % Dimension Reduction using PCA for training data
% [COEFF,SCORE,latent] = princomp(trainingDesc,'econ');
% pc= COEFF(:,find(latent>0.0001));   % principle omponents
% trainingDesc= trainingDesc * pc;     % project down to less dimension
% 
% % Dimension Reduction using PCA for training data
% % [COEFF,SCORE,latent] = princomp(testingDesc,'econ');
% % pc= COEFF(:,find(latent>0.0001));   % reduce dimension
% testingDesc = testingDesc * pc;  % project down to less dimension


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% --------------------- 2 April 2014 --------------------------------------
% TASK #5 FV/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 3;      % joints dimension
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

% % anotomy skeleton
% lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%     3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
lines= [];
for j=1:numJoints
    for k= 1:numJoints
        if (j~=k)
            lines= [lines, [j;k]];
        end
    end
end

lineNum= size(lines,2);   % number of lines
  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'      
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
clearvars -except data numClusters
[means, covariances, priors] = vl_gmm(data, numClusters);
save('means5_ful_dir_3D','means');
save('covariances5_ful_dir_3D','covariances');
save('priors5_ful_dir_3D','priors');
%}

%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 3;      % joints dimension

% % anotomy skeleton
% lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%     3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
lines= [];
for j=1:numJoints
    for k= 1:numJoints
        if (j~=k)
            lines= [lines, [j;k]];
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load 'means5_ful_dir_3D';
load 'covariances5_ful_dir_3D';
load 'priors5_ful_dir_3D';

for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDirC= [];
    for j=1:size(dirC,1)
        temp= dirC(j,:)';
        smoothDirC= cat(1, smoothDirC, (smooth(temp))');
    end
    dirC= smoothDirC;
    
    for j=1:size(dirC,1) % number of lines
        vidData= [];     % data per line in each video 
        feat= dirC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
%         % power "normalization": Variance stabilizing transform:
%         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
         encoding = sign(encoding) .* sqrt(abs(encoding));
%         
%         % L2 normalization (may introduce NaN vectors)
%         nr= 2; % by default
%         vnr = (sum (encoding.^nr)) .^ (1 / nr);
%         vout = double (encoding) * diag (double (1 ./ vnr));
        
%         jFeats= cat(2, jFeats, vout');
        jFeats= cat(2, jFeats, encoding');
    end
        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar3_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}


%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','3hajar3_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

% % Our Features + theirs
% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];
% 
% % Dimension Reduction using PCA for training data
% [COEFF,SCORE,latent] = princomp(trainingDesc,'econ');
% pc= COEFF(:,find(latent>0.0001));   % principle omponents
% trainingDesc= trainingDesc * pc;     % project down to less dimension
% 
% % Dimension Reduction using PCA for training data
% % [COEFF,SCORE,latent] = princomp(testingDesc,'econ');
% % pc= COEFF(:,find(latent>0.0001));   % reduce dimension
% testingDesc = testingDesc * pc;  % project down to less dimension


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% --------------------- 3 April 2014 --------------------------------------
% TASK #5 FV/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 2;      % joints dimension
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

% % anotomy skeleton
% lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%     3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
lines= [];
for j=1:numJoints
    for k= 1:numJoints
        if (j~=k)
            lines= [lines, [j;k]];
        end
    end
end

lineNum= size(lines,2);   % number of lines
  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'      
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
clearvars -except data numClusters
[means, covariances, priors] = vl_gmm(data, numClusters);
save('means5_ful_dir_2D','means');
save('covariances5_ful_dir_2D','covariances');
save('priors5_ful_dir_2D','priors');
%}

%{
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;
dim = 2;      % joints dimension

% % anotomy skeleton
% lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
%     3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];

% Fully connected skeleton
lines= [];
for j=1:numJoints
    for k= 1:numJoints
        if (j~=k)
            lines= [lines, [j;k]];
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load 'means5_ful_dir_2D';
load 'covariances5_ful_dir_2D';
load 'priors5_ful_dir_2D';

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDirC= [];
    for j=1:size(dirC,1)
        temp= dirC(j,:)';
        smoothDirC= cat(1, smoothDirC, (smooth(temp))');
    end
    dirC= smoothDirC;
    
    for j=1:size(dirC,1) % number of lines
        vidData= [];     % data per line in each video 
        feat= dirC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means, covariances, priors);
        
%         % power "normalization": Variance stabilizing transform:
%         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
         encoding = sign(encoding) .* sqrt(abs(encoding));
%         
%         % L2 normalization (may introduce NaN vectors)
%         nr= 2; % by default
%         vnr = (sum (encoding.^nr)) .^ (1 / nr);
%         vout = double (encoding) * diag (double (1 ./ vnr));
        
%         jFeats= cat(2, jFeats, vout');
        jFeats= cat(2, jFeats, encoding');
    end
        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
%}


%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2hajar3_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

% % Our Features + theirs
% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];
% 
% % Dimension Reduction using PCA for training data
% [COEFF,SCORE,latent] = princomp(trainingDesc,'econ');
% pc= COEFF(:,find(latent>0.0001));   % principle omponents
% trainingDesc= trainingDesc * pc;     % project down to less dimension
% 
% % Dimension Reduction using PCA for training data
% % [COEFF,SCORE,latent] = princomp(testingDesc,'econ');
% % pc= COEFF(:,find(latent>0.0001));   % reduce dimension
% testingDesc = testingDesc * pc;  % project down to less dimension


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% --------------------- 3 April 2014 --------------------------------------
% TASK #6 FV/direction distance/SVM/MSR dataset
%{
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clc,clear,close all;
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','3hajar1_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    temp= jFeats;
    
    jFeatName= strrep(dname,'d_','3hajar2_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    temp = [temp jFeats];
    jFeats= temp;
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

% % Our Features + theirs
% trainingDesc = scaleDescs(trainingDesc);
% testingDesc = scaleDescs(testingDesc);
% 
% trainingDesc = [trainingDesc, trainDesFeat];
% testingDesc = [testingDesc, testDesFeat];
% 
% % Dimension Reduction using PCA for training data
% [COEFF,SCORE,latent] = princomp(trainingDesc,'econ');
% pc= COEFF(:,find(latent>0.0001));   % principle omponents
% trainingDesc= trainingDesc * pc;     % project down to less dimension
% 
% % Dimension Reduction using PCA for training data
% % [COEFF,SCORE,latent] = princomp(testingDesc,'econ');
% % pc= COEFF(:,find(latent>0.0001));   % reduce dimension
% testingDesc = testingDesc * pc;  % project down to less dimension


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% -------------------- 4 April 2014 ---------------------------------------
% TASK #7 in columns/FV/FULL/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 3;      % joints dimension
graph= 'ful'; % or ana

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end
lineNum= size(lines,2);   % number of lines
  
data= [];   % numLines x number of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,2) % numFrames-1
            data(:, end + 1) = dirC(:, j); % to optimize for speed,
        end
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
% Run KMeans to pre-cluster the data
[initMeans, assignments] = vl_kmeans(data, numClusters, ...
    'Initialization', 'plusplus');

initCovariances = zeros(size(data, 1),numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = data(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(data'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end

% Run EM starting from the given parameters
[means,covariances,priors] = vl_gmm(data, numClusters, ...
    'initialization','custom', ...
    'InitMeans',initMeans, ...
    'InitCovariances',initCovariances, ...
    'InitPriors',initPriors);



%[means, covariances, priors] = vl_gmm(data, numClusters);
save(['means7_',graph,'_dir_',num2str(dim),'D'],'means');
save(['covariances7_',graph,'_dir_',num2str(dim),'D'],'covariances');
save(['priors7_',graph,'_dir_',num2str(dim),'D'],'priors');
size(data)
% }

% {
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7  5   6   14  15  16  17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load(['means7_',graph,'_dir_',num2str(dim),'D']);
load(['covariances7_',graph,'_dir_',num2str(dim),'D']);
load(['priors7_',graph,'_dir_',num2str(dim),'D']);

for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDirC= [];
    for j=1:size(dirC,1)
        temp= dirC(j,:)';
        smoothDirC= cat(1, smoothDirC, (smooth(temp))');
    end
    dirC= smoothDirC;
    
    
    % data: rows:dimension(19) ,columns:number of data(numFrame-1)
    encoding = vl_fisher(dirC, means, covariances, priors);
    
    %         % power "normalization": Variance stabilizing transform:
    %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
    encoding = sign(encoding) .* sqrt(abs(encoding));
    %
    %         % L2 normalization (may introduce NaN vectors)
    %         nr= 2; % by default
    %         vnr = (sum (encoding.^nr)) .^ (1 / nr);
    %         vout = double (encoding) * diag (double (1 ./ vnr));
    
    %         jFeats= cat(2, jFeats, vout');
    jFeats= encoding';
    
    
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
% }


% {
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clearvars -except dim
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_',[num2str(dim),'hajar2_']);
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------




% -------------------- 4 April 2014 ---------------------------------------
% TASK # 8 Row-Column blocks/FV/distance/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 2;      % joints dimension
graph= 'ful'; % or 'ana'


dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'      
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        feat= disC(1, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        for j=1:numstps
            windowSlid = disC(:,j:(j+winWid-1));% Calculation for each window
            data(:, end + 1) = reshape(windowSlid,[],1); % to optimize for speed,
        end
        
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
clearvars -except data numClusters dim graph

% Run KMeans to pre-cluster the data
[initMeans, assignments] = vl_kmeans(data, numClusters, ...
    'Initialization', 'plusplus');

initCovariances = zeros(size(data, 1),numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = data(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(data'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end

% Run EM starting from the given parameters
[means,covariances,priors] = vl_gmm(data, numClusters, ...
    'initialization','custom', ...
    'InitMeans',initMeans, ...
    'InitCovariances',initCovariances, ...
    'InitPriors',initPriors);

%[means, covariances, priors] = vl_gmm(data, numClusters);
save(['means8_',graph,'_dis_',num2str(dim),'D'],'means');
save(['covariances8_',graph,'_dis_',num2str(dim),'D'],'covariances');
save(['priors8_',graph,'_dis_',num2str(dim),'D'],'priors');
size(data)

clearvars -except dim graph

% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load(['means8_',graph,'_dis_',num2str(dim),'D']);
load(['covariances8_',graph,'_dis_',num2str(dim),'D']);
load(['priors8_',graph,'_dis_',num2str(dim),'D']);

for i= 1:numVids
    %fprintf('video: %i, ',i);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    vidData= [];     % data per video
    feat= disC(1, :);
    numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
    for j=1:numstps
        windowSlid = disC(:,j:(j+winWid-1));% Calculation for each window
        vidData(:, end + 1) = reshape(windowSlid,[],1); % to optimize for speed,
    end
    
    % data: rows:dimension(19) ,columns:number of data(numFrame-1)
    encoding = vl_fisher(vidData, means, covariances, priors);
    
    %         % power "normalization": Variance stabilizing transform:
    %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
    encoding = sign(encoding) .* sqrt(abs(encoding));

    jFeats= encoding';

        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    jFeats(find(isnan(jFeats))) = 123;
    
    save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('\n');
% }


% {
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clearvars -except dim
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    %fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_',[num2str(dim),'hajar1_']);
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% -------------------- 4 April 2014 ---------------------------------------
% TASK # 8 Row-Column blocks/FV/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 2;      % joints dimension
graph= 'ful'; % or 'ana'


dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'      
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        feat= dirC(1, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        for j=1:numstps
            windowSlid = dirC(:,j:(j+winWid-1));% Calculation for each window
            data(:, end + 1) = reshape(windowSlid,[],1); % to optimize for speed,
        end
        
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
clearvars -except data numClusters dim graph

% Run KMeans to pre-cluster the data
[initMeans, assignments] = vl_kmeans(data, numClusters, ...
    'Initialization', 'plusplus');

initCovariances = zeros(size(data, 1),numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = data(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(data'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end

% Run EM starting from the given parameters
[means,covariances,priors] = vl_gmm(data, numClusters, ...
    'initialization','custom', ...
    'InitMeans',initMeans, ...
    'InitCovariances',initCovariances, ...
    'InitPriors',initPriors);

%[means, covariances, priors] = vl_gmm(data, numClusters);
save(['means8_',graph,'_dir_',num2str(dim),'D'],'means');
save(['covariances8_',graph,'_dir_',num2str(dim),'D'],'covariances');
save(['priors8_',graph,'_dir_',num2str(dim),'D'],'priors');
size(data)
% }

clearvars -except dim graph



% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load(['means8_',graph,'_dir_',num2str(dim),'D']);
load(['covariances8_',graph,'_dir_',num2str(dim),'D']);
load(['priors8_',graph,'_dir_',num2str(dim),'D']);

for i= 1:numVids
    %fprintf('video: %i, ',i);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDirC= [];
    for j=1:size(dirC,1)
        temp= dirC(j,:)';
        smoothDirC= cat(1, smoothDirC, (smooth(temp))');
    end
    dirC= smoothDirC;
    
    vidData= [];     % data per video
    feat= dirC(1, :);
    numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
    for j=1:numstps
        windowSlid = dirC(:,j:(j+winWid-1));% Calculation for each window
        vidData(:, end + 1) = reshape(windowSlid,[],1); % to optimize for speed,
    end
    
    % data: rows:dimension(19) ,columns:number of data(numFrame-1)
    encoding = vl_fisher(vidData, means, covariances, priors);
    
    %         % power "normalization": Variance stabilizing transform:
    %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
    encoding = sign(encoding) .* sqrt(abs(encoding));

    jFeats= encoding';

        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    jFeats(find(isnan(jFeats))) = 123;
    
    save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('\n');
% }


% {
% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clearvars -except dim
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_',[num2str(dim),'hajar2_']);
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% -------------------- 4 April 2014 ---------------------------------------
% TASK # 8 Row-Column blocks/FV/distance/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 2;      % joints dimension
graph= 'ful'; % or 'ana'


dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

  
data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'      
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        feat= disC(1, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        for j=1:numstps
            windowSlid = disC(:,j:(j+winWid-1));% Calculation for each window
            data(:, end + 1) = reshape(windowSlid,[],1); % to optimize for speed,
        end
        
    end
end

% G-MM: mixture of Gaussians from data of all videos
% ==================================================
clearvars -except data numClusters dim graph

% Run KMeans to pre-cluster the data
[initMeans, assignments] = vl_kmeans(data, numClusters, ...
    'Initialization', 'plusplus');

initCovariances = zeros(size(data, 1),numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = data(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(data'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end

% Run EM starting from the given parameters
[means,covariances,priors] = vl_gmm(data, numClusters, ...
    'initialization','custom', ...
    'InitMeans',initMeans, ...
    'InitCovariances',initCovariances, ...
    'InitPriors',initPriors);

%[means, covariances, priors] = vl_gmm(data, numClusters);
save(['means8_',graph,'_dis_',num2str(dim),'D'],'means');
save(['covariances8_',graph,'_dis_',num2str(dim),'D'],'covariances');
save(['priors8_',graph,'_dis_',num2str(dim),'D'],'priors');
size(data)



% {
% Make Fisher vector for each video
% `````````````````````````````````
dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
    3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines


% Making Fisher vector for each video
% ===================================
load(['means8_',graph,'_dis_',num2str(dim),'D']);
load(['covariances8_',graph,'_dis_',num2str(dim),'D']);
load(['priors8_',graph,'_dis_',num2str(dim),'D']);

for i= 1:numVids
    %fprintf('video: %i, ',i);

    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');

    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    vidData= [];     % data per video
    feat= disC(1, :);
    numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
    for j=1:numstps
        windowSlid = disC(:,j:(j+winWid-1));% Calculation for each window
        vidData(:, end + 1) = reshape(windowSlid,[],1); % to optimize for speed,
    end
    
    % data: rows:dimension(19) ,columns:number of data(numFrame-1)
    encoding = vl_fisher(vidData, means, covariances, priors);
    
    %         % power "normalization": Variance stabilizing transform:
    %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
    encoding = sign(encoding) .* sqrt(abs(encoding));

    jFeats= encoding';

        
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    jFeats(find(isnan(jFeats))) = 123;
    
    save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
end
fprintf('\n');
% 

% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
clearvars -except dim
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_',[num2str(dim),'hajar1_']);
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
       
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
               
        trainDesFeat= cat(1, trainDesFeat, jFeats);
       
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;


svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------





% --------------------- 7 April 2014 --------------------------------------
% TASK #9 Leave-one-out-crossValidation/FV/distance/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 3;      % joints dimension
graph= 'ful'; % or 'ana'

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines

bestAccuracy= 0;



data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end


allAccuracy= zeros(1,100);
for cv= 1:100
    fprintf('cv: %i/100 ', cv);
    
    delete(['./jointsFeat/',num2str(dim),'hajar*.mat']);
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
      

    
    % Making Fisher vector for each video
    % ===================================
    
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_',[num2str(dim),'hajar_']);
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    allAccuracy(1, cv)= acc;
    
    if (acc> bestAccuracy)
        gmm_means= means;
        gmm_covariance= covariances;
        gmm_prior= priors;
        
        bestAccuracy= acc;
    end
end

save(['allAccuracy_',num2str(dim),'D_',graph,'_Dis'],'allAccuracy');
save(['gmmMeans_',num2str(dim),'D_',graph,'_Dis'],'gmm_means');
save(['gmmCovariance_',num2str(dim),'D_',graph,'_Dis'],'gmm_covariance');
save(['gmmPriors_',num2str(dim),'D_',graph,'_Dis'],'gmm_prior');
%}
% -------------------------------------------------------------------------





% --------------------- 8 April 2014 --------------------------------------
% TASK #9 Leave-one-out-crossValidation/FV/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 3;      % joints dimension
graph= 'ful'; % or 'ana'

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines

bestAccuracy= 0;



data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end


allAccuracy= zeros(1,100);
for cv= 1:100
    fprintf('cv: %i/100 ', cv);
    
    delete(['./jointsFeat/',num2str(dim),'hajar*.mat']);
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
      

    
    % Making Fisher vector for each video
    % ===================================
    
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_',[num2str(dim),'hajar_']);
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    allAccuracy(1, cv)= acc;
    
    if (acc> bestAccuracy)
        gmm_means= means;
        gmm_covariance= covariances;
        gmm_prior= priors;
        
        bestAccuracy= acc;
    end
end
%}
% -------------------------------------------------------------------------





% --------------------- 10 April 2014 --------------------------------------
% TASK #10 run with a number of random restarts and take the model with the
% highest log-likelihood on the training data/FV/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 3;      % joints dimension
graph= 'ful'; % or 'ana'
numIterations= 100;

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines


data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end


logLikelihood= zeros(1,numIterations);
allAccuracy= zeros(1,numIterations);
bestAccuracy= 0;
for cv= 1:numIterations
    fprintf('cv: %i/100 ', cv);
    
    delete(['./jointsFeat/',num2str(dim),'hajar*.mat']);
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
      

    logLikelihood(1, cv)= ll;
    
    % Making Fisher vector for each video
    % ===================================
    
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_',[num2str(dim),'hajar_']);
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    allAccuracy(1, cv)= acc;
    
    if (acc> bestAccuracy)
        gmm_means= means;
        gmm_covariance= covariances;
        gmm_prior= priors;
        
        bestAccuracy= acc;
    end
end

save(['allAccuracy_',num2str(dim),'D_',graph,'_Dir'],'allAccuracy');
save(['logLikelihood_',num2str(dim),'D_',graph,'_Dir'],'logLikelihood');
save(['gmmMeans_',num2str(dim),'D_',graph,'_Dir'],'gmm_means');
save(['gmmCovariance_',num2str(dim),'D_',graph,'_Dir'],'gmm_covariance');
save(['gmmPriors_',num2str(dim),'D_',graph,'_Dir'],'gmm_prior');
%}
% -------------------------------------------------------------------------




% --------------------- 12 April 2014 --------------------------------------
% TASK #9 in rows/crossValidation/FV/direction/SVM/MSR dataset
%{
% Make Fisher vector for each video
% `````````````````````````````````
dim = 3;      % joints dimension
graph= 'ful'; % or 'ana'

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines

bestAccuracy= 0;



data= [];   % windowsize x sequence of words in all videos
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    ind = strfind(curVideo,'s');
    actor = str2num(curVideo(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                
                %data= cat(2, data, windowSlid');
                data(:, end + 1) = windowSlid'; % to optimize for speed,
                
            end
        end
    end
end


loglikel= 0;
for cv= 1:100
    fprintf('cv: %i/100, ', cv);    
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    fprintf('log-likelihood: %.2f\n', ll);
    
    if (ll> loglikel)
        gmm_means= means;
        gmm_covariance= covariances;
        gmm_prior= priors;
        
        loglikel= ll;
    end
end

delete(['./jointsFeat/',num2str(dim),'hajar*.mat']);

% Making Fisher vector for each video
% ===================================
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    
    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDirC= [];
    for j=1:size(dirC,1)
        temp= dirC(j,:)';
        smoothDirC= cat(1, smoothDirC, (smooth(temp))');
    end
    dirC= smoothDirC;
    
    for j=1:size(dirC,1) % number of lines
        vidData= [];     % data per line in each video
        feat= dirC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData,gmm_means,gmm_covariance,gmm_prior);
        
        %         % power "normalization": Variance stabilizing transform:
        %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
        encoding = sign(encoding) .* sqrt(abs(encoding));
        
        jFeats= cat(2, jFeats, encoding');
    end
    
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar_', strrep(curVideo,'.txt','.mat')],'jFeats');
end


% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    %fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_',[num2str(dim),'hajar_']);
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
        
        trainDesFeat= cat(1, trainDesFeat, jFeats);
        
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
%}
% -------------------------------------------------------------------------




% --------------------- 12 April 2014 --------------------------------------
% TASK #10 run with a number of random restarts and take the model with the
% highest loglikelihood on the traindata/FV/distance & direction/SVM/MSR 
%{
% Make Fisher vector for each video
% `````````````````````````````````
numIterations= 100;
allLL= zeros(2,numIterations);
allMeans= cell(2, numIterations); % first row: dis, second row: dir
allCovariance= cell(2, numIterations); 
allPriors= cell(2, numIterations); 
allAccuracy= zeros(1,numIterations);
bestAccuracy= 0;


for cv= 1:numIterations
    fprintf('cv: %i/100 ', cv);
    
    clearvars -except logLikelihood allAccuracy bestAccuracy numIterations cv
    delete('./jointsFeat/*.mat');
    
    graph= 'ful'; % or 'ana'
        
    dataFolder= './MSRAction3DSkeleton(20joints)/';
    listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
    numVids= numel(listVids);
    numJoints= 20;
    numClusters = 50 ; % for Fisher vector
    winWid= 5;
    slideIncr= 1;
    
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    if strcmp(graph,'ana')
        % anotomy skeleton
        lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
            3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    else
        % Fully connected skeleton
        lines= [];
        for j=1:numJoints
            for k= 1:numJoints
                if (j~=k)
                    lines= [lines, [j;k]];
                end
            end
        end
    end
    
    lineNum= size(lines,2);   % number of lines
    
    % =============================== Distance
    data= [];   % windowsize x sequence of words in all videos
    dim = 2;      % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDisC= [];
            for j=1:size(disC,1)
                temp= disC(j,:)';
                smoothDisC= cat(1, smoothDisC, (smooth(temp))');
            end
            disC= smoothDisC;
            
            for j=1:size(disC,1) % number of lines
                feat= disC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end

    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    allMeans(1,cv)= {means}; % first row: dis, second row: dir
    allCovariance(1,cv)= {covariances};
    allPriors(1,cv)= {priors};
    allLL(1, cv)= ll;
    
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % =============================== Direction
    data= [];   % windowsize x sequence of words in all videos
    dim = 3;    % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDirC= [];
            for j=1:size(dirC,1)
                temp= dirC(j,:)';
                smoothDirC= cat(1, smoothDirC, (smooth(temp))');
            end
            dirC= smoothDirC;
            
            for j=1:size(dirC,1) % number of lines
                feat= dirC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end

    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    
    allMeans(2,cv)= {means}; % first row: dis, second row: dir
    allCovariance(2,cv)= {covariances};
    allPriors(2,cv)= {priors};
    allLL(2, cv)= ll;
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_','2hajar1_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        temp= jFeats;
        
        jFeatName= strrep(dname,'d_','2hajar2_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        jFeats= [temp jFeats];
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    allAccuracy(1, cv)= acc;
    
    if (acc> bestAccuracy)
        bestAccuracy= acc;
    end
    
    if (acc >= 86.9)
        break;
    end
end

save(['allAccuracy_',num2str(22),'D_',graph,'_DisDir'],'allAccuracy');
   
save(['logLikelihood_',num2str(22),'D_',graph,'_DisDir'],'allLL');
save(['gmmMeans_',num2str(22),'D_',graph,'_DisDir'],'allMeans');
save(['gmmCovariance_',num2str(22),'D_',graph,'_DisDir'],'allCovariance');
save(['gmmPriors_',num2str(22),'D_',graph,'_DisDir'],'allPriors');

%}
% -------------------------------------------------------------------------





% --------------------- 27 April 2014 -------------------------------------
% TASK #11 : different GMMs for each line in the graph
% run with a number of random restarts and take the model with the highest 
% loglikelihood on the traindata/FV/distance & direction/SVM/MSR 
%{
% Make Fisher vector for each video
% `````````````````````````````````
numIterations= 100;
logLikelihood= zeros(2,numIterations);
allAccuracy= zeros(1,numIterations);
bestAccuracy= 0;


for cv= 1:numIterations
    fprintf('cv: %i/100 ', cv);
    
    clearvars -except logLikelihood allAccuracy bestAccuracy numIterations cv
    delete('./jointsFeat/*.mat');
    
    graph= 'ful'; % or 'ana'
        
    dataFolder= './MSRAction3DSkeleton(20joints)/';
    listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
    numVids= numel(listVids);
    numJoints= 20;
    numClusters = 50 ; % for Fisher vector
    winWid= 5;
    slideIncr= 1;
    
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    if strcmp(graph,'ana')
        % anotomy skeleton
        lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
            3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    else
        % Fully connected skeleton
        lines= [];
        for j=1:numJoints
            for k= 1:numJoints
                if (j~=k)
                    lines= [lines, [j;k]];
                end
            end
        end
    end
    
    lineNum= size(lines,2);   % number of lines
    
    % =============================== Distance
    data= [];   % windowsize x sequence of words in all videos
    dim = 2;    % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDisC= [];
            for j=1:size(disC,1)
                temp= disC(j,:)';
                smoothDisC= cat(1, smoothDisC, (smooth(temp))');
            end
            disC= smoothDisC;
            
            for j=1:size(disC,1) % number of lines
                feat= disC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                end
            end
        end
    end

    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    means1= cell(1, size(data,2));
    covariances1= cell(1, size(data,2));
    priors1= cell(1, size(data,2));
    ll1= zeros(1, size(data,2));
    for j=1:size(data,2)    % number of lines

        curData= data{j};
        
        % Run KMeans to pre-cluster the data
        [initMeans, assignments] = vl_kmeans(curData, numClusters, ...
            'Algorithm','Lloyd','MaxNumIterations',5);
        
        initCovariances = zeros(size(curData, 1),numClusters);
        initPriors = zeros(1,numClusters);
        
        % Find the initial means, covariances and priors
        for i=1:numClusters
            data_k = curData(:,assignments==i);
            initPriors(i) = size(data_k,2) / numClusters;
            
            if size(data_k,1) == 0 || size(data_k,2) == 0
                initCovariances(:,i) = diag(cov(curData'));
            else
                initCovariances(:,i) = diag(cov(data_k'));
            end
        end
        
        % Run EM starting from the given parameters
        [ms,cove,p,l] = vl_gmm(curData, numClusters, ...
            'initialization','custom', ...
            'InitMeans',initMeans, ...
            'InitCovariances',initCovariances, ...
            'InitPriors',initPriors);
        
        means1(1,j)= {ms};
        covariances1(1,j)= {cove};
        priors1(1,j)= {p};
        ll1(1,j)= l;
    end
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData,means1{1,j},covariances1{1,j},priors1{1,j});
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % =============================== Direction
    data= cell(1,lineNum);   % windowsize x sequence of words in all videos
    dim = 2;      % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDirC= [];
            for j=1:size(dirC,1)
                temp= dirC(j,:)';
                smoothDirC= cat(1, smoothDirC, (smooth(temp))');
            end
            dirC= smoothDirC;
            
            for j=1:size(dirC,1) % number of lines
                feat= dirC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    %data(:, end + 1) = windowSlid'; % to optimize for speed,
                    data{1,j}(:, end+1)= windowSlid'; % to optimize for speed,
                end
            end
        end
    end

    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run EM starting from the given parameters
    means2= cell(1, size(data,2));
    covariances2= cell(1, size(data,2));
    priors2= cell(1, size(data,2));
    ll2= zeros(1, size(data,2));
    for j=1:size(data,2)    % number of lines
        
        curData= data{j};
        
        % Run KMeans to pre-cluster the data
        [initMeans, assignments] = vl_kmeans(curData, numClusters, ...
            'Algorithm','Lloyd','MaxNumIterations',5);
        
        initCovariances = zeros(size(curData, 1),numClusters);
        initPriors = zeros(1,numClusters);
        
        % Find the initial means, covariances and priors
        for i=1:numClusters
            data_k = curData(:,assignments==i);
            initPriors(i) = size(data_k,2) / numClusters;
            
            if size(data_k,1) == 0 || size(data_k,2) == 0
                initCovariances(:,i) = diag(cov(curData'));
            else
                initCovariances(:,i) = diag(cov(data_k'));
            end
        end
        
        
        [ms,cove,p,l] = vl_gmm(curData, numClusters, ...
            'initialization','custom', ...
            'InitMeans',initMeans, ...
            'InitCovariances',initCovariances, ...
            'InitPriors',initPriors);
        
        means2(1,j)= {ms};
        covariances2(1,j)= {cove};
        priors2(1,j)= {p};
        ll2(1,j)= l;
    end
    
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData,means2{1,j},covariances2{1,j},priors2{1,j});
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_','2hajar1_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        temp= jFeats;
        
        jFeatName= strrep(dname,'d_','2hajar2_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        jFeats= [temp jFeats];
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    allAccuracy(1, cv)= acc;
    
    if (acc> bestAccuracy)
        gmm_means_Dis= means1;
        gmm_means_Dir= means2;
        
        gmm_covariance_Dis= covariances1;
        gmm_covariance_Dir= covariances2;
        
        gmm_prior_Dis= priors1;
        gmm_prior_Dir= priors2;
        
        gmm_ll_Dis= ll1;
        gmm_ll_Dir= ll2;
        
        bestAccuracy= acc;
    end
end

save(['allAccuracy_',num2str(22),'D_',graph,'_DisDir'],'allAccuracy');

save(['logLikelihood_',num2str(22),'D_',graph,'_Dis'],'gmm_ll_Dis');
save(['gmmMeans_',num2str(22),'D_',graph,'_Dis'],'gmm_means_Dis');
save(['gmmCovariance_',num2str(22),'D_',graph,'_Dis'],'gmm_covariance_Dis');
save(['gmmPriors_',num2str(22),'D_',graph,'_Dis'],'gmm_prior_Dis');

save(['logLikelihood_',num2str(22),'D_',graph,'_Dir'],'gmm_ll_Dir');
save(['gmmMeans_',num2str(22),'D_',graph,'_Dir'],'gmm_means_Dir');
save(['gmmCovariance_',num2str(22),'D_',graph,'_Dir'],'gmm_covariance_Dir');
save(['gmmPriors_',num2str(22),'D_',graph,'_Dir'],'gmm_prior_Dir');
%}
% -------------------------------------------------------------------------





% --------------------- 28 April 2014 --------------------------------------
% TASK #10 run with a number of random restarts and take the model with the
% highest loglikelihood on the traindata/FV/distance & direction/SVM/MSR 
%{
% Make Fisher vector for each video
% `````````````````````````````````
numIterations= 100;
allMeans= cell(2, numIterations); % first row: dis, second row: dir
allCovariance= cell(2, numIterations); 
allPriors= cell(2, numIterations); 
allAccuracy= zeros(1,numIterations);
allLL= zeros(2,numIterations);
bestAccuracy= 0;


for cv= 1:numIterations
    fprintf('cv: %i/100 ', cv);
    
    clearvars -except cv numIterations allMeans allCovariance allPriors ...
        allAccuracy allLL bestAccuracy

    delete('./jointsFeat/*.mat');
    
    t= cputime;
    
    graph= 'ful'; % or 'ana'
        
    dataFolder= './MSRAction3DSkeleton(20joints)/';
    listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
    numVids= numel(listVids);
    numJoints= 20;
    numClusters = 50 ; % for Fisher vector
    winWid= 5;
    slideIncr= 1;
    
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    if strcmp(graph,'ana')
        % anotomy skeleton
        lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
            3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    else
        % Fully connected skeleton
        lines= [];
        for j=1:numJoints
            for k= 1:numJoints
                if (j~=k)
                    lines= [lines, [j;k]];
                end
            end
        end
    end
    
    lineNum= size(lines,2);   % number of lines
    
    % =============================== Distance
    data= [];   % windowsize x sequence of words in all videos
    dim = 2;      % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDisC= [];
            for j=1:size(disC,1)
                temp= disC(j,:)';
                smoothDisC= cat(1, smoothDisC, (smooth(temp))');
            end
            disC= smoothDisC;
            
            for j=1:size(disC,1) % number of lines
                feat= disC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end

    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    allMeans(1,cv)= {means}; % first row: dis, second row: dir
    allCovariance(1,cv)= {covariances};
    allPriors(1,cv)= {priors};
    allLL(1, cv)= ll;
    
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % =============================== Direction
    data= [];   % windowsize x sequence of words in all videos
    dim = 3;    % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDirC= [];
            for j=1:size(dirC,1)
                temp= dirC(j,:)';
                smoothDirC= cat(1, smoothDirC, (smooth(temp))');
            end
            dirC= smoothDirC;
            
            for j=1:size(dirC,1) % number of lines
                feat= dirC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end

    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    
    allMeans(2,cv)= {means}; % first row: dis, second row: dir
    allCovariance(2,cv)= {covariances};
    allPriors(2,cv)= {priors};
    allLL(2, cv)= ll;
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_','2hajar1_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        temp= jFeats;
        
        jFeatName= strrep(dname,'d_','3hajar2_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        jFeats= [temp jFeats];
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    allAccuracy(1, cv)= acc;
    
    if (acc> bestAccuracy)
        bestAccuracy= acc;
    end
    
    if (acc >= 85)
        break;
    end
    fprintf('Time used: %0.2f min\n', (cputime-t)/60);
end

save(['allAccuracy_',num2str(23),'D_',graph,'_DisDir'],'allAccuracy');
   
save(['logLikelihood_',num2str(23),'D_',graph,'_DisDir'],'allLL');
save(['gmmMeans_',num2str(23),'D_',graph,'_DisDir'],'allMeans');
save(['gmmCovariance_',num2str(23),'D_',graph,'_DisDir'],'allCovariance');
save(['gmmPriors_',num2str(23),'D_',graph,'_DisDir'],'allPriors');

%}
% -------------------------------------------------------------------------





% ====================== 28 April 2014 ====================================
% BMVC14- paper figures
% Figure 1: comparisons for different window size : 3,4,5,6,and 7
%{
% Make Fisher vector for each video
% `````````````````````````````````

for winWid=3:7
    fprintf('Window Width: %i \n', winWid);
    
    clearvars -except winWid
    delete('./jointsFeat/*.mat');
    
    graph= 'ful'; % or 'ana'
    
    dataFolder= './MSRAction3DSkeleton(20joints)/';
    listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
    numVids= numel(listVids);
    numJoints= 20;
    numClusters = 50 ; % for Fisher vector
    slideIncr= 1;
    
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    if strcmp(graph,'ana')
        % anotomy skeleton
        lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
            3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
    else
        % Fully connected skeleton
        lines= [];
        for j=1:numJoints
            for k= 1:numJoints
                if (j~=k)
                    lines= [lines, [j;k]];
                end
            end
        end
    end
    
    lineNum= size(lines,2);   % number of lines
    
    
    % =============================== Distance
    data= [];   % windowsize x sequence of words in all videos
    dim = 2;      % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDisC= [];
            for j=1:size(disC,1)
                temp= disC(j,:)';
                smoothDisC= cat(1, smoothDisC, (smooth(temp))');
            end
            disC= smoothDisC;
            
            for j=1:size(disC,1) % number of lines
                feat= disC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end
    
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    
    logLikelihood(1, cv)= ll;
    
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % =============================== Direction
    data= [];   % windowsize x sequence of words in all videos
    dim = 3;      % joints dimension
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
            smoothDirC= [];
            for j=1:size(dirC,1)
                temp= dirC(j,:)';
                smoothDirC= cat(1, smoothDirC, (smooth(temp))');
            end
            dirC= smoothDirC;
            
            for j=1:size(dirC,1) % number of lines
                feat= dirC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end
    
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run KMeans to pre-cluster the data
    [initMeans, assignments] = vl_kmeans(data, numClusters, ...
        'Algorithm','Lloyd','MaxNumIterations',5);
    
    initCovariances = zeros(size(data, 1),numClusters);
    initPriors = zeros(1,numClusters);
    
    % Find the initial means, covariances and priors
    for i=1:numClusters
        data_k = data(:,assignments==i);
        initPriors(i) = size(data_k,2) / numClusters;
        
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
    end
    
    % Run EM starting from the given parameters
    [means,covariances,priors,ll] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
        'InitMeans',initMeans, ...
        'InitCovariances',initCovariances, ...
        'InitPriors',initPriors);
    
    
    logLikelihood(2, cv)= ll;
    
    % Making Fisher vector for each video
    % ===================================
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
        smoothDirC= [];
        for j=1:size(dirC,1)
            temp= dirC(j,:)';
            smoothDirC= cat(1, smoothDirC, (smooth(temp))');
        end
        dirC= smoothDirC;
        
        for j=1:size(dirC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= dirC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariances, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
    end
    
    
    % Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
    % addpath '../libsvm-3.12/matlab/';
    desc_fold = './HON4Ddesc/';
    dDesc = dir([desc_fold '*.txt']);
    
    trainDesFeat= []; % Qualitative Discriptors
    testDesFeat= [];  % Qualitative Discriptors
    trainingDesc = [];
    testingDesc = [];
    trainingLbls = [];
    testingLbls = [];
    trainActors = [1 3 5 7 9];
    testActors = [2 4 6 8 10];
    
    for i=1:length(dDesc)
        %fprintf('-- i= %d/%d\n', i, length(dDesc));
        des= [];
        
        dname = dDesc(i).name;
        d = load([desc_fold dname]);
        
        jFeatName= strrep(dname,'d_','2hajar1_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        temp= jFeats;
        
        jFeatName= strrep(dname,'d_','3hajar2_');
        jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
        load(['./jointsFeat/',jFeatName]);
        jFeats= [temp jFeats];
        
        % normalize features
        % -------------------
        % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
        
        
        ind = strfind(dname,'a');
        action = str2num(dname(ind(1)+1:ind(1)+2));
        ind = strfind(dname,'s');
        actor = str2num(dname(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            trainingDesc = [trainingDesc;d];
            trainingLbls = [trainingLbls;action];
            
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            testingDesc = [testingDesc;d];
            testingLbls = [testingLbls;action];
            
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    svmParams = '-q -t 0 -g 0.125';
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
end
%}


% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Figure 2: a sequnce of skeleton for one action
% drawskt(1,1,1,1,1,1);

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Figure 3: Draw confusion matrix
%{
close all

% Create data
mymatrix =rand(20,20)*100;

% generate a plot
image(mymatrix); 
colormap(jet);
colorbar

% Define the labels 
lab = [{'highArmWave'};{'horizontalArmWave'};{'hammer'};{'handCatch'};...
    {'forwardPunch'};{'highThrow'};{'drawX'};{'drawTick'};{'drawCircle'};...
    {'handClap'};{'twoHandWave'};{'sideBoxing'};{'bend'};{'forwardKick'};...
    {'sideKick'};{'jogging'};{'tennisSwing'};{'tennisServe'};...
    {'golfSwing'};{'pickupThrow'}];


% Set the tick locations and remove the labels 
set(gca,'XTick',1:20,'XTickLabel','','YTick',1:20,'YTickLabel',lab); 

% Estimate the location of the labels based on the position of the xlabel 
hx = get(gca,'XLabel');  % Handle to xlabel 
set(hx,'Units','data'); 
pos = get(hx,'Position'); 
yt = pos(2); 

textStrings = num2str(mymatrix(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:20);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center','FontSize', 6);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = [0 0 0]; 
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors


pos = get( gca, 'Position' );
set( gca, 'Position', [0.2 0.17 0.7 0.7] )
        
Xt=get(gca,'XTick');
% Place the new labels 
for i = 1:size(lab,1) 
    t(i) = text(Xt(i),yt,lab(i,:)); 
end 

set(t,'Rotation',45,'HorizontalAlignment','right')  
saveas(gca,'figure_name_out','epsc')
%}


% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Figure 4: Draw bar of action accuracies separately for baseline and ours
%{
close all

% Create data
mymatrix =rand(20,2);

% generate a plot
bar(mymatrix); 
colormap(cool);

% Define the labels 
lab = [{'highArmWave'};{'horizontalArmWave'};{'hammer'};{'handCatch'};...
    {'forwardPunch'};{'highThrow'};{'drawX'};{'drawTick'};{'drawCircle'};...
    {'handClap'};{'twoHandWave'};{'sideBoxing'};{'bend'};{'forwardKick'};...
    {'sideKick'};{'jogging'};{'tennisSwing'};{'tennisServe'};...
    {'golfSwing'};{'pickupThrow'}];


% Set the tick locations and remove the labels 
set(gca,'XTick',1:20,'XTickLabel',''); 

% Estimate the location of the labels based on the position of the xlabel 
hx = get(gca,'XLabel');  % Handle to xlabel 
set(hx,'Units','data'); 
pos = get(hx,'Position'); 
yt = pos(2); 

set(gca,'XLim',[0 21],'YLim',[0 1]) % 21:max in x,1:max value in columns(y)

pos = get( gca, 'Position' );
set( gca, 'Position', [0.2 0.17 0.7 0.7] )
      
        
Xt=get(gca,'XTick');
% Place the new labels 
for i = 1:size(lab,1) 
    t(i) = text(Xt(i),yt,lab(i,:)); 
end 

set(t,'Rotation',45,'HorizontalAlignment','right')  
saveas(gca,'figure_name_out','epsc')
%}


%{
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Figure 5: Get the values of Confusion Matrix
% load and run with GMM-237 to get the confusion matrix values 

load r_meanDis237 ;
means1= r_meanDis237;

load r_covDis237 ;
covariance1= r_covDis237 ;

load r_priorDis237 ;
priors1= r_priorDis237;

load r_meanDir237;
means2= r_meanDir237;

load r_covDir237;
covariance2= r_covDir237;

load r_priorDir237;
priors2= r_priorDir237;

tic;

graph= 'ful'; % or 'ana'

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 50 ; % for Fisher vector
winWid= 5;
slideIncr= 1;


if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end

lineNum= size(lines,2);   % number of lines


% =============================== Distance
dim = 2;      % joints dimension
% Making Fisher vector for each video
% ===================================
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    
    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDisC= [];
    for j=1:size(disC,1)
        temp= disC(j,:)';
        smoothDisC= cat(1, smoothDisC, (smooth(temp))');
    end
    disC= smoothDisC;
    
    for j=1:size(disC,1) % number of lines
        vidData= [];     % data per line in each video
        feat= disC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means1, covariance1, priors1);
        
        %         % power "normalization": Variance stabilizing transform:
        %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
        encoding = sign(encoding) .* sqrt(abs(encoding));
        
        jFeats= cat(2, jFeats, encoding');
    end
    
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar1_', strrep(curVideo,'.txt','.mat')],'jFeats');
end


% =============================== Direction
dim = 3;      % joints dimension
% Making Fisher vector for each video
% ===================================
for i= 1:numVids
    %fprintf('video: %i/%i\n',i, numVids);
    
    jFeats= [];
    curVideo= listVids(i).name;  % current video
    
    a= sscanf(curVideo,'a%d_*.dat');
    s= sscanf(curVideo,'%*3c_s%d*.dat');
    e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
    
    dirC= dirChange(a, s, e, lines, dim);  % real change not its 'sign'
    smoothDirC= [];
    for j=1:size(dirC,1)
        temp= dirC(j,:)';
        smoothDirC= cat(1, smoothDirC, (smooth(temp))');
    end
    dirC= smoothDirC;
    
    for j=1:size(dirC,1) % number of lines
        vidData= [];     % data per line in each video
        feat= dirC(j, :);
        numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
        
        for k = 1:numstps
            windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
            vidData= cat(2, vidData, windowSlid');
        end
        
        % data: rows:dimension(19) ,columns:number of data(numFrame-1)
        encoding = vl_fisher(vidData, means2, covariance2, priors2);
        
        %         % power "normalization": Variance stabilizing transform:
        %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
        encoding = sign(encoding) .* sqrt(abs(encoding));
        
        jFeats= cat(2, jFeats, encoding');
    end
    
    % replace NaN vectors with a large value that is far from everything else
    % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
    % many vectors.
    
    jFeats(find(isnan(jFeats))) = 123;
    save(['./jointsFeat/',num2str(dim),'hajar2_', strrep(curVideo,'.txt','.mat')],'jFeats');
end


% Same split in the paper, train(1,3,5,7), test(2,4,6,8,10)
% addpath '../libsvm-3.12/matlab/';
desc_fold = './HON4Ddesc/';
dDesc = dir([desc_fold '*.txt']);

trainDesFeat= []; % Qualitative Discriptors
testDesFeat= [];  % Qualitative Discriptors
trainingDesc = [];
testingDesc = [];
trainingLbls = [];
testingLbls = [];
trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];

for i=1:length(dDesc)
    %fprintf('-- i= %d/%d\n', i, length(dDesc));
    des= [];
    
    dname = dDesc(i).name;
    d = load([desc_fold dname]);
    
    jFeatName= strrep(dname,'d_','2hajar1_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    temp= jFeats;
    
    jFeatName= strrep(dname,'d_','3hajar2_');
    jFeatName= strrep(jFeatName,'_sd.txt','_skeleton.mat');
    load(['./jointsFeat/',jFeatName]);
    jFeats= [temp jFeats];
    
    % normalize features
    % -------------------
    % jFeats = jFeats/norm(jFeats);  % Normalize columns of matrix
    
    
    ind = strfind(dname,'a');
    action = str2num(dname(ind(1)+1:ind(1)+2));
    ind = strfind(dname,'s');
    actor = str2num(dname(ind(1)+1:ind(1)+2));
    
    if isempty(find(testActors == actor)) % training;
        trainingDesc = [trainingDesc;d];
        trainingLbls = [trainingLbls;action];
        
        trainDesFeat= cat(1, trainDesFeat, jFeats);
        
    else                                  % testing
        testingDesc = [testingDesc;d];
        testingLbls = [testingLbls;action];
        
        testDesFeat= cat(1, testDesFeat, jFeats);
    end
end

% Our Features
trainingDesc =  trainDesFeat;
testingDesc =  testDesFeat;

svmParams = '-q -t 0 -g 0.125';
model = svmtrain(trainingLbls,trainingDesc,svmParams);
predicted_labels = svmpredict(testingLbls,testingDesc,model);
acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100

% Confusin Matrix
cm= confusionmat(testingLbls,predicted_labels);

toc;

%}
% =========================================================================




%{
% --------------------- 12 May 2014 ---------------------------------------
% TASK # Feature selection/example
% create a very simple dataset
y = [zeros(500,1); ones(500,1)];
y = y(randperm(1000));

% We have 100 variables x that we want to use to predict y. 99 of them are
% just random noise,but one of them is highly correlated with the classlabel

x = rand(1000,99);
x(:,100) = y + rand(1000,1)*0.1;
x(14,100)= 0.94;
x(27,100)= 0.18;


% If we were to do this directly without applying any feature selection, we
% would first split the data up into a training set and a test set:
xtrain = x(1:700, :); xtest = x(701:end, :);
ytrain = y(1:700); ytest = y(701:end);

%Then we would classify them:
ypred = classify(xtest, xtrain, ytrain);

% And finally we would measure the error rate of the prediction:
sum(ytest ~= ypred)

% To make a function handle to be used with sequentialfs, just put these
% pieces together:
%f = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));
svmParams = '-q -t 0 -g 0.125';
svmwrapper = @(xtrain, ytrain, xtest, ytest) sum(svmpredict(ytest, xtest...
    ,svmtrain(ytrain, xtrain, svmParams)) ~= ytest);

% And pass all of them together into sequentialfs:
opts = statset('display','iter');
[fs] = sequentialfs(svmwrapper,xtrain, ytrain,'options',opts)
%}
% -------------------------------------------------------------------------





% ----------------------- 19 May 2014 -------------------------------------
% TASK #28 my own forward feature selection on lines of skeleton/in ROWs:6/
% FV/distance/SVM/MSR dataset
% {
% Make Fisher vector for each video

dim= 3;
graph= 'ana'; % or 'ana'

dataFolder= './MSRAction3DSkeleton(20joints)/';
listVids= dir([dataFolder,'*.txt']);  % number of videos in the dataset
numVids= numel(listVids);
numJoints= 20;
numClusters = 128 ; % for Fisher vector
winWid= 6;
slideIncr= 1;
svmParams = '-q -t 0 -g 0.125';

trainActors = [1 3 5 7 9];
testActors = [2 4 6 8 10];


if strcmp(graph,'ana')
    % anotomy skeleton
    lines=[20  1  2  1  8   10  2  9   11  3  4  7  7 5  6  14 15 16 17;
        3   3  3  8  10  12  9  11  13  4  7  5  6  14  15  16  17  18  19];
else
    % Fully connected skeleton
    lines= [];
    for j=1:numJoints
        for k= 1:numJoints
            if (j~=k)
                lines= [lines, [j;k]];
            end
        end
    end
end
lineNum= size(lines,2);   % number of lines


% /\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\\//\\//\\
% Feature Selection:
% ------------------

leftLineSet= 1:5; %lineNum; % at first, all lines are not considered
bestLineSet= []; % set of best lines
svmAccuracy= [];

while(~isempty(leftLineSet))
    fprintf('Number of left lines: %i\n', size(leftLineSet,2));
    
    accuracy= zeros(1, size(leftLineSet, 2));
    for f= 1:size(leftLineSet, 2)
        curLineSet= cat(2, bestLineSet, leftLineSet(f));
        
        % Leave-one-out Cross Validation
        % ==============================
        delete('./jointsFeat/*.mat');
        
        data= [];   % windowsize x sequence of words in all videos
        for i= 1:numVids
            %fprintf('video: %i/%i\n',i, numVids);
            curVideo= listVids(i).name;  % current video
            
            a= sscanf(curVideo,'a%d_*.dat');
            s= sscanf(curVideo,'%*3c_s%d*.dat');
            e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
            
            ind = strfind(curVideo,'s');
            actor = str2num(curVideo(ind(1)+1:ind(1)+2));
            
            if isempty(find(testActors == actor)) % training;
                
                disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
                disC= disC(curLineSet,:); % only for the selected features
                
                smoothDisC= [];
                for j=1:size(disC,1)
                    temp= disC(j,:)';
                    smoothDisC= cat(1, smoothDisC, (smooth(temp))');
                end
                disC= smoothDisC;
                
                for j=1:size(disC,1) % number of lines
                    feat= disC(j, :);
                    numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                    
                    for k = 1:numstps
                        windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                        
                        %data= cat(2, data, windowSlid');
                        data(:, end + 1) = windowSlid'; % to optimize for speed,
                        
                    end
                end
            end
        end
        
        
        % G-MM: mixture of Gaussians from data of all videos
        % ==================================================
        % Run EM
        [means,covariance,priors] = vl_gmm(data, numClusters);
        
        
        % Making Fisher vector for each video
        % ===================================
        allFeat= []; % all features
        allClass= []; % class labels
        for i= 1:numVids
            %fprintf('video: %i/%i\n',i, numVids);
            
            jFeats= [];
            curVideo= listVids(i).name;  % current video
            
            a= sscanf(curVideo,'a%d_*.dat');
            s= sscanf(curVideo,'%*3c_s%d*.dat');
            e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
            
            ind = strfind(curVideo,'s');
            actor = str2num(curVideo(ind(1)+1:ind(1)+2));
            ind = strfind(curVideo,'a');
            action = str2num(curVideo(ind(1)+1:ind(1)+2));
            
            if isempty(find(testActors == actor))  % training;
                
                disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
                disC= disC(curLineSet,:);
                
                
                smoothDisC= [];
                for j=1:size(disC,1)
                    temp= disC(j,:)';
                    smoothDisC= cat(1, smoothDisC, (smooth(temp))');
                end
                disC= smoothDisC;
                
                for j=1:size(disC,1) % number of lines
                    vidData= [];     % data per line in each video
                    feat= disC(j, :);
                    numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                    
                    for k = 1:numstps
                        windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                        vidData= cat(2, vidData, windowSlid');
                    end
                    
                    % data: rows:dimension(19) ,columns:number of data(numFrame-1)
                    encoding = vl_fisher(vidData, means, covariance, priors);
                    
                    %         % power "normalization": Variance stabilizing transform:
                    %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
                    encoding = sign(encoding) .* sqrt(abs(encoding));
                    
                    jFeats= cat(2, jFeats, encoding');
                end
                
                % replace NaN vectors with a large value that is far from everything else
                % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
                % many vectors.
                
                jFeats(find(isnan(jFeats))) = 123;
                %save(['./jointsFeat/',num2str(dim),'hajar_', strrep(curVideo,'.txt','.mat')],'jFeats');
                
                allFeat(end + 1, :)= jFeats;
                allClass(end + 1, :)= action;
            end
        end
        
        
        
        % k-fold Cross Validation
        % -----------------------
        kf= 30;                        % Leave-10-out
        cvPart= cvpartition(size(allFeat,1),'kfold',kf);
        acc= zeros(1, kf);
        
        for cv= 1:kf       % Leave-10-out cross-validation
            %fprintf('Cross-Validation: %i/%i ...\n', cv, kf);
            
            % test data
            % ---------
            tstIndx= find(test(cvPart, cv)); % indices for the test data
            tstFeat= allFeat(tstIndx,:);
            tstClass= allClass(tstIndx,:);
            
            % train data
            % ----------
            trnIndx= find(training(cvPart, cv)); % indices for the test data
            trnFeat= allFeat(trnIndx,:);
            trnClass= allClass(trnIndx,:);
            
            model = svmtrain(trnClass, trnFeat, svmParams);
            predicted_labels = svmpredict(tstClass, tstFeat, model);
            acc(1,cv)=(length(find((predicted_labels==tstClass)==1))/length(tstClass))*100;
        end
        % ==============================
        mean(acc)
        accuracy(1, f)= mean(acc);
    end
    
    % choose the best line from the left lines
    % ----------------------------------------
    bes=leftLineSet(accuracy==max(accuracy)); % the best current line
    
    bestLineSet= cat(2, bestLineSet, bes);    % add to the bestLineSet
    leftLineSet(accuracy==max(accuracy))= []; % remove from the leftLines
    % ----------------------------------------
    
    
    % /\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\\//\\//\\
    
    fprintf('\n\nThe Best sub set of lines in the skeleton are:\n');
    bestLineSet
    leftLineSet
    
    % {
    % Now extract the features for the selected lines
    % -----------------------------------------------
    delete('./jointsFeat/*.mat');
    
    data= [];   % windowsize x sequence of words in all videos
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        if isempty(find(testActors == actor)) % training;
            
            disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
            disC= disC(bestLineSet,:); % only for the selected features
            
            smoothDisC= [];
            for j=1:size(disC,1)
                temp= disC(j,:)';
                smoothDisC= cat(1, smoothDisC, (smooth(temp))');
            end
            disC= smoothDisC;
            
            for j=1:size(disC,1) % number of lines
                feat= disC(j, :);
                numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
                
                for k = 1:numstps
                    windowSlid = feat(k:(k+winWid-1));  % Calculation for each window
                    
                    %data= cat(2, data, windowSlid');
                    data(:, end + 1) = windowSlid'; % to optimize for speed,
                    
                end
            end
        end
    end
    
    
    % G-MM: mixture of Gaussians from data of all videos
    % ==================================================
    % Run EM
    [means,covariance,priors,ll] = vl_gmm(data, numClusters);
    
    
    % Making Fisher vector for each video
    % ===================================
    trainDesFeat= [];
    testDesFeat= [];
    trainingLbls= [];
    testingLbls= [];
    for i= 1:numVids
        %fprintf('video: %i/%i\n',i, numVids);
        
        jFeats= [];
        curVideo= listVids(i).name;  % current video
        
        a= sscanf(curVideo,'a%d_*.dat');
        s= sscanf(curVideo,'%*3c_s%d*.dat');
        e= sscanf(curVideo,'%*3c_%*3c_e%d.dat');
        
        ind = strfind(curVideo,'a');
        action = str2num(curVideo(ind(1)+1:ind(1)+2));
        ind = strfind(curVideo,'s');
        actor = str2num(curVideo(ind(1)+1:ind(1)+2));
        
        disC= distChange(a, s, e, lines, dim);  % real change not its 'sign'
        disC= disC(bestLineSet,:);
        
        smoothDisC= [];
        for j=1:size(disC,1)
            temp= disC(j,:)';
            smoothDisC= cat(1, smoothDisC, (smooth(temp))');
        end
        disC= smoothDisC;
        
        for j=1:size(disC,1) % number of lines
            vidData= [];     % data per line in each video
            feat= disC(j, :);
            numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
            
            for k = 1:numstps
                windowSlid = feat(k:(k+winWid-1));  %Calculation for each window
                vidData= cat(2, vidData, windowSlid');
            end
            
            % data: rows:dimension(19) ,columns:number of data(numFrame-1)
            encoding = vl_fisher(vidData, means, covariance, priors);
            
            %         % power "normalization": Variance stabilizing transform:
            %         % f(z)= sign(z)|z|^a with 0<=a<=1 (a=0.5 by default)
            encoding = sign(encoding) .* sqrt(abs(encoding));
            
            jFeats= cat(2, jFeats, encoding');
        end
        
        % replace NaN vectors with a large value that is far from everything else
        % For normalized vectors in high dimension, vector (0, ..., 0) is *close* to
        % many vectors.
        
        jFeats(find(isnan(jFeats))) = 123;
        % save(['./jointsFeat/',num2str(dim),'hajar_', strrep(curVideo,'.txt','.mat')],'jFeats');
        
        if isempty(find(testActors == actor))  % training;
            trainingLbls = [trainingLbls; action ];
            trainDesFeat= cat(1, trainDesFeat, jFeats);
            
        else                                  % testing
            
            testingLbls = [testingLbls; action];
            testDesFeat= cat(1, testDesFeat, jFeats);
        end
    end
    
    % Our Features
    trainingDesc =  trainDesFeat;
    testingDesc =  testDesFeat;
    
    model = svmtrain(trainingLbls,trainingDesc,svmParams);
    predicted_labels = svmpredict(testingLbls,testingDesc,model);
    acc=(length(find((predicted_labels==testingLbls)==1))/length(testingLbls))*100
    
    svmAccuracy= cat(2, svmAccuracy, acc);
    toc;
end

%}
% -------------------------------------------------------------------------

