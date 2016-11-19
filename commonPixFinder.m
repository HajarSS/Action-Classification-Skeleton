% In the name of GOD...
% ---------------------
% Working on KR conference
% Start: 2013-09-10


%
%--------------------------------------------------------- 11 Nov 2013
%-------------------- Evaluation measurement
%

%{
function comPixNum= commonPixFinder(gtTr,gtFr_s,testTr,testFr_s) 

% Input:
% gtTr,gtFr_s: track, start frame in the video for
% the ground truth track
% testTr,testFr_s: track,start frame for the test track

% Output:
% comPixNum: number of common pixels between test and ground truth rectangle for
% each frame

comPixNum= zeros(1, size(gtTr,1));
m= max(gtFr_s , testFr_s);
for i=1:size(gtTr,1)
    if ((m-testFr_s+i)>size(testTr,1)) || ((m-gtFr_s+i)>size(gtTr,1))
       break;
    end
    comPixNum(m-gtFr_s+i)= rectint(gtTr(m-gtFr_s+i,:), testTr(m-testFr_s+i,:)); % common area
end
%}


% version 2
function intersectMeasure= commonPixFinder(gtTr,gtFr_s,testTr,testFr_s, thr) 

% Input:
% gtTr,gtFr_s: track, start frame in the video for the ground truth track
% testTr,testFr_s: track,start frame for the test track
% thr: threshold for intersection/union

% Output:
% intersectMeasure: a measurement to show how much intersection exists
% between two rectangles

m= max(gtFr_s , testFr_s);

interArea= [];  % eshterake a,b  % intersection area
unionArea= [];  % ejtemae a,b  
sizeTst= [];    
for i=1:size(gtTr,1)
    if ((m-testFr_s+i)>size(testTr,1)) || ((m-gtFr_s+i)>size(gtTr,1))
       break;
    end
    interArea= cat(1, interArea, rectint(gtTr(m-gtFr_s+i,:), ...
        testTr(m-testFr_s+i,:))); % common area
    x= max((gtTr(m-gtFr_s+i,1)+gtTr(m-gtFr_s+i,3)), ...
        (testTr(m-testFr_s+i,1)+testTr(m-testFr_s+i,3)))-...
        min(gtTr(m-gtFr_s+i,1), testTr(m-testFr_s+i,1));
    y= max((gtTr(m-gtFr_s+i,2)+gtTr(m-gtFr_s+i,4)), ...
        (testTr(m-testFr_s+i,2)+testTr(m-testFr_s+i,4)))-...
        min(gtTr(m-gtFr_s+i,2), testTr(m-testFr_s+i,2));
    unionArea= cat(1, unionArea, x*y);  % common area (ejtema A,B)
    % size of the test rectangle
    sizeTst= cat(1,sizeTst, testTr(m-testFr_s+i,3)*testTr(m-testFr_s+i,4));
end

count= 0; % number of frames with good overlapping with ground trut
for i=1:size(interArea,1)
    if ((interArea(i)/unionArea(i))>=thr) || ((interArea(i)/sizeTst(i))>=0.9)
        count= count+1;
    end
end
intersectMeasure= (count/size(interArea,1));













