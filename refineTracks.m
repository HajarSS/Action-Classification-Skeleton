% In the name of GOD...
% ---------------------
% Working on KR conference
% Start: 2013-09-10



% Refining tracks to remove zero rows
function [tr, fr_se]= refineTracks(act,videoNum)
% Input:
% act: three letter to show the action (like 'wal' for walking)
% videoNum: video number

% Output:
% tr: one (1*n) cell of k*4 rectangles for the track which n is the number
% of tracks for this video (shown by videoNum)
% fr_se: a 2*n matrix which 2 columns are for start frame and end frame for
% the track


load (['./tracks/tr_', act, num2str(videoNum),'.mat']);
load (['./tracks/frIdx_', act, num2str(videoNum),'.mat']);


tr= {};
fr_se= [];
for i=1:size(tracks,2)
    for j=1:size(tracks{i}, 2)
        rectTr= tracks{i}{j};  % it's a matrix of k*4 (k:length of track)
        
        zeroRows= find(all(rectTr==0,2));  % finds all rows with zero values
        if ~isempty(zeroRows) % if there are some zero rows in the track
            % state 1
            % -------
            % when from frame min(zeroRows) to end rows are zero
            if ((max(zeroRows)-min(zeroRows)+1)==size(zeroRows,1))&&...
                    (zeroRows(end)==size(rectTr,1))
                rectTr(~any(rectTr,2),:)= [];  % delete zero rows
            else
                % state 2
                % -------
                % when tracker loose object in some frames and then finds it
                k= 1;
                while (k<= size(zeroRows,1))
                    curFr= zeroRows(k)-1;   % frame before zero rows
                    lastFr= zeroRows(k)+1;% frame after zero rows

                    while ((k<size(zeroRows,1)) && (lastFr==zeroRows(k+1)))
                        k= k+1;
                        lastFr= zeroRows(k)+1;
                    end
                    
                    if (k==size(zeroRows,1))&&(zeroRows(end)==size(rectTr,1))
                        rectTr(curFr+1:end,:)= [];  % delete zero rows
                        break;
                    end
                    
                    k= k+1;                
                    % number of missed frames between lastFrame & currentFrame
                    missFrames= lastFr-curFr-1; 
                    newRect= rectTr(curFr,:);
                    for fr= (curFr+1):(lastFr-1)
                        newRect= (rectTr(lastFr,:)-rectTr(curFr,:))/(missFrames+1)+...
                            newRect;
                        rectTr(fr,:)= round(newRect);
                    end
                end % while end
                
            end
            
            
        end
        
        tr= cat(2, tr, rectTr);
        fr_se= cat(2, fr_se, [frIdx(i); frIdx(i)+size(rectTr,1)-1]);
    end
end
