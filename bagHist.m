% In the name of GOD...
% ---------------------

% 12 Feb 2014
% Bag-of-words Histogram

function histo= bagHist(feat, winWid, slideIncr,m1, m2)

% Input:
% feat: a feature vector of k different values (it can be different each
% time this function is called), k here is 3 for distance {-1,0,1}
% winWid: Sliding window width, we chose 3
% slideIncr:  Slide for each iteration, we chose 1
% numWords: number of words in the histogram which is k^(winSize) (here is
% 3^3= 27 words
% m1,m2: minimum and maximum of the feature (-1,0,1 for distance)
% (1,2,3,4,5,6,7,8 for the direction)

% Output: is a vector of counting the words
% k= unique(feat); % finds k (number of different values and theirselves)


% MAking the words
% ================
% Combinations with replacement (3^3= 27 words)
x = m1:m2; % 1:8; % -1:1;   % the elements you want to choose from
i = 1:length(x);  % indices into the vector x
[i1,i2,i3,i4] = ndgrid(i); % all possible combinations of indices into x
words = x([i4(:) i3(:) i2(:) i1(:)]); % or use cat(2,x3(:),x2(:),x1(:))


% Combinations with replacement (3^4= 81 states)
% x = -1:1;         % the elements you want to choose from
% i = 1:length(x);  % indices into the vector x
% [i1,i2,i3,i4] = ndgrid(i); % all possible combinations of indices into x
% y = x([i4(:) i3(:) i2(:) i1(:)]); % or use cat(2,x3(:),x2(:),x1(:))


histo= zeros(1, size(words,1));
numstps = (length(feat)-winWid)/slideIncr + 1; % Number of windows
for i = 1:numstps 
   windowSlid = feat(i:(i+winWid-1));  %Calculation for each window
   [~,indx] = ismember(words, windowSlid, 'rows');
   indx= find(indx==1);
   histo(1,indx)= histo(1,indx) + 1; % histogram
end

