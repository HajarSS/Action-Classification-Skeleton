% In the name of GOD...
% ---------------------

% 17 Feb 2014
% Joint Bag-of-words Histogram

function histo= jointBagHist(feat, words)

% Input:
% Output: is a vector of counting the words

% The L1 distance between two vectors is defined as:  sum(abs(x-y));

histo= zeros(1, size(words,1));
for i = 1:size(feat, 1)
   [~,indx] = ismember(words, feat(i, :), 'rows');
   indx= find(indx==1);
   
   if isempty(indx) % this may happen for test videos
       L1_dist= sum(abs(bsxfun(@minus, words, feat(i, :))), 2); % L1-distance
       [~,indx]= min(L1_dist); % new index shows the closest exiting word
   end
   histo(1,indx)= histo(1,indx) + 1; % histogram
end
%}
