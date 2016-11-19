% In the Name of GOD
%*******************


%
% -------------------------------------------- 19 Nov 2013
% Bag of words technique for HMM

function clustIDX= BagofWords1(data, nClust)
% Input
% data: a matrix of m*n which m is (number of frames)*(number of videos)
% and n is each feature dimension (20 in this case) 
% nClust: the number of words (clusters)

% Output
% clustIDX: a vector of 1*nClust of words


[~,centers]= kmeans(data,nClust,'emptyaction','singleton','display','iter');

% calculate distance of each instance to all cluster centers
clustIDX= zeros(size(data,1),1);
for i=1:size(data,1)
    %fprintf('pixel:%i...\n', i);
    D= zeros(1,nClust);
    for j=1:nClust
        D(1,j) = sum((data(i,:)-centers(j,:)).^2);
    end
    [~,clustIDX(i,1)]= min(D);
end

