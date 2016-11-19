% ----- 24 Feb 2014
% This function extracts the distance between all joints and the direction
% between the lines (lines connect the joints).

function jFeats=jointFeatExtractor(a, s, e, dim, lines)

% First Step:  
% Read all the joint co-ordinates in a vector called 'joints'
% -----------------------------------------------------------
% a: action, s: subject, e: instance

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
joNum= size(B, 1);   % number of joints(20)

      
if dim==2   % 2D features
    % {
    % Calculate the distance between the joints
    % -----------------------------------------
    distance= []; % distance between each pair points
    for f=1:frNum
        dis= [];
        for j= 1:size(lines,2) % number of lines (19)
            c1= lines(1,j) 
            c2= lines(2,j) 
            X= [B(c1,f,1) B(c1,f,2); B(c2,f,1) B(c2,f,2)];
            
            dis= [dis ; pdist(X,'euclidean')] 
        end
        distance= [distance dis];
    end
    % output of this step (distance): is a 380xfrNum which 380=joNumx(joNum-1)
    %}
    
    %{
    % Extract statistical features
    % ----------------------------
    des= [];
    des= cat(2, des, max(distance,[],2));  % in rows
    des= cat(2, des, mean(distance,2));
    des= cat(2, des, median(distance,2));
    des= cat(2, des, min(distance,[],2));
    des= cat(2, des, mode(distance,2));
    des= cat(2, des, std(distance,0,2));
    des= cat(2, des, var(distance,0,2));
    distance= des;
    %}
    
    %{
    % Calculate the direction between the lines
    % -----------------------------------------
    direct= []; 
    for f=1:frNum
        d= [];
        for j= 1:size(lines,2) % number of lines (19)
            c1= lines(1,j);
            c2= lines(2,j);
            
            x= B(c2,f,1) - B(c1,f,1);
            y= B(c2,f,2) - B(c1,f,2);
            
            d= [d ; Direction_Stimate(x,y)];
        end
        direct= [direct d];
    end
    %}
    
    %{
    % Extract statistical features
    % ----------------------------
    des= [];
    des= cat(2, des, max(direct,[],2));  % in rows
    des= cat(2, des, mean(direct,2));
    des= cat(2, des, median(direct,2));
    des= cat(2, des, min(direct,[],2));
    des= cat(2, des, mode(direct,2));
    des= cat(2, des, std(direct,0,2));
    des= cat(2, des, var(direct,0,2));
    direct= des;
    %}

    %{
    top= [];
    for f= 1:frNum
        d= [];  
        for j= 1:size(lines,2) % number of lines (19)
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

                d= [d ; topFinder(s_a, e_a, s_b, e_b)]; 
            end
        end
        top= [top d];        
    end
    % direct is a (lineNum)x(frNum) matrix
    %
    
    % Extract statistical features
    % ----------------------------
    des= [];
    des= cat(2, des, max(top,[],2));  % in rows
    des= cat(2, des, mean(top,2));
    des= cat(2, des, median(top,2));
    des= cat(2, des, min(top,[],2));
    des= cat(2, des, mode(top,2));
    des= cat(2, des, std(top,0,2));
    des= cat(2, des, var(top,0,2));
    top= des;
    %}
    
    % normalize features
    % -------------------
    distance = zscore(distance);
    %direct = zscore(direct);
    %top= zscore(top);
    
    %jFeats= [distance ; direct; top];
    %jFeats= top;
    jFeats= distance;
    
elseif dim==3  % 3D features
    % 
    % Calculate the distance between the joints
    % -----------------------------------------
    distance= []; % distance between each pair points
    for f=1:frNum
        dis= [];
        for j= 1:size(lines,2) % number of lines (19)
            c1= lines(1,j);
            c2= lines(2,j);
            X= [B(c1,f,1) B(c1,f,2) B(c1,f,3); B(c2,f,1) B(c2,f,2) B(c2,f,3)];
            
            dis= [dis ; pdist(X,'euclidean')];
        end
        distance= [distance dis];
    end
    % output of this step (distance): is a 380xfrNum which 380=joNumx(joNum-1)

    % Extract statistical features
    % ----------------------------
    des= [];
    des= cat(2, des, max(distance,[],2));  % in rows
    des= cat(2, des, mean(distance,2));
    des= cat(2, des, median(distance,2));
    des= cat(2, des, min(distance,[],2));
    des= cat(2, des, mode(distance,2));
    des= cat(2, des, std(distance,0,2));
    des= cat(2, des, var(distance,0,2));
    distance= des;
    
    %
    % Calculate the direction between the lines
    % -----------------------------------------
    direct= [];
    for f=1:frNum
        d= [];
        for j= 1:size(lines,2) % number of lines (19)
            c1= lines(1,j);
            c2= lines(2,j);
            
            x= B(c2,f,1) - B(c1,f,1);
            y= B(c2,f,2) - B(c1,f,2);
            z= B(c2,f,3) - B(c1,f,3);
            
            d= [d; Direction_Stimate(x,y)];
            d= [d; Direction_Stimate(y,z)];
            d= [d; Direction_Stimate(x,z)];
        end
        direct= [direct d];
    end

    
    % Extract statistical features
    % ----------------------------
    des= [];
    des= cat(2, des, max(direct,[],2));  % in rows
    des= cat(2, des, mean(direct,2));
    des= cat(2, des, median(direct,2));
    des= cat(2, des, min(direct,[],2));
    des= cat(2, des, mode(direct,2));
    des= cat(2, des, std(direct,0,2));
    des= cat(2, des, var(direct,0,2));
    direct= des;
    %
    
    % normalize features
    % -------------------
    distance = zscore(distance);
    direct = zscore(direct);
    
    jFeats= [distance ; direct];
    
end
  