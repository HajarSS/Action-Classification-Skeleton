% In the name of GOD...
% ---------------------

% 13 Feb 2014
% finds distance change from frame i to frame i+1
% this change can be: -1(decrease), 0(no change), 1(increase)

function disC= distChange(a, s, e, lines, dim)

% Output: is a vector of -1, 0, 1

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
lineNum= size(lines,2);  % number of lines
 
disC= zeros(lineNum, frNum-1);
if dim==2
    for j= 1:lineNum
        for i=1:frNum
            c1= lines(1,j);
            c2= lines(2,j);
            x= B(c2,i,1) - B(c1,i,1);
            y= B(c2,i,2) - B(c1,i,2);
            
            curDist= sqrt(x^2 + y^2); 
            
            if (i~=1) % if we are not in the first frame
                disC(j, i-1)= (curDist - preDist); 
            end
            
            preDist= curDist;
        end
    end
elseif dim==3
    for j= 1:lineNum
        for i=1:frNum
            c1= lines(1,j);
            c2= lines(2,j);
            x= B(c2,i,1) - B(c1,i,1);
            y= B(c2,i,2) - B(c1,i,2);
            z= B(c2,i,3) - B(c1,i,3);
            
            curDist= sqrt(x^2 + y^2 + z^2); 
            
            if (i~=1) % if we are not in the first frame
                disC(j, i-1)= curDist - preDist; 
            end
            
            preDist= curDist;
        end
    end
end

