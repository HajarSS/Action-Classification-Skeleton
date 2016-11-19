% In the name of GOD...
% ---------------------

% 13 Feb 2014
% Quantify direction of each line in frame i 
% this direction can be: 1,2,3,4,5,6,7,8

function dirC= dirChange(a, s, e, lines, dim)

% Output: is a vector of 1,2,3,4,5,6,7,8

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
 
% dirC= []; % size: (frNum, lineNum);
% if dim==2
%     for i= 1:frNum
%         d= [];
%         for j= 1:lineNum   
%             c1= lines(1,j);
%             c2= lines(2,j);
%             x= B(c2,i,1) - B(c1,i,1);
%             y= B(c2,i,2) - B(c1,i,2);
%             
%             d= cat(2, d, Direction_Stimate(x,y)); 
%         end
%         dirC= cat(1, dirC, d);
%     end
%     
% elseif dim==3
% end


if dim==2
    dirC= zeros(lineNum, frNum-1); % size: (lineNum, frNum-1);
    for i= 1:frNum
        theta= [];   % 19 x 1 column vector
        for j= 1:lineNum   
            c1= lines(1,j);
            c2= lines(2,j);
            x= B(c2,i,1) - B(c1,i,1); 
            y= B(c2,i,2) - B(c1,i,2); 

            if x==0 && y==0
                theta= cat(1, theta, 0);  % the angle
            else
                theta= cat(1, theta, atan(y/x));  % the angle
            end
            
        end
        if i==1 % the first frame
            preTheta= theta;
        else
            dirC(:, i-1)= theta - preTheta; 
            preTheta= theta;
        end
    end
    
elseif dim==3
    dirC= zeros(lineNum*3, frNum-1); % size: (lineNumx3, frNum-1);
    for i= 1:frNum
        theta= [];   % (19x3) x 1 column vector
        for j= 1:lineNum   
            c1= lines(1,j);
            c2= lines(2,j);
            x= B(c2,i,1) - B(c1,i,1); 
            y= B(c2,i,2) - B(c1,i,2); 
            z= B(c2,i,3) - B(c1,i,3);
            
            if x==0 && y==0
                theta= cat(1, theta, 0);  % the angle
            else
                theta= cat(1, theta, atan(y/x));  % the angle
            end
            if x==0 && z==0
                theta= cat(1, theta, 0);  % the angle
            else
                theta= cat(1, theta, atan(z/x));  % the angle
            end
            if y==0 && z==0
                theta= cat(1, theta, 0);  % the angle
            else
                theta= cat(1, theta, atan(z/y));  % the angle
            end
            
        end
        if i==1 % the first frame
            preTheta= theta;
        else
            dirC(:, i-1)= theta - preTheta; 
            preTheta= theta;
        end
    end
end