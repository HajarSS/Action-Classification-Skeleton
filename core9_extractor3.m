% In the Name of GOD
%*******************

%{
% version number 1
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  21 Sep 2012
% extracting CORE-9 information for each frame

function [core9,Core,CoreI]= core9_extractor(objNum,x1,y1,w1,h1,x2,y2,w2,h2,lastC,lastCI)

% core9: Extracted core9 features for the current frame
%Core: The area for all 9 cores in the current frame
%CoreI: The intervals for all 9 cores in the current frame

global row col %frame row and column
%row= 50; col= 100;

% 0:empty-box / 1:A / 2:B / 3:AB / 4:no-box
core9= zeros(1,39);
if(objNum==1)
    if    (x1>0   && ((x1+w1)<col)  && y1>0  && ((y1+h1)<row) )
        core9(1:9)= [0,0,0,0,1,0,0,0,0];
    elseif(x1>0   && ((x1+w1)<col)  && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,0,0,4,1,0,4,0,0];
    elseif(x1>0   && ((x1+w1)<col)  && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [0,0,4,0,1,4,0,0,4];    
    elseif(x1>0   && ((x1+w1)<col)  && y1==0 && ((y1+h1)==row))
        core9(1:9)= [0,0,4,0,1,4,4,4,4];    
    elseif(x1>0   && ((x1+w1)==col) && y1>0  &&  ((y1+h1)<row))
        core9(1:9)= [0,0,0,0,1,0,4,4,4];
    elseif(x1>0   && ((x1+w1)==col) && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,0,0,4,1,0,4,4,4];
    elseif(x1>0   && ((x1+w1)==col) && y1==0 && ((y1+h1)==row))
        core9(1:9)= [4,0,4,4,1,4,4,4,4];
    elseif(x1>0   && ((x1+w1)==col) && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [0,0,4,0,1,4,0,4,4];
        
    elseif(x1==0  && ((x1+w1)<col)  && y1>0  && ((y1+h1)<row) )
        core9(1:9)= [4,4,4,0,1,0,0,0,0];
    elseif(x1==0  && ((x1+w1)<col)  && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,0,4,0,0];
    elseif(x1==0  && ((x1+w1)<col)  && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [4,4,4,0,1,4,0,0,4];
    elseif(x1==0  && ((x1+w1)<col)  && y1==0 && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,4,4,0,4];
    elseif(x1==0  && ((x1+w1)==col) && y1>0  &&  ((y1+h1)<row))
        core9(1:9)= [4,4,4,0,1,0,4,4,4];
    elseif(x1==0  && ((x1+w1)==col) && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,0,4,4,4];
    elseif(x1==0  && ((x1+w1)==col) && y1==0 && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,4,4,4,4]; 
    elseif(x1==0  && ((x1+w1)==col) && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [4,4,4,0,1,4,4,4,4];
    end
    
    w= [x1,w1,col-(x1+w1)];
    h= [y1,h1,row-(y1+h1)];
    
    temp= w'*h;
    temp= reshape(temp',1,9); % we have 9 cores
    Core= temp;
    
    core9(10:18)= ranking(temp,0); 
    core9(25:33)= sign(temp-lastC);   
    
    %............................
    temp= [w,h];
    CoreI= temp;
    core9(34:39)= sign(temp-lastCI);
    core9(19:24)= ranking(temp,0);

elseif(objNum==2)
    bigW1= [w1, x2-(x1+w1), w2];
    bigW2= [x2-x1, x1+w1-x2, (x2+w2)-(x1+w1)];
    bigW3= [x1-x2, w1, (x2+w2)-(x1+w1)];
    bigW4= [x2-x1, w2, (x1+w1)-(x2+w2)];
    bigW5= [x1-x2, (x2+w2)-x1, (x1+w1)-(x2+w2)];
    bigW6= [w2, x1-(x2+w2), w1];
        
    bigH1= [h1, y1-(y2+h2), h2];
    bigH2= [(y1+h1)-(y2+h2), (y2+h2)-y1, y1-y2];
    bigH3= [(y2+h2)-(y1+h1), h1, y1-y2];
    bigH4= [(y1+h1)-(y2+h2), h2, y2-y1];
    bigH5= [(y2+h2)-(y1+h1), (y1+h1)-y2, y2-y1];
    bigH6= [h2, y2-(y1+h1), h1];
        
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(y1>(y2+h2))    % row1
        h= bigH1;
        if((x1+w1)<x2)
            core9(1:9)= [1,0,0,0,0,0,0,0,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,0,0,4,4,4,0,0,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,0,0,1,0,2,0,0,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,0,2,0,0,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,0,2,1,0,2,0,0,2];
            w= bigW3; 
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,0,2,1,0,2,4,4,4];
            w= bigW3; 
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,0,2,1,0,0];
            w= bigW5; 
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,0,0,1,0,2,1,0,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,0,0,1,0,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,0,2,1,0,2,1,0,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,0,2,4,4,4,1,0,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,0,2,0,0,0,1,0,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,0,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);


    elseif(y1==(y2+h2))  % row2
        h= bigH1;
        if((x1+w1)<x2)
            core9(1:9)= [1,4,0,0,4,0,0,4,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,4,0,4,4,4,0,4,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,4,0,1,4,2,0,4,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,4,2,0,4,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,4,2,1,4,2,0,4,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,4,2,1,4,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,4,2,1,4,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,4,0,1,4,2,1,4,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,4,0,1,4,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,4,2,1,4,2,1,4,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,4,2,4,4,4,1,4,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,4,2,0,4,0,1,4,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,4,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end

        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y2<y1) && (y1<(y2+h2)) && ((y2+h2)<(y1+h1)))  % row3
        h= bigH2;
        if((x1+w1)<x2)
            core9(1:9)= [1,1,0,0,0,0,0,2,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,1,0,4,4,4,0,2,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,1,0,1,3,2,0,2,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,3,2,0,2,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,2,2,1,3,2,0,2,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,2,2,1,3,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,3,2,1,1,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,1,0,1,3,2,1,1,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,1,0,1,3,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,2,2,1,3,2,1,1,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,2,2,4,4,4,1,1,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,2,2,0,0,0,1,1,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,3,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1>y2) && ((y1+h1)==(y2+h2)))  % row4
        h= bigH2;
        if((x1+w1)<x2)
            core9(1:9)= [4,1,0,4,0,0,4,2,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [4,1,0,4,4,4,4,2,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,1,0,4,3,2,4,2,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,4,3,2,4,2,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,2,2,4,3,2,4,2,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,2,2,4,3,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,4,3,2,4,1,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,1,0,4,3,2,4,1,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,1,0,4,3,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,2,2,4,3,2,4,1,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [4,2,2,4,4,4,4,1,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [4,2,2,4,0,0,4,1,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,4,3,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1>y2) && ((y1+h1)<(y2+h2)))   % row5
        h= bigH3;
        if((x1+w1)<x2)
            core9(1:9)= [0,1,0,0,0,0,2,2,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,1,0,4,4,4,2,2,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,1,0,2,3,2,2,2,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,3,2,2,2,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,2,2,2,3,2,2,2,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,2,2,2,3,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,3,2,0,1,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,0,2,3,2,0,1,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,1,0,2,3,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,2,2,2,3,2,0,1,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,2,2,4,4,4,0,1,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,2,2,0,0,0,0,1,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,3,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);


    elseif((y1==y2) && ((y1+h1)<(y2+h2)))   % row6
        h= bigH3;
        if((x1+w1)<x2)
            core9(1:9)= [0,1,4,0,0,4,2,2,4];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,1,4,4,4,4,2,2,4];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,1,4,2,3,4,2,2,4];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,3,4,2,2,4];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,2,4,2,3,4,2,2,4];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,2,4,2,3,4,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,3,4,0,1,4];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,4,2,3,4,0,1,4];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,1,4,2,3,4,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,2,4,2,3,4,0,1,4];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,2,4,4,4,4,0,1,4];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,2,4,0,0,4,0,1,4];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,3,4,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1<y2) && ((y1+h1)==(y2+h2)))   % row7
        h= bigH4;
        if((x1+w1)<x2)
            core9(1:9)= [4,1,1,4,0,0,4,2,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [4,1,1,4,4,4,4,2,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,1,1,4,3,1,4,2,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,4,3,1,4,2,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,2,0,4,3,1,4,2,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,2,0,4,3,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,4,3,1,4,1,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,1,1,4,3,1,4,1,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,1,1,4,3,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,2,0,4,3,1,4,1,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [4,2,0,4,4,4,4,1,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [4,2,0,4,0,0,4,1,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,4,3,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1<y2) && ((y1+h1)>(y2+h2)))  % row8
        h= bigH4;
        if((x1+w1)<x2)
            core9(1:9)= [1,1,1,0,0,0,0,2,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,1,1,4,4,4,0,2,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,1,1,1,3,1,0,2,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,3,1,0,2,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,2,0,1,3,1,0,2,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,2,0,1,3,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,3,1,1,1,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,1,1,1,3,1,1,1,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,1,1,1,3,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,2,0,1,3,1,1,1,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,2,0,4,4,4,1,1,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,2,0,0,0,0,1,1,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,3,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1==y2) && (h1>h2))   % row9
        h= bigH4;
        if((x1+w1)<x2)
            core9(1:9)= [1,1,4,0,0,4,0,2,4];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,1,4,4,4,4,0,2,4];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,1,4,1,3,4,0,2,4];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,3,4,0,2,4];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,2,4,1,3,4,0,2,4];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,2,4,1,3,4,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,3,4,1,1,4];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,1,4,1,3,4,1,1,4];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,1,4,1,3,4,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,2,4,1,3,4,1,1,4];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,2,4,4,4,4,1,1,4];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,2,4,0,0,4,1,1,4];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,3,4,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1<y2) && (y2<(y1+h1)) && ((y1+h1)<(y2+h2)))  % row10  
        h= bigH5;
        if((x1+w1)<x2)
            core9(1:9)= [0,1,1,0,0,0,2,2,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,1,1,4,4,4,2,2,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,2,2,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,3,1,2,2,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,2,0,2,3,1,2,2,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,2,0,2,3,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,3,1,0,1,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,0,1,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,0,1,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,2,0,4,4,4,0,1,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,2,0,0,0,0,0,1,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,3,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1+h1)==y2)  % row11
        h= bigH5;
        if((x1+w1)<x2)
            core9(1:9)= [0,4,1,0,4,0,2,4,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,4,1,4,4,4,2,4,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,4,1,2,4,1,2,4,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,4,1,2,4,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,4,0,2,4,1,2,4,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,4,0,2,4,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,4,1,0,4,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,4,1,2,4,1,0,4,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,4,1,2,4,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,4,0,2,4,1,0,4,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,4,0,4,4,4,0,4,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,4,0,0,4,0,0,4,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,4,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1+h1)<y2)  % row12
        h= bigH6;
        if((x1+w1)<x2)
            core9(1:9)= [0,0,1,0,0,0,2,0,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,0,1,4,4,4,2,0,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,0,1,2,0,1,2,0,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,0,1,2,0,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,0,0,2,0,1,2,0,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,0,0,2,0,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,0,1,0,0,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,0,1,2,0,1,0,0,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,0,1,2,0,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,0,0,2,0,1,0,0,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,0,0,4,4,4,0,0,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,0,0,0,0,0,0,0,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,0,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1==y2) && (h1==h2))  % row13
        h= bigH5;
        if((x1+w1)<x2)
            core9(1:9)= [4,1,4,4,0,4,4,2,4];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [4,1,4,4,4,4,4,2,4];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,1,4,4,3,4,4,2,4];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,4,3,4,4,2,4];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,2,4,4,3,4,4,2,4];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,2,4,4,3,4,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,4,3,4,4,1,4];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,1,4,4,3,4,4,1,4];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,1,4,4,3,4,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,2,4,4,3,4,4,1,4];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [4,2,4,4,4,4,4,1,4];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [4,2,4,4,0,4,4,1,4];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,4,3,4,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    end
end
%}


%{
% version number 2
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  23 Aug 2013
% extracting CORE-9 information for each frame

%function [core9,Core,CoreI]= core9_extractor(objNum,x1,y1,w1,h1,x2,y2,w2,h2,lastC,lastCI)
function [core9,Core,CoreI]= core9_extractor2(objNum,rec1,rec2,lastC,lastCI)

% core9: Extracted core9 features for the current frame
%Core: The area for all 9 cores in the current frame
%CoreI: The intervals for all 9 cores in the current frame

global row col %frame row and column
%row= 50; col= 100;

% 0:empty-box / 1:A / 2:B / 3:AB / 4:no-box
core9= zeros(1,39);
if(objNum==1)
    x1= rec1(1);
    y1= rec1(2);
    w1= rec1(3)-rec1(1);
    h1= rec1(4)-rec1(2);

    if    (x1>0   && ((x1+w1)<col)  && y1>0  && ((y1+h1)<row) )
        core9(1:9)= [0,0,0,0,1,0,0,0,0];
    elseif(x1>0   && ((x1+w1)<col)  && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,0,0,4,1,0,4,0,0];
    elseif(x1>0   && ((x1+w1)<col)  && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [0,0,4,0,1,4,0,0,4];    
    elseif(x1>0   && ((x1+w1)<col)  && y1==0 && ((y1+h1)==row))
        core9(1:9)= [0,0,4,0,1,4,4,4,4];    
    elseif(x1>0   && ((x1+w1)==col) && y1>0  &&  ((y1+h1)<row))
        core9(1:9)= [0,0,0,0,1,0,4,4,4];
    elseif(x1>0   && ((x1+w1)==col) && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,0,0,4,1,0,4,4,4];
    elseif(x1>0   && ((x1+w1)==col) && y1==0 && ((y1+h1)==row))
        core9(1:9)= [4,0,4,4,1,4,4,4,4];
    elseif(x1>0   && ((x1+w1)==col) && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [0,0,4,0,1,4,0,4,4];
        
    elseif(x1==0  && ((x1+w1)<col)  && y1>0  && ((y1+h1)<row) )
        core9(1:9)= [4,4,4,0,1,0,0,0,0];
    elseif(x1==0  && ((x1+w1)<col)  && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,0,4,0,0];
    elseif(x1==0  && ((x1+w1)<col)  && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [4,4,4,0,1,4,0,0,4];
    elseif(x1==0  && ((x1+w1)<col)  && y1==0 && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,4,4,0,4];
    elseif(x1==0  && ((x1+w1)==col) && y1>0  &&  ((y1+h1)<row))
        core9(1:9)= [4,4,4,0,1,0,4,4,4];
    elseif(x1==0  && ((x1+w1)==col) && y1>0  && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,0,4,4,4];
    elseif(x1==0  && ((x1+w1)==col) && y1==0 && ((y1+h1)==row))
        core9(1:9)= [4,4,4,4,1,4,4,4,4]; 
    elseif(x1==0  && ((x1+w1)==col) && y1==0 && ((y1+h1)<row) )
        core9(1:9)= [4,4,4,0,1,4,4,4,4];
    end
    
    w= [x1,w1,col-(x1+w1)];
    h= [y1,h1,row-(y1+h1)];
    
    temp= w'*h;
    temp= reshape(temp',1,9); % we have 9 cores
    Core= temp;
    
    core9(10:18)= ranking(temp,0); 
    core9(25:33)= sign(temp-lastC);   
    
    %............................
    temp= [w,h];
    CoreI= temp;
    core9(34:39)= sign(temp-lastCI);
    core9(19:24)= ranking(temp,0);
    
elseif(objNum==2)
    x1= rec1(1);
    y1= rec1(2);
    w1= rec1(3)-rec1(1);
    h1= rec1(4)-rec1(2);
    
    x2= rec2(1);
    y2= rec2(2);
    w2= rec2(3)-rec2(1);
    h2= rec2(4)-rec2(2);
    
    bigW1= [w1, x2-(x1+w1), w2];
    bigW2= [x2-x1, x1+w1-x2, (x2+w2)-(x1+w1)];
    bigW3= [x1-x2, w1, (x2+w2)-(x1+w1)];
    bigW4= [x2-x1, w2, (x1+w1)-(x2+w2)];
    bigW5= [x1-x2, (x2+w2)-x1, (x1+w1)-(x2+w2)];
    bigW6= [w2, x1-(x2+w2), w1];
        
    bigH1= [h1, y1-(y2+h2), h2];
    bigH2= [(y1+h1)-(y2+h2), (y2+h2)-y1, y1-y2];
    bigH3= [(y2+h2)-(y1+h1), h1, y1-y2];
    bigH4= [(y1+h1)-(y2+h2), h2, y2-y1];
    bigH5= [(y2+h2)-(y1+h1), (y1+h1)-y2, y2-y1];
    bigH6= [h2, y2-(y1+h1), h1];
        
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(y1>(y2+h2))    % row1
        h= bigH1;
        if((x1+w1)<x2)
            core9(1:9)= [1,0,0,0,0,0,0,0,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,0,0,4,4,4,0,0,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,0,0,1,0,2,0,0,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,0,2,0,0,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,0,2,1,0,2,0,0,2];
            w= bigW3; 
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,0,2,1,0,2,4,4,4];
            w= bigW3; 
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,0,2,1,0,0];
            w= bigW5; 
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,0,0,1,0,2,1,0,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,0,0,1,0,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,0,2,1,0,2,1,0,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,0,2,4,4,4,1,0,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,0,2,0,0,0,1,0,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,0,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);


    elseif(y1==(y2+h2))  % row2
        h= bigH1;
        if((x1+w1)<x2)
            core9(1:9)= [1,4,0,0,4,0,0,4,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,4,0,4,4,4,0,4,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,4,0,1,4,2,0,4,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,4,2,0,4,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,4,2,1,4,2,0,4,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,4,2,1,4,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,4,2,1,4,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,4,0,1,4,2,1,4,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,4,0,1,4,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,4,2,1,4,2,1,4,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,4,2,4,4,4,1,4,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,4,2,0,4,0,1,4,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,4,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end

        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y2<y1) && (y1<(y2+h2)) && ((y2+h2)<(y1+h1)))  % row3
        h= bigH2;
        if((x1+w1)<x2)
            core9(1:9)= [1,1,0,0,0,0,0,2,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,1,0,4,4,4,0,2,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,1,0,1,3,2,0,2,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,3,2,0,2,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,2,2,1,3,2,0,2,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,2,2,1,3,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,3,2,1,1,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,1,0,1,3,2,1,1,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,1,0,1,3,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,2,2,1,3,2,1,1,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,2,2,4,4,4,1,1,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,2,2,0,0,0,1,1,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,3,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1>y2) && ((y1+h1)==(y2+h2)))  % row4
        h= bigH2;
        if((x1+w1)<x2)
            core9(1:9)= [4,1,0,4,0,0,4,2,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [4,1,0,4,4,4,4,2,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,1,0,4,3,2,4,2,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,4,3,2,4,2,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,2,2,4,3,2,4,2,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,2,2,4,3,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,4,3,2,4,1,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,1,0,4,3,2,4,1,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,1,0,4,3,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,2,2,4,3,2,4,1,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [4,2,2,4,4,4,4,1,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [4,2,2,4,0,0,4,1,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,4,3,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1>y2) && ((y1+h1)<(y2+h2)))   % row5
        h= bigH3;
        if((x1+w1)<x2)
            core9(1:9)= [0,1,0,0,0,0,2,2,2];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,1,0,4,4,4,2,2,2];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,1,0,2,3,2,2,2,2];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,3,2,2,2,2];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,2,2,2,3,2,2,2,2];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,2,2,2,3,2,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,3,2,0,1,0];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,0,2,3,2,0,1,0];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,1,0,2,3,2,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,2,2,2,3,2,0,1,0];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,2,2,4,4,4,0,1,0];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,2,2,0,0,0,0,1,0];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,3,2,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);


    elseif((y1==y2) && ((y1+h1)<(y2+h2)))   % row6
        h= bigH3;
        if((x1+w1)<x2)
            core9(1:9)= [0,1,4,0,0,4,2,2,4];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,1,4,4,4,4,2,2,4];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,1,4,2,3,4,2,2,4];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,3,4,2,2,4];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,2,4,2,3,4,2,2,4];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,2,4,2,3,4,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,3,4,0,1,4];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,4,2,3,4,0,1,4];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,1,4,2,3,4,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,2,4,2,3,4,0,1,4];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,2,4,4,4,4,0,1,4];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,2,4,0,0,4,0,1,4];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,3,4,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1<y2) && ((y1+h1)==(y2+h2)))   % row7
        h= bigH4;
        if((x1+w1)<x2)
            core9(1:9)= [4,1,1,4,0,0,4,2,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [4,1,1,4,4,4,4,2,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,1,1,4,3,1,4,2,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,4,3,1,4,2,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,2,0,4,3,1,4,2,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,2,0,4,3,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,4,3,1,4,1,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,1,1,4,3,1,4,1,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,1,1,4,3,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,2,0,4,3,1,4,1,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [4,2,0,4,4,4,4,1,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [4,2,0,4,0,0,4,1,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,4,3,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1<y2) && ((y1+h1)>(y2+h2)))  % row8
        h= bigH4;
        if((x1+w1)<x2)
            core9(1:9)= [1,1,1,0,0,0,0,2,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,1,1,4,4,4,0,2,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,1,1,1,3,1,0,2,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,3,1,0,2,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,2,0,1,3,1,0,2,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,2,0,1,3,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,3,1,1,1,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,1,1,1,3,1,1,1,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,1,1,1,3,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,2,0,1,3,1,1,1,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,2,0,4,4,4,1,1,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,2,0,0,0,0,1,1,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,3,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1==y2) && (h1>h2))   % row9
        h= bigH4;
        if((x1+w1)<x2)
            core9(1:9)= [1,1,4,0,0,4,0,2,4];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [1,1,4,4,4,4,0,2,4];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [1,1,4,1,3,4,0,2,4];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,1,3,4,0,2,4];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,2,4,1,3,4,0,2,4];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,2,4,1,3,4,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,1,3,4,1,1,4];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [1,1,4,1,3,4,1,1,4];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [1,1,4,1,3,4,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,2,4,1,3,4,1,1,4];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [0,2,4,4,4,4,1,1,4];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [0,2,4,0,0,4,1,1,4];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,1,3,4,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1<y2) && (y2<(y1+h1)) && ((y1+h1)<(y2+h2)))  % row10  
        h= bigH5;
        if((x1+w1)<x2)
            core9(1:9)= [0,1,1,0,0,0,2,2,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,1,1,4,4,4,2,2,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,2,2,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,3,1,2,2,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,2,0,2,3,1,2,2,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,2,0,2,3,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,3,1,0,1,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,0,1,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,1,1,2,3,1,0,1,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,2,0,4,4,4,0,1,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,2,0,0,0,0,0,1,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,3,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1+h1)==y2)  % row11
        h= bigH5;
        if((x1+w1)<x2)
            core9(1:9)= [0,4,1,0,4,0,2,4,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,4,1,4,4,4,2,4,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,4,1,2,4,1,2,4,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,4,1,2,4,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,4,0,2,4,1,2,4,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,4,0,2,4,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,4,1,0,4,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,4,1,2,4,1,0,4,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,4,1,2,4,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,4,0,2,4,1,0,4,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,4,0,4,4,4,0,4,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,4,0,0,4,0,0,4,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,4,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1+h1)<y2)  % row12
        h= bigH6;
        if((x1+w1)<x2)
            core9(1:9)= [0,0,1,0,0,0,2,0,0];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [0,0,1,4,4,4,2,0,0];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [0,0,1,2,0,1,2,0,0];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,2,0,1,2,0,0];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [2,0,0,2,0,1,2,0,0];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [2,0,0,2,0,1,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,2,0,1,0,0,1];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [0,0,1,2,0,1,0,0,1];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [0,0,1,2,0,1,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [2,0,0,2,0,1,0,0,1];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [2,0,0,4,4,4,0,0,1];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [2,0,0,0,0,0,0,0,1];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,2,0,1,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    elseif((y1==y2) && (h1==h2))  % row13
        h= bigH5;
        if((x1+w1)<x2)
            core9(1:9)= [4,1,4,4,0,4,4,2,4];
            w= bigW1;
        elseif((x1+w1)==x2)
            core9(1:9)= [4,1,4,4,4,4,4,2,4];
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,1,4,4,3,4,4,2,4];
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            core9(1:9)= [4,4,4,4,3,4,4,2,4];
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            core9(1:9)= [4,2,4,4,3,4,4,2,4];
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,2,4,4,3,4,4,4,4];
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            core9(1:9)= [4,4,4,4,3,4,4,1,4];
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,1,4,4,3,4,4,1,4];
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            core9(1:9)= [4,1,4,4,3,4,4,4,4];
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            core9(1:9)= [4,2,4,4,3,4,4,1,4];
            w= bigW5;
        elseif(x1==(x2+w2))
            core9(1:9)= [4,2,4,4,4,4,4,1,4];
            w= bigW5;
        elseif(x1>(x2+w2))
            core9(1:9)= [4,2,4,4,0,4,4,1,4];
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            core9(1:9)= [4,4,4,4,3,4,4,4,4];
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(10:18)= ranking(temp,0); 
        core9(25:33)= sign(temp-lastC);   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(34:39)= sign(temp-lastCI);
        core9(19:24)= ranking(temp,0);

    end
end
%}




%{
% version number 3
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  14 Oct 2013
% extracting 3 AI-features for each frame:
% 1.change-core9  2.change-CoreInterval  3.distance

function [core9,Core,CoreI]= core9_extractor3(objNum,rec1,rec2,lastC,lastCI)

% Inputs:
% objNum: number of objects (1 or 2)
% rec1,rec2: object rectangles [x,y,w,h] 
% lastC,lastCI: 9and6 values vectore for the last Core and Core Intervals
% Outputs:
% core9: 3 features (16-values): 
%                 1.change-core9 (9-values), 
%                 2.change-CoreInterval (6-values), 
%                 3.distance (1-value)
% Core: The area for all 9 cores in the current frame (9-values)
% CoreI: The intervals for all 9 cores in the current frame(6-values)

global row col %frame row and column

% 0:empty-box / 1:A / 2:B / 3:AB / 4:no-box
core9= zeros(1,16); % 16-values
if(objNum==1)
    x1= rec1(1);
    y1= rec1(2);
    w1= rec1(3);
    h1= rec1(4);


    w= [x1,w1,col-(x1+w1)];
    h= [y1,h1,row-(y1+h1)];
    
    temp= w'*h;
    temp= reshape(temp',1,9); % we have 9 cores
    Core= temp;
    
    core9(10:18)= ranking(temp,0); 
    core9(25:33)= sign(temp-lastC);   
    
    %............................
    temp= [w,h];
    CoreI= temp;
    core9(34:39)= sign(temp-lastCI);
    
elseif(objNum==2)
    x1= rec1(1);
    y1= rec1(2);
    w1= rec1(3);
    h1= rec1(4);
    
    x2= rec2(1);
    y2= rec2(2);
    w2= rec2(3);
    h2= rec2(4);
    
    bigW1= [w1, x2-(x1+w1), w2];
    bigW2= [x2-x1, x1+w1-x2, (x2+w2)-(x1+w1)];
    bigW3= [x1-x2, w1, (x2+w2)-(x1+w1)];
    bigW4= [x2-x1, w2, (x1+w1)-(x2+w2)];
    bigW5= [x1-x2, (x2+w2)-x1, (x1+w1)-(x2+w2)];
    bigW6= [w2, x1-(x2+w2), w1];
        
    bigH1= [h1, y1-(y2+h2), h2];
    bigH2= [(y1+h1)-(y2+h2), (y2+h2)-y1, y1-y2];
    bigH3= [(y2+h2)-(y1+h1), h1, y1-y2];
    bigH4= [(y1+h1)-(y2+h2), h2, y2-y1];
    bigH5= [(y2+h2)-(y1+h1), (y1+h1)-y2, y2-y1];
    bigH6= [h2, y2-(y1+h1), h1];
        
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(y1>(y2+h2))    % row1
        h= bigH1;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3; 
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3; 
        elseif((x1==x2) && (w1>w2))
            w= bigW5; 
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   % change-9core 

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif(y1==(y2+h2))  % row2
        h= bigH1;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end

        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y2<y1) && (y1<(y2+h2)) && ((y2+h2)<(y1+h1)))  % row3
        h= bigH2;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1>y2) && ((y1+h1)==(y2+h2)))  % row4
        h= bigH2;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1>y2) && ((y1+h1)<(y2+h2)))   % row5
        h= bigH3;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;


    elseif((y1==y2) && ((y1+h1)<(y2+h2)))   % row6
        h= bigH3;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;


        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1<y2) && ((y1+h1)==(y2+h2)))   % row7
        h= bigH4;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1<y2) && ((y1+h1)>(y2+h2)))  % row8
        h= bigH4;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1==y2) && (h1>h2))   % row9
        h= bigH4;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1<y2) && (y2<(y1+h1)) && ((y1+h1)<(y2+h2)))  % row10  
        h= bigH5;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1+h1)==y2)  % row11
        h= bigH5;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1+h1)<y2)  % row12
        h= bigH6;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2)) 
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1==y2) && (h1==h2))  % row13
        h= bigH5;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    end
    cent= [(x1+(w1/2)),(y1+(h1/2)); (x2+(w2/2)),(y2+(h2/2))]; % rectangles' centroids
    core9(16)= pdist(cent,'euclidean'); % distance
end
%}





% version number 4
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  23 Oct 2013
% extracting 5 AI-features for each frame:
% 1.change-core9  2.change-CoreInterval  3.distance 
% 4.speed-obj1(based on its centroid)   5.speed-obj2

function [core9,Core,CoreI,c1,c2]= ...
    core9_extractor3(objNum,rec1,rec2,lastC,lastCI,cent1,cent2)

% Inputs:
% objNum: number of objects (1 or 2)
% rec1,rec2: object rectangles [x,y,w,h] 
% lastC,lastCI: 9and6 values vectore for the last Core and Core Intervals
% cent1,cent2: last centroids for the rectangles 1 and 2 [cent_row, cent_col]
% Outputs:
% core9: 5 features (20-values): 
%                 1.change-core9 (9-values), 
%                 2.change-CoreInterval (6-values), 
%                 3.distance (1-value)
%                 4.speed-obj1(based on its centroid)(1-value)   
%                 5.speed-obj2(1-value)
%                 6.size of obj1,2(2-values)
% Core: The area for all 9 cores in the current frame (9-values)
% CoreI: The intervals for all 9 cores in the current frame(6-values)

global row col %frame row and column

% 0:empty-box / 1:A / 2:B / 3:AB / 4:no-box
core9= zeros(1,20); % 20-values
if(objNum==1)
    x1= rec1(1);
    y1= rec1(2);
    w1= rec1(3);
    h1= rec1(4);


    w= [x1,w1,col-(x1+w1)];
    h= [y1,h1,row-(y1+h1)];
    
    temp= w'*h;
    temp= reshape(temp',1,9); % we have 9 cores
    Core= temp;
    
    core9(10:18)= ranking(temp,0); 
    core9(25:33)= sign(temp-lastC);   
    
    %............................
    temp= [w,h];
    CoreI= temp;
    core9(34:39)= sign(temp-lastCI);
    
elseif(objNum==2)
    x1= rec1(1);
    y1= rec1(2);
    w1= rec1(3);
    h1= rec1(4);
    
    x2= rec2(1);
    y2= rec2(2);
    w2= rec2(3);
    h2= rec2(4);
    
    bigW1= [w1, x2-(x1+w1), w2];
    bigW2= [x2-x1, x1+w1-x2, (x2+w2)-(x1+w1)];
    bigW3= [x1-x2, w1, (x2+w2)-(x1+w1)];
    bigW4= [x2-x1, w2, (x1+w1)-(x2+w2)];
    bigW5= [x1-x2, (x2+w2)-x1, (x1+w1)-(x2+w2)];
    bigW6= [w2, x1-(x2+w2), w1];
        
    bigH1= [h1, y1-(y2+h2), h2];
    bigH2= [(y1+h1)-(y2+h2), (y2+h2)-y1, y1-y2];
    bigH3= [(y2+h2)-(y1+h1), h1, y1-y2];
    bigH4= [(y1+h1)-(y2+h2), h2, y2-y1];
    bigH5= [(y2+h2)-(y1+h1), (y1+h1)-y2, y2-y1];
    bigH6= [h2, y2-(y1+h1), h1];
        
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if(y1>(y2+h2))    % row1
        h= bigH1;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3; 
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3; 
        elseif((x1==x2) && (w1>w2))
            w= bigW5; 
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   % change-9core 

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif(y1==(y2+h2))  % row2
        h= bigH1;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end

        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y2<y1) && (y1<(y2+h2)) && ((y2+h2)<(y1+h1)))  % row3
        h= bigH2;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1>y2) && ((y1+h1)==(y2+h2)))  % row4
        h= bigH2;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1>y2) && ((y1+h1)<(y2+h2)))   % row5
        h= bigH3;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;


    elseif((y1==y2) && ((y1+h1)<(y2+h2)))   % row6
        h= bigH3;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;


        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1<y2) && ((y1+h1)==(y2+h2)))   % row7
        h= bigH4;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1<y2) && ((y1+h1)>(y2+h2)))  % row8
        h= bigH4;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1==y2) && (h1>h2))   % row9
        h= bigH4;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1<y2) && (y2<(y1+h1)) && ((y1+h1)<(y2+h2)))  % row10  
        h= bigH5;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1+h1)==y2)  % row11
        h= bigH5;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1+h1)<y2)  % row12
        h= bigH6;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2)) 
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    elseif((y1==y2) && (h1==h2))  % row13
        h= bigH5;
        if((x1+w1)<x2)
            w= bigW1;
        elseif((x1+w1)==x2)
            w= bigW1;
        elseif((x1<x2) && (x2<(x1+w1)) && ((x1+w1)<(x2+w2)))
            w= bigW2;
        elseif((x1==x2) && (w1<w2))
            w= bigW2;
        elseif((x1>x2) && ((x1+w1)<(x2+w2)))
            w= bigW3;
        elseif((x1>x2) && ((x1+w1)==(x2+w2)))
            w= bigW3;
        elseif((x1==x2) && (w1>w2))
            w= bigW5;
        elseif((x1<x2) && ((x1+w1)>(x2+w2)))
            w= bigW4;
        elseif((x1<x2) && ((x1+w1)==(x2+w2)))
            w= bigW4;
        elseif((x1>x2) && ((x1+w1)>(x2+w2)))
            w= bigW5;
        elseif(x1==(x2+w2))
            w= bigW5;
        elseif(x1>(x2+w2))
            w= bigW6;
        elseif((x1==x2) && (w1==w2))
            w= bigW5;
        else
            sprintf('error...');
        end
        temp= w'*h;
        temp= reshape(temp',1,9); % we have 9 cores
        Core= temp;

        core9(1:9)= temp-lastC;   

        %............................
        temp= [w,h];
        CoreI= temp;
        core9(10:15)= temp-lastCI;

    end
    cent= [(x1+(w1/2)),(y1+(h1/2)); (x2+(w2/2)),(y2+(h2/2))]; % rectangles' centroids
    core9(16)= pdist(cent,'euclidean'); % distance

    c1= [(x1+(w1/2)),(y1+(h1/2))]; % new centroids for rect1
    c2= [(x2+(w2/2)),(y2+(h2/2))]; % new centroids for rect2
    
    core9(17)= pdist([c1; cent1],'euclidean');
    core9(18)= pdist([c2; cent2],'euclidean');

    core9(19)= w1*h1; % size of obj1
    core9(20)= w2*h2; % size of obj2
end

