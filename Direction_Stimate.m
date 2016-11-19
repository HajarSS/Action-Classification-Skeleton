% In the Name of GOD...
% 11 Sep 2013

function d = Direction_Stimate(x, y)

if (x==0 && y>0)        d= 30;
elseif(x==0 && y<0)     d= 70;
elseif(x>0 && y==0)     d= 10;
elseif(x<0 && y==0)     d= 50;
elseif(x==0 && y==0)    d= 0;
else

	r= abs(x/y);
	if     ((x>0)&&(y>0)) 
       		if (r>6)         d= 10;
       		elseif(r<0.15)      d= 30;
       		else                  d= 20;
       		end
	elseif ((x<0)&&(y>0))   
       		if (r>6)         d= 50;
       		elseif(r<0.15)      d= 30;
       		else                  d= 40;
       		end
	elseif ((x<0)&&(y<0))   
       		if (r>6)         d= 50;
       		elseif(r<0.15)      d= 70;
       		else                  d= 60;
       		end
	elseif ((x>0)&&(y<0))   
       		if (r>6)         d= 10;
       		elseif(r<0.15)      d= 70;
       		else                  d= 80;
       		end
	else			      d= 0;
	end
end


if(d<10 || d>80) d=0; end


