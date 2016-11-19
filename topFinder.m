% In the Name of GOD...
% 11 Feb 2014

function top = topFinder(s_a, e_a, s_b, e_b)
% There are 24 relations

% sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
% position= -1: right, 0: on the line, 1: left

% A ? s_b
As_b=sign((e_a(1)-s_a(1))*(s_b(2)-s_a(2))-(e_a(2)-s_a(2))*(s_b(1)-s_a(1))); 

% A ? e_b
Ae_b=sign((e_a(1)-s_a(1))*(e_b(2)-s_a(2))-(e_a(2)-s_a(2))*(e_b(1)-s_a(1))); 

% B ? s_a
Bs_a=sign((e_b(1)-s_b(1))*(s_a(2)-s_b(2))-(e_b(2)-s_b(2))*(s_a(1)-s_b(1))); 

% B ? e_a
Be_a=sign((e_b(1)-s_b(1))*(e_a(2)-s_b(2))-(e_b(2)-s_b(2))*(e_a(1)-s_b(1))); 


if ((As_b<0) && (Ae_b<0) && (Bs_a<0) && (Be_a<0))
    top= 1;    % A rrrr B
elseif ((As_b<0) && (Ae_b<0) && (Bs_a<0) && (Be_a>0))
    top= 2;    % A rrrl B
elseif ((As_b<0) && (Ae_b<0) && (Bs_a>0) && (Be_a<0))
    top= 3;    % A rrlr B
elseif ((As_b<0) && (Ae_b<0) && (Bs_a>0) && (Be_a>0))
    top= 4;    % A rrll B
elseif ((As_b<0) && (Ae_b>0) && (Bs_a<0) && (Be_a<0))
    top= 5;    % A rlrr B
elseif ((As_b<0) && (Ae_b>0) && (Bs_a>0) && (Be_a<0))
    top= 6;    % A rllr B
elseif ((As_b<0) && (Ae_b>0) && (Bs_a>0) && (Be_a>0))
    top= 7;    % A rlll B
elseif ((As_b>0) && (Ae_b<0) && (Bs_a<0) && (Be_a<0))
    top= 8;    % A lrrr B
elseif ((As_b>0) && (Ae_b<0) && (Bs_a<0) && (Be_a>0))
    top= 9;    % A lrrl B 
elseif ((As_b>0) && (Ae_b<0) && (Bs_a>0) && (Be_a>0))
    top= 10;    % A lrll B
elseif ((As_b>0) && (Ae_b>0) && (Bs_a<0) && (Be_a<0))
    top= 11;    % A llrr B
elseif ((As_b>0) && (Ae_b>0) && (Bs_a<0) && (Be_a>0))
    top= 12;    % A llrl B
elseif ((As_b>0) && (Ae_b>0) && (Bs_a>0) && (Be_a<0))
    top= 13;    % A lllr B
elseif ((As_b>0) && (Ae_b>0) && (Bs_a>0) && (Be_a>0))
    top= 14;    % A llll B
elseif ((As_b==0) && (Ae_b>0) && (Bs_a>0) && (Be_a==0) && isequal(e_a,s_b))
    top= 15;    % A ells B
elseif ((As_b==0) && (Ae_b<0) && (Bs_a<0) && (Be_a==0) && isequal(e_a,s_b))
    top= 16;    % A errs B
elseif ((As_b>0) && (Ae_b==0) && (Bs_a<0) && (Be_a==0) && isequal(e_a,e_b))
    top= 17;    % A lere B
elseif ((As_b<0) && (Ae_b==0) && (Bs_a>0) && (Be_a==0) && isequal(e_a,e_b))
    top= 18;    % A rele B
elseif ((As_b==0) && (Ae_b>0) && (Bs_a==0) && (Be_a<0) && isequal(s_a,s_b))
    top= 19;    % A slsr B
elseif ((As_b==0) && (Ae_b<0) && (Bs_a==0) && (Be_a>0) && isequal(s_a,s_b))
    top= 20;    % A srsl B
elseif ((As_b>0) && (Ae_b==0) && (Bs_a==0) && (Be_a>0) && isequal(s_a,e_b))
    top= 21;    % A lsel B
elseif ((As_b<0) && (Ae_b==0) && (Bs_a==0) && (Be_a<0) && isequal(s_a,e_b))
    top= 22;    % A rser B
elseif (isequal(s_a,s_b) && isequal(e_a,e_b))
    top= 23;    % A sese B
elseif (isequal(s_a,e_b) && isequal(e_a,s_b))
    top= 24;    % A eses B
else
    %error('No topology relations! ;)\n');
    top= 25;
end

% figure(33),
% a=[115 117;140 147];
% plot(a(1,:),a(2,:),'-.');
% hold on
% b= [124 127;122 182];
% plot(b(1,:),b(2,:),'-*');
% hold off