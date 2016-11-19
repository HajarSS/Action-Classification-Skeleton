function hists = scaleDescs(hists)
hists_scaled = 0.*hists;
for i =1:size(hists,2)
    mi = min(hists(:,i));
    ma = max(hists(:,i));
    
    if (ma ~= 0)
        hists_scaled(:,i) = 2*((hists(:,i) - mi)/(ma-mi)) - 1;
    else
        hists_scaled(:,i) = hists(:,i);
    end
end
hists = hists_scaled;