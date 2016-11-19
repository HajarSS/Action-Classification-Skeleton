function repositionAxes( ax )
        % Reposition the axes so that there's room for the labels
        % Note that we only do this if the OuterPosition is the thing being
        % controlled
       % if ~strcmpi( get( ax, 'ActivePositionProperty' ), 'OuterPosition' )
%fprintf('hhhhhhhh\n');
        %    return;
        %end
        
        % Work out the maximum height required for the labels
        labelHeight = getLabelHeight(ax);
        
        % Remove listeners while we mess around with things, otherwise we'll
        % trigger redraws recursively
        removeListeners( ax );
        
        % Change to normalized units for the position calculation
        oldUnits = get( ax, 'Units' );
        set( ax, 'Units', 'Normalized' );
        
        % Not sure why, but the extent seems to be proportional to the height of the axes.
        % Correct that now.
        set( ax, 'ActivePositionProperty', 'Position' );
        pos = get( ax, 'Position' );
        axesHeight = pos(4);
        % Make sure we don't adjust away the axes entirely!
        heightAdjust = min( (axesHeight*0.9), labelHeight*axesHeight );
        
        % Move the axes
        if isappdata( ax, 'OriginalAxesPosition' )
            pos = getappdata( ax, 'OriginalAxesPosition' );
        else
            pos = get(ax,'Position');
            setappdata( ax, 'OriginalAxesPosition', pos );
        end
        if strcmpi( get( ax, 'XAxisLocation' ), 'Bottom' )
            % Move it up and reduce the height
            set( ax, 'Position', pos+[0 heightAdjust 0 -heightAdjust] )
        else
            % Just reduce the height
            set( ax, 'Position', pos+[0 0 0 -heightAdjust] )
        end
        set( ax, 'Units', oldUnits );
        set( ax, 'ActivePositionProperty', 'OuterPosition' );
        
        % Make sure we find out if axes properties are changed
        addListeners( ax );
        
    end % repositionAxes
