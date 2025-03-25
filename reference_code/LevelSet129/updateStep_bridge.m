%% -- Design Update --
function [struc, lsf] = updateStep_bridge(lsf, shapeSens, topSens, stepLength, topweight)
    % Smooth the sensitivities
    [shapeSens] = conv2(padarray(shapeSens, [1,1], 'replicate'), 1/6*[0 1 0; 1 2 1; 0 1 0], 'valid');
    [topSens] = conv2(padarray(topSens, [1,1], 'replicate'), 1/6*[0 1 0; 1 2 1; 0 1 0], 'valid');
    % Load bearing pixels must remain solid - simple bridge
    shapeSens(end, [1, round(end/2):round(end/2+1), end]) = 0;
    topSens(end, [1, round(end/2):round(end/2+1), end]) = 0;
    % Design update via evolution
    [struc, lsf] = evolve(-shapeSens, topSens.*(lsf(2:end-1,2:end-1)<0), lsf, stepLength, topweight);
end