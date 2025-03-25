%% --Evolution of Level-Set Function
function [struc, lsf] = evolve(v, g, lsf, stepLength, w)
    % Extend sensitivities using a zero border
    vFull = zeros(size(v)+2); vFull(2:end-1, 2:end-1) = v;
    gFull = zeros(size(g)+2); gFull(2:end-1, 2:end-1) = g;
    % Choose time step for evolution based on CFL value
    dt = 0.1/max(abs(v(:)));
    % Evolve for total time stepLength * CFL value
    for i = 1:(10*stepLength)
        % Calculate derivatives on the grid
        dpx = circshift(lsf, [0, -1]) - lsf;
        dmx = lsf - circshift(lsf, [0, 1]);
        dpy = circshift(lsf, [-1, 0]) - lsf;
        dmy = lsf - circshift(lsf, [1, 0]);
        % Update level set function using an upwind scheme
        lsf = lsf - dt * min(vFull, 0).*...,
            sqrt( min(dmx, 0).^2+max(dpx, 0).^2+min(dmy, 0).^2+max(dpy, 0).^2 )...,
            - dt * max(vFull, 0) .* ...,
            sqrt( max(dmx, 0).^2+min(dpx, 0).^2+max(dmy,0).^2+min(dpy, 0).^2 )...,
            - w*dt*gFull;
    end
    % New structure obtanied from lsf
    strucFULL = (lsf<0); struc = strucFULL(2:end-1, 2:end-1);
    % strucFULL = (lsf>0); struc = strucFULL(2:end-1, 2:end-1);
end