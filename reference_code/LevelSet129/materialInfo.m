%% Material Information
function [KE, KTr, lambda, mu] = materialInfo()
    % Set material parameters, find Lame values
    E = 1.; nu = 0.3;
    % 需要思考为什么表达式不对
    lambda = E*nu/( (1+nu)*(1-nu) ); 
    % lambda = E*nu/( (1+nu)*(1-2*nu) ); 
    mu = E/(2*(1+nu));
    % Find stiffiness matrix "KE"
    k = [1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ...,
        -1/4+nu/12 -1/8-nu/8      nu/6   1/8-3*nu/8];
    KE = E/(1-nu^2)*stiffnessMatrix(k);
    % Find "trace" matrix "KTr"
    k = [1/3 1/4 -1/3 1/4 -1/6 -1/4 1/6 -1/4];
    KTr = E/(1-nu)*stiffnessMatrix(k);
end
