function mfit(train_mat, out_mat, model_type)
% Fit a MATLAB LVGP for the modeling study (fit only; predictions via mpredict.m).
%   train_mat: .mat with X (n x (d+1), level = LAST column), Y (n x 1), noise_var (n x 1)
%   model_type: 'standard' (LVGP_fit, homoscedastic) | 'heter' (LVGP_fit_noise + aleatoric poly)
addpath('standard_lvgp'); addpath('heter_lvgp');
S = load(train_mat);
X = double(S.X); Y = double(S.Y(:)); noise_var = double(S.noise_var(:));
ind_qual = size(X, 2);

if strcmp(model_type, 'heter')
    model = LVGP_fit_noise(X, Y, noise_var, 'ind_qual', ind_qual);
    bo_options = struct('poly_degree', 2, 'poly_lambda', 1e-3);
    model.aleatoric_poly = fit_aleatoric_polymodel(X, noise_var, model, bo_options);
    model.y_var_sample = noise_var;
else
    model = LVGP_fit(X, Y, 'ind_qual', ind_qual);
end
save(out_mat, 'model', 'model_type', 'noise_var', '-v7');
end
