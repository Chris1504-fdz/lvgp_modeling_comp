function mpredict(model_mat, pred_mat, out_mat)
% Predict with a model fitted by mfit.m.
%   pred_mat: .mat with Xpred (m x (d+1), level = LAST column)
% Saves: mu (posterior mean, raw units), s2 (epistemic variance), r (aleatoric variance;
% NaN for the standard model).
addpath('standard_lvgp'); addpath('heter_lvgp');
S = load(model_mat); model = S.model; model_type = S.model_type; noise_var = S.noise_var(:);
P = load(pred_mat); Xpred = double(P.Xpred);

if strcmp(model_type, 'heter')
    pred = LVGP_predict_noise(Xpred, noise_var, model, 'MSE_on', true);
    r = predict_aleatoric_variance(Xpred, model);
else
    pred = LVGP_predict(Xpred, model, 'MSE_on', true);
    r = nan(size(Xpred, 1), 1);
end
mu = pred.Y_hat(:);
s2 = diag(pred.MSE); s2 = s2(:);
save(out_mat, 'mu', 's2', 'r', '-v7');
end
