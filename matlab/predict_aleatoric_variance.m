function var_pred = predict_aleatoric_variance(Xnew, model)
% Predicted aleatoric VARIANCE r(x) from the fitted polynomial (extracted verbatim from the
% local function in heter_lvgp/bayesian_optimizer.m / acquisition_func.m).
poly = model.aleatoric_poly;
Wnew = zeros(size(Xnew,1), numel(poly.cont_idx) + size(poly.Z_latent,2));
for ii = 1:size(Xnew,1)
    c = Xnew(ii, poly.ind_qual);
    Wnew(ii,:) = [Xnew(ii, poly.cont_idx), poly.Z_latent(c,:)];
end
Wn = (Wnew - poly.mu_W) ./ poly.std_W;
Phi = build_poly_features(Wn, poly.degree);
log_sigma = Phi * poly.theta;
var_pred = exp(2*log_sigma);
var_pred = max(var_pred, 1e-12);
end
