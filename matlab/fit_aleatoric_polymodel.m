function poly = fit_aleatoric_polymodel(X, y_var, model, bo_options)
% Fit log(sigma) = theta' Phi([x_cont, latent categorical coordinates]).
% Extracted VERBATIM from heter_lvgp/bayesian_optimizer.m (local function) so the modeling
% study can fit the aleatoric polynomial outside the BO loop.
ind_qual = model.data.ind_qual;
d = size(X,2);
cont_idx = setdiff(1:d, ind_qual);
Z_latent = model.qual_param.z{1};

W = zeros(size(X,1), numel(cont_idx) + size(Z_latent,2));
for ii = 1:size(X,1)
    c = X(ii, ind_qual);
    W(ii,:) = [X(ii, cont_idx), Z_latent(c,:)];
end

mu_W = mean(W, 1);
std_W = std(W, 0, 1);
std_W(std_W == 0) = 1;
Wn = (W - mu_W) ./ std_W;

Phi = build_poly_features(Wn, bo_options.poly_degree);
log_sigma = 0.5 * log(max(y_var, 1e-12));
lambda = bo_options.poly_lambda;
theta = (Phi' * Phi + lambda * eye(size(Phi,2))) \ (Phi' * log_sigma);

poly.theta = theta;
poly.degree = bo_options.poly_degree;
poly.lambda = lambda;
poly.mu_W = mu_W;
poly.std_W = std_W;
poly.ind_qual = ind_qual;
poly.cont_idx = cont_idx;
poly.Z_latent = Z_latent;
end
