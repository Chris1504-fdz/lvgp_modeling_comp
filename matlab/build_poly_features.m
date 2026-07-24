function Phi = build_poly_features(Wn, degree)
% Polynomial features [1, W, W.^2, ..., W.^degree] + pairwise cross terms (verbatim extraction
% of build_poly_features_local from heter_lvgp/bayesian_optimizer.m).
Phi = ones(size(Wn,1), 1);
for deg = 1:degree
    Phi = [Phi, Wn.^deg]; %#ok<AGROW>
end
if size(Wn,2) >= 2
    for a = 1:size(Wn,2)-1
        for b = a+1:size(Wn,2)
            Phi = [Phi, Wn(:,a).*Wn(:,b)]; %#ok<AGROW>
        end
    end
end
end
