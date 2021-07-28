data {
  int<lower=0> N;                      // number of samples
  int<lower=0> R;                      // number of replicates
  int<lower=0> B;                      // number of batchs
  real depth[N];                       // sequencing depths of microbes
  int<lower=0> y[N];                   // observed microbe abundances
  int<lower=1, upper=R> ref_ids[N];    // locations of reference replicates
  int<lower=1, upper=B> batch_ids[N];  // batch ids
  real sigma_scale;
  real disp_scale;
  real reference_loc;
  real reference_scale;

}

parameters {
  vector[B] batch;             // random effects for each batch
  vector[R] reference;         // reference replicates
  real<lower=0.0> sigma;       // variance of batch random effects
  real<lower=0.0> disp;        // per microbe dispersion
  vector[N] lam;
}

transformed parameters{
  vector[C] logit_ref;
  vector[N] eta;
  logit_ref = log(inv_logit(reference));
  for (n in 1:N) {
    eta[n] = batch[batch_ids[n]] + logit_ref[ref_ids[n]];
  }
}

model {
  vector[N] eta;
  // setting priors ...
  disp ~ lognormal(0., disp_scale);   // weak overdispersion prior
  sigma ~ lognormal(0., sigma_scale); // strong batch effects variance prior
  batch ~ normal(0, sigma);           // random effects
  // uninformed reference prior
  ref_mu ~ normal(reference_loc, 5);
  ref_sigma ~ lognormal(0, reference_scale);
  reference ~ normal(ref_mu, ref_sigma);
  // sample logits and counts
  lam ~ normal(eta, disp);
  y ~ poisson_log(lam + to_vector(depth));
}

generated quantities {
  vector[N] y_predict;
  vector[N] log_lhood;
  for (n in 1:N){
    real lam_ = normal_rng(eta, disp);
    y_predict[n] = poisson_log_rng(lam_ + depth[n]);
    log_lhood[n] = poisson_log_lpmf(y[n] | lam_ + depth[n]);
  }
}
