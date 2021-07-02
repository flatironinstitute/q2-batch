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
  vector<upper=0>[R] reference;// reference replicates
  real<lower=0.0> sigma;       // variance of batch random effects
  real<lower=0.0> disp;        // per microbe dispersion
  vector[N] lam;
}

model {
  vector[N] eta;
  // setting priors ...
  disp ~ normal(0., disp_scale);   // weak overdispersion prior
  sigma ~ normal(0., sigma_scale); // strong batch effects variance prior
  batch ~ normal(0, sigma);        // random effects

  // uninformed reference prior
  reference ~ normal(reference_loc, reference_scale);
  // generating counts
  for (n in 1:N){
    eta[n] = batch[batch_ids[n]] + reference[ref_ids[n]];
  }
  lam ~ normal(eta, disp);
  y ~ poisson_log(lam + to_vector(depth));
}

generated quantities {
  vector[N] y_predict;
  vector[N] log_lhood;
  for (n in 1:N){
    real eta_ = batch[batch_ids[n]] + reference[ref_ids[n]];
    real lam_ = normal_rng(eta_, disp);
    y_predict[n] = poisson_log_rng(lam_ + depth[n]);
    log_lhood[n] = poisson_log_lpmf(y[n] | lam_ + depth[n]);
  }
}
