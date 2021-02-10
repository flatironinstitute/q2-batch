data {
  int<lower=0> N;    // number of samples
  int<lower=0> R;    // number of replicates
  int<lower=0> B;    // number of batchs
  real depth[N];     // sequencing depths of microbes
  int y[N];          // observed microbe abundances
  int<lower=1, upper=R> ref_ids[N];    // locations of reference replicates
  int<lower=1, upper=B> batch_ids[N];  // batch ids
}

parameters {
  vector[B] batch;           // random effects for each batch
  vector[R] reference;       // reference replicates
  real mu;                   // mean of batch random effects
  real<lower=0.001> sigma;   // variance of batch random effects
  real<lower=0.001> disp;    // per microbe dispersion
}

transformed parameters {
  vector[N] eta;
  vector[N] lam;
  for (n in 1:N){
    eta[n] = batch[batch_ids[n]] + reference[ref_ids[n]];
  }
}

model {
  // setting priors ...
  disp ~ cauchy(0., 5.);       // weak overdispersion prior
  mu ~ normal(0., 5.);        // uninformed batch effects mean prior
  sigma ~ cauchy(0, 5);        // weak batch effects variance prior
  batch ~ normal(mu, sigma);   // random effects
  reference ~ normal(0., 5.); // uninformed reference prior
  // generating counts
  for (n in 1:N){
    lam[n] ~ normal(eta[n], disp);
    target += poisson_log_lpmf(y[n] | lam[n] + depth[n]);
  }
}
