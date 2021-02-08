data {
  int<lower=0> N;    // number of samples
  int<lower=0> R;    // number of replicates
  int<lower=0> B;    // number of batchs
  real depth[N];     // log sequencing depths of microbes
  int y[N];          // observed microbe abundances
  int<lower=1, upper=R> ref_ids[N];    // locations of reference replicates
  int<lower=1, upper=B> batch_ids[N];  // batch ids
}

parameters {
  vector[B] batch;           // random effects for each batch
  vector<lower=0>[R] reference;       // reference replicates
  real mu;                   // mean of batch random effects
  real<lower=0.001> sigma;   // variance of batch random effects
  real<lower=0.001> disp;    // per microbe dispersion
}

transformed parameters {
  vector[N] lam;
  for (n in 1:N){
    lam[n] = batch[batch_ids[n]] + log(reference[ref_ids[n]]);
  }
}

model {
  // setting priors ...
  disp ~ inv_gamma(1., 1.);
  sigma ~ inv_gamma(1., 1.);
  mu ~ normal(0, 10);
  for (b in 1:B){
    batch[b] ~ normal(mu, sigma); // random effects
  }
  for (r in 1:R){
    reference[r] ~ gamma(1, 1); // vague gamma prior (to mimick Dir)
  }
  // generating counts
  for (n in 1:N){
    target += neg_binomial_2_log_lpmf(y[n] | lam[n] + depth[n], disp);
  }
}
