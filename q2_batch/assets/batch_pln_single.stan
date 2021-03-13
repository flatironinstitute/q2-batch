data {
  int<lower=0> N;    // number of samples
  int<lower=0> R;    // number of replicates
  int<lower=0> B;    // number of batchs
  real depth[N];     // sequencing depths of microbes
  int<lower=0> y[N];          // observed microbe abundances
  int<lower=1, upper=R> ref_ids[N];    // locations of reference replicates
  int<lower=1, upper=B> batch_ids[N];  // batch ids
}

parameters {
  vector[B] batch;           // random effects for each batch
  vector[R] reference;       // reference replicates
  real mu;                   // mean of batch random effects
  real<lower=0.0> sigma;     // variance of batch random effects
  real<lower=0.0> disp;      // per microbe dispersion
  vector[N] lam;
}

model {
  vector[N] eta;
  // setting priors ...
  disp ~ normal(0., 5);            // weak overdispersion prior
  mu ~ normal(0., 10.);            // uninformed batch effects mean prior
  sigma ~ normal(0., 5);           // weak batch effects variance prior
  batch ~ normal(mu, sigma);       // random effects
  reference ~ normal(0., 10.);     // uninformed reference prior
  // generating counts
  for (n in 1:N){
    eta[n] = batch[batch_ids[n]] + reference[ref_ids[n]];
  }
  lam ~ normal(eta, disp);
  y ~ poisson_log(lam + to_vector(depth));
}
