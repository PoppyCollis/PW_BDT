data{
  int<lower=1> nObs; // number of observations total
  int<lower=1> nSs; // number of subjects
  int<lower=1> sidx[nObs]; // index of subjects
  vector[nObs] dPrime; // d' observations
  vector[nObs] metadPrime; // meta-d' observations
  real<lower=0, upper=1> priorMCE; // prior meta-cognitive efficiency (meta-d'/d')
  real mu_low; // lower bound, mu estimates
  real mu_high; // upper bound, mu estimates
  real sig_low; // lower bound, sigma estimates
  real sig_high; // upper bound, sigma estimates
}

parameters{
  vector<lower=mu_low, upper=mu_high>[nSs] mu_subject; // d' estimate for subject
  vector<lower=mu_low, upper=mu_high>[nSs] mu2_subject; // meta-d' estimate for subject
  vector<lower=log(sig_low), upper=log(sig_high)>[nSs] log_sigma_subject; // subject + measurement d
  real<lower=log(sig_low), upper=log(sig_high)> log_sigma_population; // population sd
  real<lower=log(sig_low), upper=log(sig_high)> log_sigma_MCE; // meta-cognitive efficiency sd
}

model{
  mu_subject ~ normal(1,exp(log_sigma_population)); // prior on subject d', normal for population
  for (s in 1:nSs) {
    mu2_subject[s] ~ normal(priorMCE*mu_subject[s],exp(log_sigma_MCE)); // prior per subject on meta-d'
  }
  log_sigma_subject ~ uniform(log(sig_low),log(sig_high)); // flat prior for subject + measurement sd
  log_sigma_population ~ uniform(log(sig_low),log(sig_high)); // flat hyper-prior population sd
  for (n in 1:nObs) {
    dPrime[n] ~ normal(mu_subject[sidx[n]],exp(log_sigma_subject[sidx[n]])); // sampling statement for d' obs
    metadPrime[n] ~ normal(mu2_subject[sidx[n]],exp(log_sigma_subject[sidx[n]])); // sampling statement for meta-d' obs
  }
}

generated quantities{ // get rid of logs on sigma parameter estimates 
  vector[nSs] sigma_subject; 
  real sigma_population;
  real sigma_MCE;
  sigma_subject = exp(log_sigma_subject);
  sigma_population = exp(log_sigma_population);
  sigma_MCE = exp(log_sigma_MCE);
}
