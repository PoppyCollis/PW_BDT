# fit_dPrime_hierarchicalBaye2s.R
# This function fits a hierarchical Bayesian model to the d' data of the Cost and Confidence experiment.
# The d' of each of a subjects 7 sessions are assumed to be random draws from a normal distibution, with
# a width relating to the subject's session-by-session variability, and a mu drawn from a population of 
# d'. This population prior, normal distribution with mu=1 and free sigma parameter, describes the 
# variability of d' across subjects. It is a fair assumption that the population mean is d'=1, given we 
# used an initial thresholding proceedure to target this performance level.
#
# So what's with the 2? Well we are now including meta-d' in this hierarchical model. DESCRIBE HOW...
#
# Created by SML Nov 2017

# Preamble:
options(encoding = "UTF-8")
library("dplyr")
library("rstan")
library("shinystan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Set directory (default: work, will switch to home if dir not found):
dataPath <- "/Local/Users/locke/GoogleDrive/Library/Experiments/ConfidenceProjects/CostAndConfidence/DataAnalysis"
if (!dir.exists(dataPath)) {dataPath <- "/Users/shannonlocke/GoogleDrive/Library/Experiments/ConfidenceProjects/CostAndConfidence/DataAnalysis"}
setwd(dataPath)

# Read data:
fname <- "/data/fit_dPrime_hierarchicalBayes_rawData.txt"
fetchFile <- paste(dataPath, fname, sep = "")
indata <- read.table(fetchFile, header = T)
indata <- indata[order(indata$sidx),] # tidy up

# Data info:
nObs <- length(indata$sID)
nSs <- length(unique(indata$sID))

# Get MCMC sampling instructions and boundaries:
iter <- 4000
chains <- 4
mu_range <- c(0, 3)
sigma_range <- c(0.1, 5)

# Create list for rstan script:
data <- list(nObs = nObs,
             nSs = nSs,
             sidx = indata$sidx,
             dPrime = indata$dPrime,
             metadPrime = indata$metadPrime,
             priorMCE = 0.8,
             mu_low = mu_range[1],
             mu_high = mu_range[2],
             sig_low = sigma_range[1],
             sig_high = sigma_range[2])

# Fit stan model:
fit <- stan(file="model_dPrime_hierarchicalBayes2.stan", data=data, iter=iter, chains=chains)

# See fit summary and output samples:
# print(fit)
# plot(fit)
# traceplot(fit)
# pairs(fit)
# launch_shinystan(fit)
sims <- rstan::extract(fit)

# Collate results:
muEst <- colMeans(sims$mu_subject)
muLow95CI <- sims$mu_subject %>%  apply(2, quantile, 0.025)
muHigh95CI <- sims$mu_subject %>% apply(2, quantile, 0.975)
mu2Est <- colMeans(sims$mu2_subject)
mu2Low95CI <- sims$mu2_subject %>%  apply(2, quantile, 0.025)
mu2High95CI <- sims$mu2_subject %>% apply(2, quantile, 0.975)
sigmaEst <- colMeans(sims$sigma_subject)
sigmaLow95CI <- sims$sigma_subject %>%  apply(2, quantile, 0.025)
sigmaHigh95CI <- sims$sigma_subject %>% apply(2, quantile, 0.975)
sigmaPop <- mean(sims$sigma_population)
sigmaPopLow95CI <- quantile(sims$sigma_population, 0.025)
sigmaPopHigh95CI <- quantile(sims$sigma_population, 0.975)
sigmaMCE <- mean(sims$sigma_MCE)
sigmaMCELow95CI <- quantile(sims$sigma_MCE, 0.025)
sigmaMCEHigh95CI <- quantile(sims$sigma_MCE, 0.975)

# Store results in data frame:
res <- data.frame(sID = unique(indata$sID))
res$muEst <- muEst
res$muLow95CI <- muLow95CI
res$muHigh95CI <- muHigh95CI
res$mu2Est <- mu2Est
res$mu2Low95CI <- mu2Low95CI
res$mu2High95CI <- mu2High95CI
res$sigmaEst <- sigmaEst
res$sigmaLow95CI <- sigmaLow95CI
res$sigmaHigh95CI <- sigmaHigh95CI
res$sigmaPop <- sigmaPop
res$sigmaPopLow95CI <- sigmaPopLow95CI
res$sigmaPopHigh95CI <- sigmaPopHigh95CI
res$sigmaMCE <- sigmaMCE
res$sigmaMCELow95CI <- sigmaMCELow95CI
res$sigmaMCEHigh95CI <- sigmaMCEHigh95CI

# Export data:
fname <- "/data/fit_dPrime_hierarchicalBayes_fitData2.txt"
newFile <- paste(dataPath, fname, sep = "")
write.table(res, newFile, quote = FALSE, row.names = FALSE, sep=" ")
