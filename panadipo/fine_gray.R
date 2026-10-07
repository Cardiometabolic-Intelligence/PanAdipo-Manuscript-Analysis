# Fine-Gray subdistribution hazard model (called by panadipo.survival.fine_gray).
# Usage: Rscript fine_gray.R <model data .csv> <comma-separated covariates; the first is the exposure>
# The data have time_years, status (0 censored, 1 endpoint, 2 death before the endpoint), id, and the covariates.
# Prints: coefficient and robust SE of the exposure.
suppressMessages(library(survival))
args <- commandArgs(TRUE)
d <- read.csv(args[1])
covs <- strsplit(args[2], ",")[[1]]
d$status <- factor(d$status, levels = c(0, 1, 2), labels = c("censored", "event", "death"))
fg <- finegray(Surv(time_years, status) ~ ., data = d[, c("time_years", "status", "id", covs)], etype = "event")
f <- as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", paste(covs, collapse = " + "), "+ cluster(id)"))
m <- coxph(f, data = fg, weight = fgwt)
s <- summary(m)$coefficients
cat(sprintf("%.6f,%.6f\n", s[covs[1], "coef"], s[covs[1], "robust se"]))
