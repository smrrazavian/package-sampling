# Generate the golden corpus from R's sampling 2.10 (ADR-0006).
#
# Every design here consumes uniforms one-for-one with its Python counterpart,
# so recording R's runif draws lets Python replay the same stream and compare
# value-for-value rather than only in distribution.

S <- "/w/sampling/R"
for (f in c("as_int.r", "inclusionprobabilities.R", "UPbrewer.R", "UPtille.R",
            "UPsystematic.R", "UPpoisson.R", "UPMEqfromw.R", "UPMEpikfromq.R",
            "UPMEpiktildefrompik.R", "UPMEsfromq.R", "UPMEpik2frompikw.R",
            "UPmaxentropy.R", "UPtillepi2.R", "UPsystematicpi2.R",
            "UPmaxentropypi2.R")) source(file.path(S, f))

.u_log <- numeric(0)

# Shadows base::runif for the sourced designs, which resolve it lexically
# through globalenv. Records every draw so Python can replay the stream.
runif <- function(n = 1, ...) {
  v <- stats::runif(n, ...)
  .u_log <<- c(.u_log, v)
  v
}

num <- function(x) paste(sprintf("%.17g", as.numeric(x)), collapse = " ")

emit <- function(name, fn, pik, seed = NULL, extra = NULL) {
  .u_log <<- numeric(0)
  if (!is.null(seed)) set.seed(seed)
  res <- tryCatch(
    withCallingHandlers(
      if (is.null(extra)) fn(pik) else do.call(fn, c(list(pik), extra)),
      warning = function(w) invokeRestart("muffleWarning")
    ),
    error = function(e) structure(conditionMessage(e), class = "cerr")
  )
  cat("### ", name, "\n", sep = "")
  cat("pik ", num(pik), "\n", sep = "")
  if (!is.null(extra)) cat("extra ", num(unlist(extra)), "\n", sep = "")
  if (length(.u_log)) cat("u ", num(.u_log), "\n", sep = "")
  if (inherits(res, "cerr")) {
    cat("error ", as.character(res), "\n", sep = "")
  } else if (is.matrix(res)) {
    cat("shape ", nrow(res), " ", ncol(res), "\n", sep = "")
    cat("out ", num(as.vector(t(res))), "\n", sep = "")
  } else {
    cat("out ", num(res), "\n", sep = "")
  }
  cat("\n")
}

set.seed(20260830)
mk <- function(N, n) { x <- runif(N) + 0.05; inclusionprobabilities(x, n) }

# ---- random designs, replayed value-for-value ------------------------------
for (case in list(list(N = 12, n = 5), list(N = 40, n = 12), list(N = 7, n = 3))) {
  pik <- mk(case$N, case$n)
  tag <- sprintf("N%d_n%d", case$N, case$n)
  emit(paste0("UPbrewer_", tag),     UPbrewer,     pik, seed = 101)
  emit(paste0("UPtille_", tag),      UPtille,      pik, seed = 102)
  emit(paste0("UPsystematic_", tag), UPsystematic, pik, seed = 103)
  emit(paste0("UPpoisson_", tag),    UPpoisson,    pik, seed = 104)
  emit(paste0("UPmaxentropy_", tag), UPmaxentropy, pik, seed = 105)
}

# ---- deterministic functions, exact ----------------------------------------
for (case in list(list(N = 12, n = 5), list(N = 40, n = 12))) {
  pik <- mk(case$N, case$n)
  tag <- sprintf("N%d_n%d", case$N, case$n)
  emit(paste0("UPtillepi2_", tag),      UPtillepi2,      pik)
  emit(paste0("UPsystematicpi2_", tag), UPsystematicpi2, pik)
  emit(paste0("UPmaxentropypi2_", tag), UPmaxentropypi2, pik)
  emit(paste0("UPMEpiktildefrompik_", tag), UPMEpiktildefrompik, pik)
}

emit("UPMEqfromw_w123_n2", UPMEqfromw, c(1, 2, 3), extra = list(2))
emit("UPMEqfromw_w1234_n3", UPMEqfromw, c(1, 2, 3, 4), extra = list(3))
emit("inclusionprobabilities_basic", inclusionprobabilities, c(1, 2, 3, 4, 5), extra = list(3))
emit("inclusionprobabilities_capped", inclusionprobabilities, c(1, 1, 1, 40), extra = list(3))

# ---- inputs where R raises (ADR-0006 third category) -----------------------
emit("ERR_UPMEqfromw_n1", UPMEqfromw, c(1, 2, 3), extra = list(1))
emit("ERR_UPmaxentropy_subeps", UPmaxentropy, c(1e-7, 5e-7, 2e-7), seed = 1)
emit("ERR_UPmaxentropy_alldet", UPmaxentropy, c(0, 1, 1, 0), seed = 1)
emit("ERR_UPtillepi2_na", UPtillepi2, c(0.2, NA, 0.5, 0.3))
emit("ERR_UPbrewer_alldet", UPbrewer, c(0, 1, 1, 0), seed = 1)
