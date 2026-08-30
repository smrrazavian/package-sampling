# R with the sampling 2.10 sources, for regenerating the golden corpus.
# Only pure-R functions are needed, so the package is sourced rather than
# installed -- MASS and lpSolve are required by varest/gencalib and
# UPminimalsupport, none of which the corpus covers.
FROM alpine:latest
RUN apk add --no-cache R curl tar
WORKDIR /w
RUN curl -sSL -o sampling.tar.gz \
      https://cran.r-project.org/src/contrib/Archive/sampling/sampling_2.10.tar.gz \
 && tar xzf sampling.tar.gz && rm sampling.tar.gz
