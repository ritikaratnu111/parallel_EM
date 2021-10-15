# parallel_EM
Parallel EM algorithm in multi core environment using MPI paradigm

With growing interests in modelling and analysis of data and the avilability of large data, it becomes time consuming to deal with sequential Machine Learning algorithms. Gaussian
Mixture Models (GMM) are popular probabilistic models that assume data points are generated from finite number of Gaussian densities. Expectation Maximization(EM) algorithm is often used in the Maximum Likelihood(ML) estimation of GMM parameters. Given the importance of EM algorithm and the embrassingly parallel steps involved in the iterations of EM algorithm motivates the need for parallelization of EM algorithm. Our results show good speed ups for varying input sizes.
