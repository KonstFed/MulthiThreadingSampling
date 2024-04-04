#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <omp.h>

#include "distributions.h"

#define N_THREADS 8

using namespace std;

class RandomGenerator {
    private:
        ProbabilityFunction& prob;

    public:
        RandomGenerator(ProbabilityFunction& prob): prob(prob) {};

        vector<double> sample_ppf(unsigned short n_threads, unsigned int n_samples) {
            vector<double> out(n_samples);
            #pragma omp parallel default(none) shared(n_samples, out) num_threads(n_threads)
            {
                unsigned int myseed = omp_get_thread_num();
                double p;
                unsigned int tmp;
                #pragma omp for private(p, tmp)
                for (int i = 0; i < n_samples; ++i) {
                    tmp = rand_r(&myseed);
                    p = static_cast<double>(tmp) / RAND_MAX;
                    out[i] = prob.ppf(p);
                }
            }
            return out;
        }

        vector<double> sample_accept_reject(unsigned short n_threads, unsigned int n_samples, double lb, double ub)
        {
            vector<double> out(n_samples);
            #pragma omp parallel num_threads(n_threads)
            {
                unsigned int myseed = omp_get_thread_num();

                unsigned int tmp;
                double x, pdf_prob, uniform_prob;
                #pragma omp for private(tmp, x, pdf_prob, uniform_prob)
                for (int i = 0; i < out.size(); ++i)
                {
                    while (true) {
                        // sample from lower and upper bound and save it to 'x'
                        tmp = rand_r(&myseed);
                        x = static_cast<double>(tmp) / RAND_MAX;
                        x = lb + x * (ub - lb);
                        // get pdf value of x
                        pdf_prob = prob.pdf(x);
                        
                        // sample from 0 to maximum of pdf
                        tmp = rand_r(&myseed);
                        uniform_prob = static_cast<double>(tmp) / RAND_MAX;
                        uniform_prob *= prob.max_pdf;

                        if (uniform_prob < pdf_prob) {
                            // accept
                            out[i] = x;
                            break;
                        }
                    }
                }

            }
            return out;
        }
};

int main() {
    ofstream experiment_time_log, sample_result;
    experiment_time_log.open("result.csv");
    experiment_time_log << "n_threads,n_samples,distribution,method,time\n";
    sample_result.open("samples.txt");
    NormalDistribution norm;
    ExponentialDistribution exp_distr(1.5);
    BetaDistribution beta(3, 4);
    RandomGenerator norm_gen(norm), exp_gen(exp_distr), beta_gen(beta);
    const unsigned int MAX_N_SAMPLES = 10000000;
    double start, end;
    double ppf_time, sample_reject_time;
    vector<double> result;

    const unsigned short THREADS2TEST[] = {1, 2, 4, 8};
    double time_ppf[sizeof(THREADS2TEST)];
    double time_acc_rej[sizeof(THREADS2TEST)];

    for (int n_samples = 1000000; n_samples <= MAX_N_SAMPLES; n_samples += 1000000)
    {
        cout << "Number of samples is: " << n_samples << "\n"; 
        cout << "Normal distribution:\n";
        for (int i = 0; i < sizeof(THREADS2TEST) / sizeof(*THREADS2TEST); i++) {
            cout << "\tNumber of threads --> " << THREADS2TEST[i] << "\n";
            double conf;

            start = omp_get_wtime();
            result = norm_gen.sample_ppf(THREADS2TEST[i], n_samples);
            end = omp_get_wtime();

            if (n_samples == MAX_N_SAMPLES && i == 3) {
                // collect sample
                sample_result << "Normal distribution, inverse cdf\n";
                for (auto _to_write: result) {
                    sample_result << _to_write << ",";
                }
                sample_result << "\n";
            }

            conf = norm.confidence(result);
            ppf_time = end - start;
            experiment_time_log << THREADS2TEST[i] << "," << n_samples << "," << "normal,inverse cdf," << ppf_time << "\n";
            cout << "\t\tInverse CDF sampling: " << ppf_time << " s.\n";


            start = omp_get_wtime();
            result = norm_gen.sample_accept_reject(THREADS2TEST[i], n_samples, norm.ppf(0.01), norm.ppf(0.99));
            end = omp_get_wtime();

            if (n_samples == MAX_N_SAMPLES && i == 3) {
                // collect sample
                sample_result << "Normal distribution, accept reject\n";
                for (auto _to_write: result) {
                    sample_result << _to_write << ",";
                }
                sample_result << "\n";
            }

            conf = norm.confidence(result);
            sample_reject_time = end - start;
            experiment_time_log << THREADS2TEST[i] << "," << n_samples << "," << "normal,accept reject," << sample_reject_time << "\n";
            cout << "\t\tAccept/Reject sampling: " << sample_reject_time << " s.\n";
        }

        cout << "Exponential distribution:\n";
        for (int i = 0; i < sizeof(THREADS2TEST) / sizeof(*THREADS2TEST); i++) {
            cout << "\tNumber of threads --> " << THREADS2TEST[i] << "\n";

            start = omp_get_wtime();
            result = exp_gen.sample_ppf(THREADS2TEST[i], n_samples);
            end = omp_get_wtime();

            if (n_samples == MAX_N_SAMPLES && i == 3) {
                // collect sample
                sample_result << "Exponential distribution, inverse cdf\n";
                for (auto _to_write: result) {
                    sample_result << _to_write << ",";
                }
                sample_result << "\n";
            }

            ppf_time = end - start;
            experiment_time_log << THREADS2TEST[i] << "," << n_samples << "," << "exponential,inverse cdf," << ppf_time << "\n";
            cout << "\t\tInverse CDF sampling: " << ppf_time << " s.\n";

            start = omp_get_wtime();
            result = exp_gen.sample_accept_reject(THREADS2TEST[i], n_samples, 0, exp_distr.ppf(0.98));
            end = omp_get_wtime();

            if (n_samples == MAX_N_SAMPLES && i == 3) {
                // collect sample
                sample_result << "Exponential distribution, accept reject\n";
                for (auto _to_write: result) {
                    sample_result << _to_write << ",";
                }
                sample_result << "\n";
            }

            sample_reject_time = end - start;
            experiment_time_log << THREADS2TEST[i] << "," << n_samples << "," << "exponential,accept reject," << sample_reject_time << "\n";

            cout << "\t\tAccept/Reject sampling: " << sample_reject_time << " s.\n";
        }
        cout << "Beta distribution\n";

        for (int i = 0; i < sizeof(THREADS2TEST) / sizeof(*THREADS2TEST); i++) {
            cout << "\tNumber of threads --> " << THREADS2TEST[i] << "\n";
            start = omp_get_wtime();
            result = exp_gen.sample_accept_reject(THREADS2TEST[i], n_samples, 0.0, 1.0);
            end = omp_get_wtime();
            if (n_samples == MAX_N_SAMPLES && i == 3) {
                // collect sample
                sample_result << "Beta distribution, accept reject\n";
                for (auto _to_write: result) {
                    sample_result << _to_write << ",";
                }
                sample_result << "\n";
            }
            sample_reject_time = end - start;
            experiment_time_log << THREADS2TEST[i] << "," << n_samples << "," << "beta,accept reject," << sample_reject_time << "\n";
            cout << "\t\tAccept/Reject sampling: " << sample_reject_time << " s.\n";

        }
    }
    experiment_time_log.close();
    sample_result.close();
    // std::cout << beta.max_pdf << "\n";
    // for (double a = 0.0; a < 1.0; a += 0.05) {
    //     std::cout << beta.pdf(a) << ",";
    // }
    // std::cout<<"\n";
}