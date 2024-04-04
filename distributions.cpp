#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <cmath>
#include <iostream>
#include <vector>
#include "distributions.h"



NormalDistribution::NormalDistribution() : mean(0), std(1) {
    this->max_pdf = this->pdf(0);
}

NormalDistribution::NormalDistribution(double mean, double std) : mean(mean), std(std) {
    this->max_pdf = this->pdf(mean);
}

double NormalDistribution::cdf(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

double NormalDistribution::pdf(double x) {
    double value = exp(-0.5 * pow((x - mean) / std, 2)) / (std * sqrt(2 * M_PI));
    return value;
}

double NormalDistribution::confidence(std::vector<double> sample) {
    double sample_mean = 0;
    for (size_t i = 0; i < sample.size(); ++i) {
        sample_mean += sample[i];
    }
    sample_mean /= sample.size();

    double p_value = (sample_mean - mean) / std;
    p_value /= 2;
    double confidence = cdf(p_value);
    return confidence;
}

double NormalDistribution::ppf(double p) {
    // Magick implementation of ppf or inverse cdf from https://gist.github.com/kmpm/1211922/6b7fcd0155b23c3dc71e6f4969f2c48785371292
    //  *  REFERENCE
    //  *
    //  *     Beasley, J. D. and S. G. Springer (1977).
    //  *     Algorithm AS 111: The percentage points of the normal distribution,
    //  *     Applied Statistics, 26, 118-121.
    //  *
    //  *      Wichura, M.J. (1988).
    //  *      Algorithm AS 241: The Percentage Points of the Normal Distribution.
    //  *      Applied Statistics, 37, 477-484.


    double r, val;

    const double q = p - 0.5;

    if (std::abs(q) <= .425) {
        r = .180625 - q * q;
        val =
            q * (((((((r * 2509.0809287301226727 +
                33430.575583588128105) * r + 67265.770927008700853) * r +
                45921.953931549871457) * r + 13731.693765509461125) * r +
                1971.5909503065514427) * r + 133.14166789178437745) * r +
                3.387132872796366608)
            / (((((((r * 5226.495278852854561 +
                28729.085735721942674) * r + 39307.89580009271061) * r +
                21213.794301586595867) * r + 5394.1960214247511077) * r +
                687.1870074920579083) * r + 42.313330701600911252) * r + 1);
    }
    else {
        if (q > 0) {
            r = 1 - p;
        }
        else {
            r = p;
        }

        r = std::sqrt(-std::log(r));

        if (r <= 5) 
        {
            r += -1.6;
            val = (((((((r * 7.7454501427834140764e-4 +
                .0227238449892691845833) * r + .24178072517745061177) *
                r + 1.27045825245236838258) * r +
                3.64784832476320460504) * r + 5.7694972214606914055) *
                r + 4.6303378461565452959) * r +
                1.42343711074968357734)
                / (((((((r *
                    1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                    r + .0151986665636164571966) * r +
                    .14810397642748007459) * r + .68976733498510000455) *
                    r + 1.6763848301838038494) * r +
                    2.05319162663775882187) * r + 1);
        }
        else { /* very close to  0 or 1 */
            r += -5;
            val = (((((((r * 2.01033439929228813265e-7 +
                2.71155556874348757815e-5) * r +
                .0012426609473880784386) * r + .026532189526576123093) *
                r + .29656057182850489123) * r +
                1.7848265399172913358) * r + 5.4637849111641143699) *
                r + 6.6579046435011037772)
                / (((((((r *
                    2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                    r + 1.8463183175100546818e-5) * r +
                    7.868691311456132591e-4) * r + .0148753612908506148525)
                    * r + .13692988092273580531) * r +
                    .59983220655588793769) * r + 1);
        }

        if (q < 0.0) {
            val = -val;
        }
    }

    return mean + std * val;
}

ExponentialDistribution::ExponentialDistribution(double lambda): lambda(lambda) {
    // maximum value will be at x = 0. exp(0) = 1. Thus, max_pdf = lambda
    this->max_pdf = lambda;
}

double ExponentialDistribution::pdf(double x) {
    if (x < 0.0) return 0.0;
    return lambda * exp(-lambda * x);
}

double ExponentialDistribution::ppf(double p) {
    return - log(1 - p) / lambda;
}

double ExponentialDistribution::cdf(double x) {
    return 1 - exp(-lambda * x);
}

double ExponentialDistribution::confidence(std::vector<double> sample) {
    return 0;
}




BetaDistribution::BetaDistribution(double alpha, double beta): alpha(alpha), beta(beta) {
    // it assumes unimodal distribution, so
    // it assumes that alpha >= 1 and beta >= 1. And alpha + beta > 2
    this->max_pdf = this->pdf((alpha - 1) / (alpha + beta - 2));
}

double BetaDistribution::pdf(double x) {
    if (x > 1 || x < 0) {
        return 0;
    }
    double value = std::pow(x, alpha - 1) * std::pow(1 - x, beta - 1);
    value /= std::beta(alpha, beta);
    return value;
}

double BetaDistribution::ppf(double p) {
    // NOT DEFINED
    return 0;
}

double BetaDistribution::cdf(double x) {
    // NOT DEFINED
    return 0;
}

double BetaDistribution::confidence(std::vector<double> sample) {
    // NOT DEFINED
    return 0;
}
