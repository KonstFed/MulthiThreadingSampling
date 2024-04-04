#include <vector>

class ProbabilityFunction {
    public:
        double max_pdf;
        virtual double pdf(double x) = 0;
        virtual double ppf(double p) = 0;
        virtual double confidence(std::vector<double> sample) = 0;
        virtual double cdf(double x) = 0;

};

class NormalDistribution : public ProbabilityFunction {
    private:
        double mean, std;
    public:
        NormalDistribution();
        NormalDistribution(double mean, double std);
        double pdf(double x);
        double ppf(double p);
        double confidence(std::vector<double> sample);
        double cdf(double x);
};

class ExponentialDistribution : public ProbabilityFunction {
    private:
        double lambda;
    public:
        ExponentialDistribution(double lambda);
        double pdf(double x);
        double ppf(double p);
        double confidence(std::vector<double> sample);
        double cdf(double x);
};

class BetaDistribution : public ProbabilityFunction {
    private:
        double alpha, beta;
    public:
        BetaDistribution(double alpha, double beta);
        double pdf(double x);
        double ppf(double p);
        double confidence(std::vector<double> sample);
        double cdf(double x);
};