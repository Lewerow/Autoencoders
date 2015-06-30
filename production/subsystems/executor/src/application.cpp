#include <executor/application.h>

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <limits>
#include <functional>
#include <numeric>
#include <map>
#include <fstream>
#include <thread>

#include <boost/tokenizer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/lexical_cast.hpp>

using namespace boost::numeric;

namespace
{
	typedef std::size_t ClassId;
	struct ClassResults
	{
		unsigned int truePositives;
		unsigned int falsePositives;
		unsigned int falseNegatives;
	};

	struct Metrics
	{
		double recall;
		double precision;
		double fmeasure;
	};

	double activation(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	double backprop_activation(double x, double workpoint)
	{
		double f = activation(workpoint);
		return f * (1 - f) * x;
	}

	double sum_diagonal(const ublas::matrix<double>& w)
	{
		assert(w.size1() == w.size2());

		double acc = 0;
		for (std::size_t i = 0; i < w.size1(); ++i)
			acc += const_cast<ublas::matrix<double>& >(w).at_element(i, i);

		return acc;
	}

	void regularize(ublas::matrix<double>& weights, double regularization_factor)
	{
		double sum = sum_diagonal(ublas::prod(ublas::trans(weights), weights));

		if (sum > regularization_factor * weights.size1() * weights.size2())
			weights *= ((regularization_factor * weights.size1() * weights.size2()) / sum);
	}

	ublas::vector<double> absolutes(ublas::vector<double>&& vec)
	{
		for (auto& v : vec)
			v = std::abs(v);

		return vec;
	}
	ublas::vector<double> step_forward(ublas::vector<double> input, const ublas::matrix<double>& used_weights, double dropout)
	{
		std::random_device dev;
		std::uniform_real_distribution<double> reals(0, 0.5);
		for (auto& i : input)
		{
			if (reals(dev) < dropout)
				i = 0;
		}

		ublas::vector<double> activations(used_weights.size2());
		ublas::vector<double> signals = ublas::prod(input, used_weights);
		std::transform(signals.begin(), signals.end(), activations.begin(), activation);

		activations[activations.size() - 1] = 1;
		return activations;
	}

	ublas::vector<double> step_back(const ublas::vector<double>& forward_activations, const ublas::matrix<double>& used_weights,
		const ublas::vector<double>& backpropagation_vector, double dropout)
	{
		ublas::vector<double> backprop_result(used_weights.size1());
		ublas::vector<double> signals = ublas::prod(used_weights, backpropagation_vector);

		std::random_device dev;
		std::uniform_real_distribution<double> reals(0, 1);
		for (std::size_t i = 0; i < backprop_result.size(); ++i)
		{
			if (reals(dev) < dropout)
				backprop_result[i] = 0;
			else
				backprop_result[i] = backprop_activation(signals[i], forward_activations[i]);
		}

		backprop_result[backprop_result.size() - 1] = 1;

		return backprop_result;
	}

	class Network
	{
	public:
		Network(std::vector<std::size_t>&& layer_sizes, double drop_out, std::ostream& out_, unsigned int print_at_, unsigned int iterations_,
            unsigned int restart_gradient_, unsigned int batches_, double epsilon_, double coeff_, double regularize_) : dropout(drop_out), out(out_)
		{
			assert(layer_sizes.size() > 1);
			for (std::size_t i = 1; i < layer_sizes.size(); ++i)
				weights.push_back(initialize_layer(layer_sizes[i] + 1, layer_sizes[i - 1] + 1));

            print_at = print_at_;
            iterations = iterations_;
            restart_gradient = restart_gradient_;
            batches_at_supervised = batches_;
            error_epsilon = epsilon_;
            learning_coefficient = coeff_;
            regularization_factor = regularize_;
		}

		Network& operator=(Network&) = delete;

		ublas::matrix<double> initialize_layer(std::size_t rows, std::size_t cols)
		{
			std::mt19937 twister;
			std::normal_distribution<double> dist(0, 1);
			auto matrix = ublas::matrix<double>(cols, rows);
			for (std::size_t i = 0; i < rows; ++i)
			{
				for (std::size_t j = 0; j < cols; ++j)
					matrix.at_element(j, i) = dist(twister);
			}

			return matrix;
		}

		ublas::vector<double> feed_forward(const ublas::vector<double>& input, double dropout)
		{
			return feed_forward_weights(input, weights, dropout);
		}

		ublas::vector<double> feed_forward_weights(const ublas::vector<double>& input, const std::vector<ublas::matrix<double> >& used_weights, double dropout)
		{
			ublas::vector<double> intermediate = step_forward(input, used_weights[0], 0);
			for (std::size_t i = 1; i < used_weights.size(); ++i)
				intermediate = step_forward(intermediate, used_weights[i], dropout);

			return intermediate;
		}

		void train_layer_wise(const std::vector<ublas::vector<double> >& inputs, const std::vector<ublas::vector<double> >& outputs)
		{
            out << "Starting unsupervised training" << std::endl;
            std::cout << "Starting unsupervised training" << std::endl;
            unsupervised_train(inputs);

			out << "Starting supervised training" << std::endl;
            std::cout << "Starting supervised training" << std::endl;
			for (std::size_t i = 0; i < batches_at_supervised; ++i)
			{
                std::cout << "Starting supervised training, batch: " << i << std::endl;
                std::vector<ublas::vector<double> > ins(inputs.begin() + i * inputs.size() / batches_at_supervised, inputs.begin() + (i + 1) * inputs.size() / batches_at_supervised);
				std::vector<ublas::vector<double> > outs(outputs.begin() + i * outputs.size() / batches_at_supervised, outputs.begin() + (i + 1) * outputs.size() / batches_at_supervised);

				weights = supervised_train(ins, outs, weights, 0.0);

			}
		}

		void unsupervised_train(const std::vector<ublas::vector<double> >& labelless_training_data)
		{
            out << "Unsupervised: Layer 1" << std::endl;
            std::cout << "Unsupervised: Layer 1" << std::endl;
            std::vector<std::vector<ublas::vector<double> > > layer_inputs;
			layer_inputs.push_back(labelless_training_data);

			weights[0] = adapt_layer(0, layer_inputs[0]) * (1 - dropout);
			std::vector<ublas::vector<double> > layer_results(layer_inputs[0].size());
			std::transform(layer_inputs[0].begin(), layer_inputs[0].end(), layer_results.begin(), std::bind(step_forward, std::placeholders::_1, weights[0], 0.0));
			layer_inputs.push_back(std::move(layer_results));

			for (std::size_t i = 1; i < weights.size() - 1; ++i)
			{
				out << "Layer " << i + 1 << std::endl;
                std::cout << "Unsupervised: Layer " << i + 1 << std::endl;
                weights[i] = adapt_layer(i, layer_inputs[i]) * (1 - dropout);

				std::vector<ublas::vector<double> > layer_results(layer_inputs[i].size());
				std::transform(layer_inputs[i].begin(), layer_inputs[i].end(), layer_results.begin(), std::bind(step_forward, std::placeholders::_1, weights[i], 0.0));
				layer_inputs.push_back(std::move(layer_results));
			}
		}

		ublas::matrix<double> adapt_layer(std::size_t layerNumber, const std::vector<ublas::vector<double> >& inputs)
		{
			assert(inputs.size() > 0);

			std::vector<ublas::matrix<double> >  fake_weights;
			fake_weights.push_back(weights[layerNumber]);
			fake_weights.push_back(initialize_layer(inputs[0].size(), weights[layerNumber].size2()));
			return supervised_train(inputs, inputs, fake_weights, dropout)[0];
		}
		
		std::vector<ublas::matrix<double> > supervised_train(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs, 
			const std::vector<ublas::matrix<double> >& weights, double dropout)
		{
			assert(inputs.size() == outputs.size());
			std::vector<ublas::matrix<double> > adapted_weights = weights;

			adapted_weights = conjugate_gradient_training(inputs, outputs, adapted_weights, iterations, dropout);			
			out << "Average error rate: " << average_error(inputs, outputs, adapted_weights, dropout) << std::endl;

			return adapted_weights;
		}

		std::vector<ublas::matrix<double> > calculate_gradient(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			const std::vector<ublas::matrix<double> >& weights,
			double dropout)
		{
			std::vector<ublas::matrix<double> > whole_gradient;
			for (std::size_t i = 0; i < weights.size(); ++i)
				whole_gradient.emplace_back(ublas::matrix<double>(weights[i].size1(), weights[i].size2(), 0));

			for (std::size_t in = 0; in < inputs.size(); ++in)
			{
				const ublas::vector<double>& input = inputs[in];
				const ublas::vector<double>& output = outputs[in];

				std::vector<ublas::vector<double> > activations = { input };
				for (std::size_t i = 0; i < weights.size(); ++i)
					activations.push_back(step_forward(activations[i], weights[i], dropout));

				std::vector<ublas::vector<double> > backward_activations(activations.size());
				backward_activations.back() = activations.back() - output;
				for (std::size_t i = weights.size(); i > 0; --i)
					backward_activations[i - 1] = step_back(activations[i - 1], weights[i - 1], backward_activations[i], dropout);

				for (std::size_t i = 0; i < weights.size(); ++i)
				{
					for (std::size_t j = 0; j < activations[i].size(); ++j)
					{
						for (std::size_t k = 0; k < backward_activations[i + 1].size(); ++k)
						{
							whole_gradient[i].at_element(j, k) += activations[i][j] * backward_activations[i + 1][k];
						}
					}
				}
			}

			for (auto& g : whole_gradient)
				g /= inputs.size();

			return whole_gradient;
		}

		std::vector<ublas::matrix<double> > conjugate_gradient_training(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			std::vector<ublas::matrix<double> > weights, unsigned int iterations, double dropout)
		{

			std::vector<ublas::matrix<double> > previous_gradient(weights.size());
			for (std::size_t i = 0; i < weights.size(); ++i)
				previous_gradient[i] = ublas::matrix<double>(weights[i].size1(), weights[i].size2(), 0);

			double previous_error = std::numeric_limits<double>::max();
			double current_error = 0;
			double current_learning_coefficient = learning_coefficient;
			for (unsigned int iteration = 0; iteration < iterations; ++iteration)
			{
				current_error = average_error(inputs, outputs, weights, dropout);
				if (current_error < error_epsilon)
					break;

                if (iteration % print_at == 0)
                    std::cout << "At iteration: " << iteration << " current error: " << current_error << std::endl;

				out << "ER:" << iteration << ";" << current_error << std::endl;				
				out << "LC:" << iteration << ";" << current_learning_coefficient << std::endl;

				auto gradient = calculate_gradient(inputs, outputs, weights, dropout);
				std::vector<ublas::matrix<double> > direction(gradient.size());
				for (std::size_t i = 0; i < weights.size(); ++i)
					direction[i] = ublas::matrix<double>(weights[i].size1(), weights[i].size2(), 0);

				for (std::size_t j = 0; j < gradient.size(); ++j)
				{
					double beta = 0;
					if (iteration % restart_gradient == 0)
					{
						beta = 0;
						out << "Restarting gradient calculation" << std::endl;
					}
					else
					{
						beta = sum_diagonal(ublas::prod(ublas::trans(gradient[j]), gradient[j] - previous_gradient[j])) /
							sum_diagonal(ublas::prod(ublas::trans(previous_gradient[j]), previous_gradient[j]));
					}

					direction[j] = gradient[j] + beta * direction[j];
					previous_gradient[j] = std::move(gradient[j]);

					regularize(direction[j], regularization_factor);
					weights[j] -= (direction[j] * current_learning_coefficient);
					regularize(direction[j], regularization_factor * 10);
				}

				previous_error = current_error;
			}
			return weights;
		}

		ublas::vector<double> test_weights(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			const std::vector<ublas::matrix<double> >& weights,
			double dropout)
		{
			assert(inputs.size() == outputs.size());
			ublas::vector<double> errors(weights.back().size2(), 0);
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				auto out = feed_forward_weights(inputs[i], weights, dropout);
				errors += absolutes(out - outputs[i]);
			}

			return errors / inputs.size();
		}

		double average_error(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			const std::vector<ublas::matrix<double> >& weights,
			double dropout)
		{
			assert(inputs.size() == outputs.size());
			assert(inputs.size() > 0); 

			auto errors = test_weights(inputs, outputs, weights, dropout);
			return ublas::sum(errors) / errors.size();
		}

		std::vector<ClassId> classify(const std::vector<ublas::vector<double> >& inputs, const std::vector<ublas::matrix<double> >& weights)
		{
			std::vector<ClassId> results;
			results.reserve(inputs.size());
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				auto out = feed_forward_weights(inputs[i], weights, 0.0);
				results.push_back(std::distance(out.begin(), std::max_element(out.begin(), out.end() - 1)));
			}

			return results;
		}

		void test(const std::vector<ublas::vector<double> >& inputs, const std::vector<ublas::vector<double> >& outputs)
		{
			out << "Average error rate: " << average_error(inputs, outputs, weights, 0.0) << std::endl;
		}

		void report_classification_from_outputs(const std::vector<ublas::vector<double> >& inputs, const std::vector<ublas::vector<double> >& outputs, const std::vector<ublas::matrix<double> >& weights)
		{
			std::vector<ClassId> classes;
			classes.resize(outputs.size());
			std::transform(outputs.begin(), outputs.end(), classes.begin(), [](const ublas::vector<double>& outs) {return std::distance(outs.begin(), std::max_element(outs.begin(), outs.end() - 1)); });

			report_classification(inputs, classes, weights);
		}

		void report_classification(const std::vector<ublas::vector<double> >& inputs, const std::vector<ClassId>& corrects, const std::vector<ublas::matrix<double> >& weights)
		{
			std::map<ClassId, ClassResults> results;
			auto classifications = classify(inputs, weights);
			assert(classifications.size() == corrects.size());

			for (std::size_t i = 0; i < corrects.size(); ++i)
			{
				if (corrects[i] != classifications[i])
				{
					++results[classifications[i]].falsePositives;
					++results[corrects[i]].falseNegatives;
				}
				else
					++results[corrects[i]].truePositives;
			}

			std::vector<Metrics> metrics;
			metrics.resize(results.size());
			std::transform(results.begin(), results.end(), metrics.begin(), [](const std::pair<ClassId, ClassResults>& res){
				Metrics m;
				m.precision = (res.second.truePositives + res.second.falsePositives) > 0 ? (res.second.truePositives) / double(res.second.truePositives + res.second.falsePositives) : 0;
				m.recall = (res.second.truePositives + res.second.falseNegatives) > 0 ? (res.second.truePositives) / double(res.second.truePositives + res.second.falseNegatives) : 0;
				m.fmeasure = (m.precision + m.recall) > 0 ? 2 * (m.precision * m.recall) / (m.precision + m.recall) : 0;
				return m;
			});
			
			if (metrics.size() > 0)
			{
                out << "Average precision: " << std::accumulate(metrics.begin(), metrics.end(), 0.0, [](double acc, Metrics m){return acc + m.precision; }) / metrics.size() << std::endl;
                out << "Average recall: " << std::accumulate(metrics.begin(), metrics.end(), 0.0, [](double acc, Metrics m){return acc + m.recall; }) / metrics.size() << std::endl;
                out << "Average F-measure: " << std::accumulate(metrics.begin(), metrics.end(), 0.0, [](double acc, Metrics m){return acc + m.fmeasure; }) / metrics.size() << std::endl;
                std::cout << "Average precision: " << std::accumulate(metrics.begin(), metrics.end(), 0.0, [](double acc, Metrics m){return acc + m.precision; }) / metrics.size() << std::endl;
                std::cout << "Average recall: " << std::accumulate(metrics.begin(), metrics.end(), 0.0, [](double acc, Metrics m){return acc + m.recall; }) / metrics.size() << std::endl;
                std::cout << "Average F-measure: " << std::accumulate(metrics.begin(), metrics.end(), 0.0, [](double acc, Metrics m){return acc + m.fmeasure; }) / metrics.size() << std::endl;
            }
			else
			{
                out << "No metrics!" << std::endl;
                std::cout << "No metrics!" << std::endl;
            }
		}

		std::vector<ublas::matrix<double> >  weights;
        unsigned int print_at;
		unsigned int iterations;
		unsigned int restart_gradient;
        unsigned int batches_at_supervised;
		double error_epsilon;
		double learning_coefficient;
		double regularization_factor;
		double dropout;
		std::ostream& out;
	};

	std::pair<std::vector<ublas::vector<double> >, std::vector<ublas::vector<double> > > load_data(std::size_t inputs, std::size_t outputs, const std::string& path)
	{
		std::fstream file(path.c_str(), std::ios_base::in);
		if (!file.is_open())
			throw std::runtime_error("Cannot open input file: " + path);

		std::vector<ublas::vector<double> > ins;
		std::vector<ublas::vector<double> > outs;

		while (!file.eof())
		{
			char c;
			double d;

			ublas::vector<double> in(inputs + 1);
			for (std::size_t i = 0; i < inputs; ++i)
			{
				file >> d >> c;
				if (c != ',')
					throw std::runtime_error("Invalid input file");

				in[i] = d;
			}
			in[inputs] = 1;
			ins.push_back(in);

			ublas::vector<double> out(outputs + 1);
			for (std::size_t i = 0; i < outputs; ++i)
			{
				file >> d >> c;
				if (c != ',' && (c != '\n' && c != '\r' && i < outputs - 1))
					throw std::runtime_error("Invalid input file");

				out[i] = d;
			}
			out[outputs] = 1;
			outs.push_back(out);
				
			if (std::isdigit(c))
				file.putback(c);
			while (!std::isdigit(file.peek()) && file.good())
				file.get(c);
		}

		return std::make_pair(ins, outs);
	}

	std::pair<std::vector<ublas::vector<double> >, std::vector<ublas::vector<double> > > generate_linear(std::size_t inputs, std::size_t outputs, std::size_t amount)
	{
		const double min = 0.1;
		const double max = 0.9;
		std::random_device rand;
		std::mt19937 mt(rand());
		std::uniform_real_distribution<double> reals(min, max);
		std::vector<ublas::vector<double> > ins;
		std::vector<ublas::vector<double> > outs;
		for (std::size_t i = 0; i < amount; ++i)
		{
			double a = reals(mt);
			double b = reals(mt);
			ublas::vector<double> in = ublas::vector<double>(inputs);

			double avg = 0;
			for (std::size_t j = 0; j < inputs; ++j) 
			{
				double z = reals(mt);
				avg += z;
				in[j] = z;
			}

			avg /= inputs;

			ublas::vector<double> out = ublas::vector<double>(outputs);
			double diff = (max - min) / outputs;
			for (int j = 0; j < outputs; ++j) 
			{
				double desc = avg - (min + diff * j);
				if (desc > 0 && desc < diff)
					out[j] = max;
				else
					out[j] = min;
			}
				
			ins.push_back(in);
			outs.push_back(out);
		}

		return std::make_pair(ins, outs);
	}
}

void dump(std::pair<std::vector<ublas::vector<double> >, std::vector<ublas::vector<double> > >& data, std::string path)
{
    auto& in = data.first;
    auto& out = data.second;
    assert(in.size() == out.size());

    std::ofstream outf(path, std::ios_base::trunc);
	if (!outf.is_open())
		throw std::runtime_error("Cannot open file for writing: " + path);

    for (std::size_t i = 0; i < in.size(); ++i)
    {
		auto j = in[i].begin();
		outf << (*j);
        for (++j; j < in[i].end(); ++j)
            outf << "," << (*j);

		for (auto k = out[i].begin(); k < out[i].end(); ++k)
            outf << "," << (*k);

        outf << std::endl;
    }
    
    std::cout << "Dumped data to " << path <<std::endl;
}

namespace executor
{
	int whole_application::run()
	{
		if (generate)
		{
			auto train_data = generate_linear(inputs, outputs, train_instances);
			dump(train_data, train_data_path);

			auto test_data = generate_linear(inputs, outputs, test_instances);
			dump(test_data, test_data_path);
		}

		auto train_data = load_data(inputs, outputs, train_data_path);
		auto test_data = load_data(inputs, outputs, test_data_path);

		std::ofstream stream(output_path);
		auto network = std::make_unique<Network>(std::move(configuration), dropout, stream, print_at, max_iterations, restart_gradient_after, batches_at_supervised, acceptable_error_rate,
			learning_coefficient, regularization_factor);

		network->test(test_data.first, test_data.second);
		network->train_layer_wise(train_data.first, train_data.second);
		network->test(test_data.first, test_data.second);
		network->report_classification_from_outputs(test_data.first, test_data.second, network->weights);
		

		return 0;
	}
}