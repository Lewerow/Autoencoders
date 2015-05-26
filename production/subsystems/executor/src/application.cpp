#include <executor/application.h>

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <limits>
#include <functional>
#include <numeric>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

using namespace boost::numeric;

namespace
{
	double activation(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	double backprop_activation(double x, double workpoint)
	{
		double f = activation(workpoint);
		return f * (1 - f) * x;
	}

	ublas::vector<double> absolutes(ublas::vector<double>&& vec)
	{
		for (auto& v : vec)
			v = std::abs(v);

		return vec;
	}
	ublas::vector<double> step_forward(const ublas::vector<double>& input, const ublas::matrix<double>& used_weights)
	{
		ublas::vector<double> activations(used_weights.size2());
		ublas::vector<double> signals = ublas::prod(input, used_weights);
		std::transform(signals.begin(), signals.end(), activations.begin(), activation);

		return activations;
	}

	ublas::vector<double> step_back(const ublas::vector<double>& forward_activations, const ublas::matrix<double>& used_weights,
		const ublas::vector<double>& backpropagation_vector)
	{
		ublas::vector<double> backprop_result(used_weights.size1());
		ublas::vector<double> signals = ublas::prod(used_weights, backpropagation_vector);

		for (std::size_t i = 0; i < backprop_result.size(); ++i)
			backprop_result[i] = backprop_activation(signals[i], forward_activations[i]);

		return backprop_result;
	}

	class Network
	{
	public:
		Network(std::vector<std::size_t>&& layer_sizes) 
		{
			assert(layer_sizes.size() > 1);
			for (std::size_t i = 1; i < layer_sizes.size(); ++i)
				weights.push_back(initialize_layer(layer_sizes[i], layer_sizes[i - 1]));
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

		ublas::vector<double> feed_forward(const ublas::vector<double>& input)
		{
			return feed_forward_weights(input, weights);
		}

		ublas::vector<double> feed_forward_weights(const ublas::vector<double>& input, const std::vector<ublas::matrix<double> >& used_weights)
		{
			ublas::vector<double> intermediate = input;
			for (std::size_t i = 0; i < used_weights.size(); ++i)
				intermediate = step_forward(intermediate, used_weights[i]);

			return intermediate;
		}

		void train_layer_wise(const std::vector<ublas::vector<double> >& inputs, const std::vector<ublas::vector<double> >& outputs)
		{
			std::cout << "Starting unsupervised training" << std::endl;
			unsupervised_train(inputs);
			std::cout << "Starting supervised training" << std::endl;
			weights = supervised_train(inputs, outputs, weights);
		}

		void unsupervised_train(const std::vector<ublas::vector<double> >& labelless_training_data)
		{
			std::vector<std::vector<ublas::vector<double> > > layer_inputs;
			layer_inputs.push_back(labelless_training_data);

			for (std::size_t i = 0; i < weights.size() - 1; ++i)
			{
				weights[i] = adapt_layer(i, layer_inputs[i]);

				std::vector<ublas::vector<double> > layer_results(layer_inputs[i].size());
				std::transform(layer_inputs[i].begin(), layer_inputs[i].end(), layer_results.begin(), std::bind(step_forward, std::placeholders::_1, weights[i]));
				layer_inputs.push_back(std::move(layer_results));
			}
		}

		ublas::matrix<double> adapt_layer(std::size_t layerNumber, const std::vector<ublas::vector<double> >& inputs)
		{
			assert(inputs.size() > 0);

			std::vector<ublas::matrix<double> >  fake_weights;
			fake_weights.push_back(weights[layerNumber]);
			fake_weights.push_back(initialize_layer(inputs[0].size(), weights[layerNumber].size2()));
			return supervised_train(inputs, inputs, fake_weights)[0];
		}
		
		std::vector<ublas::matrix<double> > supervised_train(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs, 
			const std::vector<ublas::matrix<double> >& weights)
		{
			assert(inputs.size() == outputs.size());
			std::vector<ublas::matrix<double> > adapted_weights = weights;

			for (unsigned int iteration = 0; iteration < iterations; ++iteration)
			{
				adapted_weights = conjugate_gradient_training(inputs, outputs, adapted_weights);			
				std::cout << "Average error rate: " << average_error(inputs, outputs, adapted_weights) << std::endl;
				if (average_error(inputs, outputs, adapted_weights) < error_epsilon)
					break;
			}

			return adapted_weights;
		}

		std::vector<ublas::matrix<double> > calculate_gradient(const ublas::vector<double>& input,
			const ublas::vector<double>& output,
			const std::vector<ublas::matrix<double> >& weights)
		{
			auto difference = feed_forward_weights(input, weights) - output;

			std::vector<ublas::vector<double> > activations = { input };
			for (std::size_t i = 0; i < weights.size(); ++i)
				activations.push_back(step_forward(activations[i], weights[i]));

			std::vector<ublas::vector<double> > backward_activations(activations.size());
			backward_activations.back() = activations.back() - output;
			for (std::size_t i = weights.size(); i > 0; --i)
				backward_activations[i - 1] = step_back(activations[i - 1], weights[i - 1], backward_activations[i]);

			std::vector<ublas::matrix<double> > whole_gradient;
			for (std::size_t i = 1; i < weights.size(); ++i)
			{
				whole_gradient.emplace_back(ublas::matrix<double>(weights[i - 1].size1(), weights[i - 1].size2()));
				for (std::size_t j = 0; j < activations[i - 1].size(); ++j)
				{
					for (std::size_t k = 0; k < backward_activations[i].size(); ++k)
					{
						whole_gradient.back().at_element(j, k) = activations[i - 1][j] * backward_activations[i][k];
					}
				}
			}

			return whole_gradient;
		}

		std::vector<ublas::matrix<double> > conjugate_gradient_training(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			std::vector<ublas::matrix<double> > weights)
		{

			std::vector<ublas::matrix<double> > previous_gradient(weights.size());
			for (std::size_t i = 0; i < weights.size(); ++i)
				previous_gradient[i] = ublas::matrix<double>(weights[i].size1(), weights[i].size2(), 0);

			std::size_t gradient_iterations = 0;
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				auto gradient = calculate_gradient(inputs[i], outputs[i], weights);
				std::vector<ublas::matrix<double> > direction(gradient.size());
				for (std::size_t j = 0; j < gradient.size(); ++j)
				{
					double beta = 0.3;
					/*if (i > 0)
					    beta = ublas::norm_2(ublas::prod(ublas::prod(ublas::trans(gradient[j]), (gradient[j] - previous_gradient[j])), ublas::scalar_vector<double>(gradient[j].size2()))) /
							ublas::norm_2(ublas::prod(ublas::prod(ublas::trans(previous_gradient[j]), previous_gradient[j]), ublas::scalar_vector<double>(previous_gradient[j].size2())));
							*/
					if (++gradient_iterations >= restart_gradient)
					{
						beta = 0;
						gradient_iterations = 0;
						std::cout << "Restarting gradient calculation" << std::endl;
					}
					direction[j] = gradient[j];/// +beta * beta * previous_gradient[j];
					previous_gradient[j] = std::move(gradient[j]);

					weights[j] += direction[j] * learning_coefficient;
				}
			}

			return weights;
		}

		ublas::vector<double> test_weights(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			const std::vector<ublas::matrix<double> >& weights)
		{
			assert(inputs.size() == outputs.size());
			ublas::vector<double> errors(weights.back().size2(), 0);
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				auto out = feed_forward_weights(inputs[i], weights);
				errors += absolutes(out - outputs[i]);
			}

			return errors / inputs.size();
		}

		double average_error(const std::vector<ublas::vector<double> >& inputs,
			const std::vector<ublas::vector<double> >& outputs,
			const std::vector<ublas::matrix<double> >& weights)
		{
			assert(inputs.size() == outputs.size());
			assert(inputs.size() > 0); 

			auto errors = test_weights(inputs, outputs, weights);
			return ublas::sum(errors) / errors.size();
		}

		void test(const std::vector<ublas::vector<double> >& inputs, const std::vector<ublas::vector<double> >& outputs)
		{
			std::cout << "Average error rate: " << average_error(inputs, outputs, weights) << std::endl;
		}

		static std::unique_ptr<Network> configure(std::vector<std::size_t>&& layer_sizes)
		{
			return std::make_unique<Network>(std::move(layer_sizes));
		}

	private:
		std::vector<ublas::matrix<double> >  weights;
		const unsigned int iterations = 1000;
		const unsigned int restart_gradient = 5000;
		const unsigned int max_batch_size = 100;
		const double error_epsilon = 0.1;
		const double learning_coefficient = 0.7;
	};



	std::pair<std::vector<ublas::vector<double> >, std::vector<ublas::vector<double> > > generate_linear()
	{
		std::random_device rand;
		std::mt19937 mt(rand());
		std::uniform_real_distribution<double> reals(0.1, 0.9);
		std::vector<ublas::vector<double> > ins;
		std::vector<ublas::vector<double> > outs;
		for (std::size_t i = 0; i < 500; ++i)
		{
			double a = reals(mt);
			double b = reals(mt);
			ublas::vector<double> in = ublas::vector<double>(2);
			in[0] = a;
			in[1] = b;
			ublas::vector<double> out = ublas::vector<double>(2);

			if (a > b)
			{
				out[0] = 0.9;
				out[1] = 0.1;
			}
			else
			{
				out[1] = 0.9;
				out[0] = 0.1;
			}
			ins.push_back(in);
			outs.push_back(out);
		}

		return std::make_pair(ins, outs);
	}
}

namespace executor
{
	int whole_application::run()
	{

		auto network = Network::configure({ 2, 15, 2 });

		auto data = generate_linear();

		network->test(data.first, data.second);
		network->train_layer_wise(data.first, data.second);
		network->test(data.first, data.second);
	
		return 0;
	}
}