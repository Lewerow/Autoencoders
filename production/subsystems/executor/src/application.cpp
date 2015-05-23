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

namespace
{
	double activation(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	boost::numeric::ublas::vector<double> absolutes(boost::numeric::ublas::vector<double>&& vec)
	{
		for (auto& v : vec)
			v = std::abs(v);

		return vec;
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

		boost::numeric::ublas::matrix<double> initialize_layer(std::size_t rows, std::size_t cols)
		{
			std::mt19937 twister;
			std::normal_distribution<double> dist(0, 0.1);
			auto matrix = boost::numeric::ublas::matrix<double>(rows, cols);
			for (std::size_t i = 0; i < rows; ++i)
			{
				for (std::size_t j = 0; j < cols; ++j)
					matrix.at_element(i, j) = dist(twister);
			}

			return matrix;
		}

		boost::numeric::ublas::vector<double> feed_forward(const boost::numeric::ublas::vector<double>& input)
		{
			return feed_forward_weights(input, weights);
		}

		boost::numeric::ublas::vector<double> feed_forward_weights(const boost::numeric::ublas::vector<double>& input, const std::vector<boost::numeric::ublas::matrix<double> >& used_weights)
		{
			boost::numeric::ublas::vector<double> output = input;
			for (std::size_t i = 0; i < used_weights.size(); ++i)
				output = step(output, i);

			return output;
		}

		boost::numeric::ublas::vector<double> step(const boost::numeric::ublas::vector<double>& input, const std::size_t layerNumber)
		{
			boost::numeric::ublas::vector<double> activations(weights[layerNumber].size1());
			boost::numeric::ublas::vector<double> signals = boost::numeric::ublas::prod(input, weights[layerNumber]);
			std::transform(signals.begin(), signals.end(), activations.begin(), activation);
			
			return activations;
		}

		void train_layer_wise(const std::vector<boost::numeric::ublas::vector<double> >& inputs, const std::vector<boost::numeric::ublas::vector<double> >& outputs)
		{
			unsupervised_train(inputs);
			weights = supervised_train(inputs, outputs, weights);
		}

		void unsupervised_train(const std::vector<boost::numeric::ublas::vector<double> >& labelless_training_data)
		{
			std::vector<std::vector<boost::numeric::ublas::vector<double> > > layer_inputs;
			layer_inputs.push_back(labelless_training_data);

			for (std::size_t i = 0; i < weights.size() - 1; ++i)
			{
				weights[i] = adapt_layer(i, layer_inputs[i]);

				std::vector<boost::numeric::ublas::vector<double> > layer_results(layer_inputs[i].size());
				std::transform(layer_inputs[i].begin(), layer_inputs[i].end(), layer_results.begin(), std::bind(&Network::step, this, std::placeholders::_1, i));
				layer_inputs.push_back(std::move(layer_results));
			}
		}

		boost::numeric::ublas::matrix<double> adapt_layer(std::size_t layerNumber, const std::vector<boost::numeric::ublas::vector<double> >& inputs)
		{
			assert(inputs.size() > 0);

			std::vector<boost::numeric::ublas::matrix<double> >  fake_weights;
			fake_weights.push_back(weights[layerNumber]);
			fake_weights.push_back(initialize_layer(weights[layerNumber].size2(), inputs[0].size()));
			return supervised_train(inputs, inputs, fake_weights)[0];
		}
		
		std::vector<boost::numeric::ublas::matrix<double> > supervised_train(const std::vector<boost::numeric::ublas::vector<double> >& inputs,
			const std::vector<boost::numeric::ublas::vector<double> >& outputs, 
			const std::vector<boost::numeric::ublas::matrix<double> >& weights)
		{
			assert(inputs.size() == outputs.size());
			std::vector<boost::numeric::ublas::matrix<double> > adapted_weights = weights;

			for (unsigned int iteration = 0; iteration < iterations; ++iteration)
			{
    			adapted_weights = conjugate_gradient_training(inputs, outputs, adapted_weights);					

				if (average_error(inputs, outputs, adapted_weights) < error_epsilon)
					break;
			}

			return adapted_weights;
		}

		std::vector<boost::numeric::ublas::matrix<double> > calculate_gradient(const std::vector<boost::numeric::ublas::vector<double> >& inputs,
			const std::vector<boost::numeric::ublas::vector<double> >& outputs,
			const std::vector<boost::numeric::ublas::matrix<double> >& weights)
		{

		}

		std::vector<boost::numeric::ublas::matrix<double> > conjugate_gradient_training(const std::vector<boost::numeric::ublas::vector<double> >& inputs,
			const std::vector<boost::numeric::ublas::vector<double> >& outputs,
			const std::vector<boost::numeric::ublas::matrix<double> >& weights)
		{
			auto gradient = calculate_gradient(inputs, outputs, weights);


		}

		boost::numeric::ublas::vector<double> test_weights(const std::vector<boost::numeric::ublas::vector<double> >& inputs,
			const std::vector<boost::numeric::ublas::vector<double> >& outputs,
			const std::vector<boost::numeric::ublas::matrix<double> >& weights)
		{
			assert(inputs.size() == outputs.size());
			boost::numeric::ublas::vector<double> errors(weights.back().size1);
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				auto out = feed_forward_weights(inputs[i], weights);
				errors += absolutes(out - outputs[i]);
			}

			return errors / inputs.size();
		}

		double average_error(const std::vector<boost::numeric::ublas::vector<double> >& inputs,
			const std::vector<boost::numeric::ublas::vector<double> >& outputs,
			const std::vector<boost::numeric::ublas::matrix<double> >& weights)
		{
			assert(inputs.size() == outputs.size());
			assert(inputs.size() > 0); 

			auto errors = test_weights(inputs, outputs, weights);
			return std::accumulate(errors.begin(), errors.end(), 0.0, boost::numeric::ublas::norm_2<double>) / outputs[0].size();
		}

		void test(const std::vector<boost::numeric::ublas::vector<double> >& inputs, const std::vector<boost::numeric::ublas::vector<double> >& outputs)
		{
			std::cout << "Average error rate: " << average_error(inputs, outputs, weights) << std::endl;
		}

		static std::unique_ptr<Network> configure(std::vector<std::size_t>&& layer_sizes)
		{
			return std::make_unique<Network>(std::move(layer_sizes));
		}

	private:
		std::vector<boost::numeric::ublas::matrix<double> >  weights;
		const unsigned int iterations = 100;
		const unsigned int max_batch_size = 100;
		const double error_epsilon = 0.1;

	};
}

namespace executor
{
	int whole_application::run()
	{
		auto network = Network::configure({ 4, 15, 3 });
/*		network.train(data.inputs, data.outputs);
		network.test(data.pairs);
	*/
		return 0;
	}
}