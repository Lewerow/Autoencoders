#include <executor/application.h>

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <limits>
#include <functional>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace
{
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
			boost::numeric::ublas::vector<double> output = input;
			for (std::size_t i = 0; i < weights.size(); ++i)
				output = step(output, i);

			return output;
		}

		boost::numeric::ublas::vector<double> step(const boost::numeric::ublas::vector<double>& input, const std::size_t layerNumber)
		{
			boost::numeric::ublas::vector<double> activations(weights[layerNumber].size1);
			boost::numeric::ublas::vector<double> signals = boost::numeric::ublas::prod(input, weights[layerNumber]);
			std::transform(signals.begin(), signals.end(), activations.begin(), activation);
			
			return activations;
		}

		double activation(double x)
		{
			return 1.0 / (1.0 + std::exp(-x));
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
				std::transform(layer_inputs[i].begin(), layer_inputs[i].end(), layer_results.begin(), std::bind(step, std::placeholders::_1, i));
				layer_inputs.push_back(std::move(layer_results));
			}
		}

		boost::numeric::ublas::matrix<double> adapt_layer(std::size_t layerNumber, const std::vector<boost::numeric::ublas::vector<double> >& inputs)
		{
			assert(inputs.size() > 0);

			std::vector<boost::numeric::ublas::matrix<double> >  fake_weights;
			fake_weights.push_back(weights[layerNumber]);
			fake_weights.push_back(initialize_layer(weights[layerNumber].size2, inputs[0].size()));
			return supervised_train(inputs, inputs, fake_weights)[0];
		}
		
		std::vector<boost::numeric::ublas::matrix<double> > supervised_train(const std::vector<boost::numeric::ublas::vector<double> >& inputs,
			const std::vector<boost::numeric::ublas::vector<double> >& outputs, 
			const std::vector<boost::numeric::ublas::matrix<double> >& weights)
		{

		}

		void test(const std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > >& data)
		{
			double error = 0;
			for (const auto& d : data)
			{
				auto out = feed_forward(d.first);
				error += boost::numeric::ublas::norm_2(out - d.second);
			}

			std::cout << "Total error rate: " << error << std::endl;
		}

		static std::unique_ptr<Network> configure(std::vector<std::size_t>&& layer_sizes)
		{
			return std::make_unique<Network>(std::move(layer_sizes));
		}

	private:
		std::vector<boost::numeric::ublas::matrix<double> >  weights;
		const int iterations = 100;

	};
}

namespace executor
{
	int whole_application::run()
	{
		auto network = Network::configure({ 4, 15, 3 });
		network.train(data.inputs, data.outputs);
		network.test(data.pairs);
	}
}