#include <memory>
#include <executor/configurator.h>
#include <executor/application.h>

#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>

namespace
{
    std::vector<std::size_t> configure_network(std::string textual, std::size_t ins, std::size_t outs)
    {
        std::vector<std::string> elems;
        boost::algorithm::split(elems, textual, [](char c){return c == ',';});

        std::vector<std::size_t> sizes;
		sizes.push_back(ins);
        for (auto& a: elems)
            sizes.push_back(boost::lexical_cast<std::size_t>(a));

		sizes.push_back(outs);
        return sizes;
    }
}

namespace executor
{
	configurator::configurator(const boost::program_options::variables_map& vars)
	{
		auto app = std::make_unique<executor::whole_application>();
        app->train_instances = vars.at("train_instances_per_class").as<unsigned int>();
        app->test_instances = vars.at("test_instances_per_class").as<unsigned int>();
        app->output_path = vars.at("output").as<std::string>();
        app->dropout = vars.at("dropout").as<double>();
        app->train_data_path = vars.at("train_data").as<std::string>();
		app->test_data_path = vars.at("test_data").as<std::string>();
		app->inputs = vars.at("inputs").as<unsigned int>();
		app->outputs = vars.at("outputs").as<unsigned int>();
		app->generate = vars.at("generate").as<bool>();

        app->max_iterations = vars.at("max_iterations").as<unsigned int>();
        app->restart_gradient_after = vars.at("restart_gradient_after").as<unsigned int>();
        app->acceptable_error_rate = vars.at("epsilon").as<double>();
        app->learning_coefficient = vars.at("learning_coefficient").as<double>();
        app->regularization_factor = vars.at("regularization_factor").as<double>();
        app->print_at = vars.at("print_at").as<unsigned int>();
        app->batches_at_supervised = vars.at("batches_at_supervised").as<unsigned int>();

        app->configuration = configure_network(vars.at("configuration").as<std::string>(), app->inputs, app->outputs);

        this->app = std::move(app);
	}

	configurator::~configurator()
	{}

	executor::application& configurator::get_application() const
	{
		return *app;
	}
}
