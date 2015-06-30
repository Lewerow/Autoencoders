#include <executor/options.h>
#include <boost/program_options.hpp>

namespace executor
{
	boost::program_options::options_description setup_options()
	{
		boost::program_options::options_description desc("General options");
		desc.add_options()
			("help,h", "Print this message and exit")
			("version,v", "Print version descriptor and exit")
			("output,o", boost::program_options::value<std::string>()->value_name("OUT")->default_value("out.txt"), "Location of output log file")
		    ("config_file", boost::program_options::value<std::string>()->value_name("BASIC_CONFIG")->default_value("config.ini"), "Path to file containing basic configuration options")
            ("train_instances_per_class", boost::program_options::value<unsigned int>()->default_value(500), "Number of training instances per class")
            ("test_instances_per_class", boost::program_options::value<unsigned int>()->default_value(100), "Number of test instances per class")
            ("dropout", boost::program_options::value<double>()->default_value(0.0), "Value of drop-out autoencoder parameter")
            ("train_data", boost::program_options::value<std::string>()->required(), "Location for training data. Comma-separated CSV without headers.")
			("test_data", boost::program_options::value<std::string>()->required(), "Location for test data. Comma-separated CSV without headers.")
			("inputs", boost::program_options::value<unsigned int>()->required(), "Number of input values - first fields in data")
			("outputs", boost::program_options::value<unsigned int>()->required(), "Number of output values - always after inputs. Remaining entires in a row are dropped.")
			("generate", boost::program_options::value<bool>()->default_value(false), "Generate data for teaching. Generated set is a linearly separable set of K classes. Inputs are drawn from uniform real distribution")
			("max_iterations", boost::program_options::value<unsigned int>()->default_value(500), "Maximum allowed number of iterations during each phase of teaching (iterations per layer)")
            ("restart_gradient_after", boost::program_options::value<unsigned int>()->default_value(200), "After how many iterations conjugate gradient algorithm shall be restarted?")
            ("epsilon", boost::program_options::value<double>()->default_value(0.01), "Error rate which - if reached - causes current training phase to short-circuit")
            ("learning_coefficient", boost::program_options::value<double>()->default_value(0.5), "Learning coefficient for backpropagation algorithm")
            ("regularization_factor", boost::program_options::value<double>()->default_value(0.1), "Value of regularization factor, as described in report")
            ("print_at", boost::program_options::value<unsigned int>()->default_value(30), "After how many iterations print current status to stdout? (Each iteration is printed to given log)")
            ("batches_at_supervised", boost::program_options::value<unsigned int>()->default_value(5), "How many batches shall be used for supervised finetuning?")
			("configuration", boost::program_options::value<std::string>()->default_value("15,20")->required(), "Comma-separated counts of neurons in hidden layers. Network always have proper (set in other variables) number of inputs and outputs");

		return desc;
	}
}