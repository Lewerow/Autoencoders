#ifndef APPLICATION_H_JFEWUINDVJKNfuifnewufjiwefjoiwekdsmfkjnedgivnerigovrefefdwedwecvdfgvrthbtynbtynytythbythyt
#define APPLICATION_H_JFEWUINDVJKNfuifnewufjiwefjoiwekdsmfkjnedgivnerigovrefefdwedwecvdfgvrthbtynbtynytythbythyt

#include <string>
#include <vector>

namespace executor
{
	class application
	{
	public:
		virtual ~application(){}
		virtual int run() = 0;
	};

	class whole_application : public application
	{
	public:
		int run();

        double dropout;
        unsigned int train_instances;
        unsigned int test_instances;
        std::string output_path;
        std::string train_data_path;
        std::string test_data_path;
        std::vector<std::size_t> configuration;

        unsigned int print_at;
        unsigned int max_iterations;
        unsigned int restart_gradient_after;
        double acceptable_error_rate;
        double learning_coefficient;
        double regularization_factor;
        unsigned int batches_at_supervised;

		unsigned int inputs;
		unsigned int outputs;
		bool generate;
	};
}

#endif