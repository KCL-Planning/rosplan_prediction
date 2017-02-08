#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#include <std_srvs/Empty.h>
#include <diagnostic_msgs/KeyValue.h>

#include <rosplan_knowledge_msgs/GetDomainTypeService.h>
#include <rosplan_knowledge_msgs/GetDomainAttributeService.h>
#include <rosplan_knowledge_msgs/GetInstanceService.h>
#include <rosplan_knowledge_msgs/GetAttributeService.h>

#include <rosplan_knowledge_msgs/KnowledgeUpdateService.h>
#include <rosplan_knowledge_msgs/KnowledgeUpdateServiceArray.h>
#include <rosplan_knowledge_msgs/DomainFormula.h>
#include <rosplan_knowledge_msgs/KnowledgeItem.h>

#include <squirrel_prediction_msgs/RecommendRelations.h>

namespace KCL_rosplan {

	/**
	 * This class describes a service node for prediction of the state.
	 * the node uses the ROSPlan knowledge base as the basis of the prediction, using the learning
	 * tool from UIBK:
	 * "Initial State Prediction in Planning" Krivic, S. et al. (AAAI KnowPROS 2017)
	 */
	class RPPrediction
	{

		private:

		std::string data_path;

		float lower_threshold_confidence;
		float upper_threshold_confidence;

		/* domain data */
		std::vector<std::string> types;
		std::vector<std::string> super_types;
		std::vector<rosplan_knowledge_msgs::DomainFormula> columns;

		/* state data */
		std::vector<std::string> objects;
		std::vector<std::string> object_types;
		std::vector<int> object_type_indexes;
		std::map<std::string, std::vector<rosplan_knowledge_msgs::KnowledgeItem> > props;

		/* service clients */
		ros::ServiceClient get_domain_type_client;
		ros::ServiceClient get_domain_attribute_client;
		ros::ServiceClient get_instance_client;
		ros::ServiceClient get_attribute_client;
		ros::ServiceClient knowledge_update_array_client;
		ros::ServiceClient recommender_client;

		/* check methods */
		bool propTrue(int a, int column);
		bool propFalse(int a, int column);
		bool propTrue(int a, int b, int column);
		bool propFalse(int a, int b, int column);
		bool isType(std::string a, std::string b);
		void writeInput();
		void readPrediction();

		public:

		RPPrediction(ros::NodeHandle& nh);

		bool makePrediction(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);

	};

} // close namespace
