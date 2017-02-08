#include "rosplan_interface_prediction/RPPrediction.h"

/* implementation of rosplan_interface_prediction::RPPrediction */
namespace KCL_rosplan {

	/**
	 * Constructor
	 */
	RPPrediction::RPPrediction(ros::NodeHandle& nh) {
		// KB clients
		get_domain_type_client = nh.serviceClient<rosplan_knowledge_msgs::GetDomainTypeService>("/kcl_rosplan/get_domain_types");
		get_domain_attribute_client = nh.serviceClient<rosplan_knowledge_msgs::GetDomainAttributeService>("/kcl_rosplan/get_domain_predicates");
		get_instance_client = nh.serviceClient<rosplan_knowledge_msgs::GetInstanceService>("/kcl_rosplan/get_current_instances");
		get_attribute_client = nh.serviceClient<rosplan_knowledge_msgs::GetAttributeService>("/kcl_rosplan/get_current_knowledge");
		knowledge_update_array_client = nh.serviceClient<rosplan_knowledge_msgs::KnowledgeUpdateServiceArray>("/kcl_rosplan/update_knowledge_base_array");
		recommender_client = nh.serviceClient<squirrel_prediction_msgs::RecommendRelations>("/squirrel_relations_prediction");

		upper_threshold_confidence = 0.60f;
		lower_threshold_confidence = 0.60f;

		nh.getParam("data_path", data_path);
	}

	/*----------*/
	/* SERVICES */
	/*----------*/

	/**
	 * ROS Service callback method
	 */
	bool RPPrediction::makePrediction(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res) {

		ROS_INFO("KCL: (prediction) Starting state prediction");

		// reset
		columns.clear();
		types.clear();
		super_types.clear();

		objects.clear();
		object_types.clear();
		object_type_indexes.clear();
		props.clear();

		// write input
		writeInput();

		// call learner
		squirrel_prediction_msgs::RecommendRelations srv;
		srv.request.data_path = data_path;
		srv.request.input_file = "example.csv";
		srv.request.output_file = "predicted_missing_known_full.csv";
		srv.request.number_of_columns = 2 * types.size() + columns.size();
		if(!recommender_client.call(srv)) {
			ROS_ERROR("KCL: (prediction) Prediction failed, recommender failed or unreachable");
			return false;
		}

		// read output
		readPrediction();

		ROS_INFO("KCL: (prediction) Prediction complete");

		return true;
	}

	/*-------------*/
	/* WRITING CSV */
	/*-------------*/

	/**
	 * Writes the initial state to csv file for the learner.
	 */
	void RPPrediction::writeInput() {

		ROS_INFO("KCL: (prediction) Fetching data from KB");

		// fetch domain predicates
		rosplan_knowledge_msgs::GetDomainAttributeService predSrv;
		get_domain_attribute_client.call(predSrv);

		std::vector<rosplan_knowledge_msgs::DomainFormula>::iterator pit = predSrv.response.items.begin();
		for(; pit!=predSrv.response.items.end(); pit++) {

			if(pit->typed_parameters.size()!=2 && pit->typed_parameters.size()!=1) continue;

			columns.push_back(*pit);

			// fetch propositions of predicate
			rosplan_knowledge_msgs::GetAttributeService propSrv;
			propSrv.request.predicate_name = pit->name;
			get_attribute_client.call(propSrv);

			props.insert(std::pair<std::string, std::vector<rosplan_knowledge_msgs::KnowledgeItem> >(pit->name, propSrv.response.attributes));
		}

		// fetch domain types
		rosplan_knowledge_msgs::GetDomainTypeService typeSrv;
		get_domain_type_client.call(typeSrv);
		types = typeSrv.response.types;
		super_types = typeSrv.response.super_types;

		std::vector<std::string>::iterator tit = typeSrv.response.types.begin();
		for(; tit!=typeSrv.response.types.end(); tit++) {

			std::string type = *tit;

			// fetch instances of type
			rosplan_knowledge_msgs::GetInstanceService instanceSrv;
			instanceSrv.request.type_name = type;
			get_instance_client.call(instanceSrv);

			std::vector<std::string>::iterator iit = instanceSrv.response.instances.begin();
			for(; iit!=instanceSrv.response.instances.end(); iit++) {
				objects.push_back(*iit);
				object_types.push_back(type);
		
				// fetch category of the first object (i)
				int type_index = 0;
				for(int t=0; t<types.size(); t++) {
					if(types[t] == type)
						type_index = t;
				}
				object_type_indexes.push_back(type_index);
			}
		}

		ROS_INFO("KCL: (prediction) Writing to input file");

		// prepare file
		std::ofstream inputFile;
		std::stringstream ss;
		ss << data_path << "/example.csv";
		inputFile.open(ss.str().c_str());

		// write column headers to file
		inputFile << "'object1','object2'";

		for(int i=0; i<types.size(); i++)
			inputFile << ",ob1-is-" << types[i];
		for(int i=0; i<types.size(); i++)
			inputFile << ",ob2-is-" << types[i];
		for(int i=0; i<columns.size(); i++)
			inputFile << "," << columns[i].name;

		inputFile << std::endl;

		// for pairs of objects
		for(int i=0; i<objects.size(); i++) {
		for(int j=0; j<objects.size(); j++) {

			// write names
			inputFile << "'" << objects[i] << "','" << objects[j] << "'";

			// write types
			for(int t=0; t<types.size(); t++) {
				if(isType(object_types[i],types[t])) inputFile << ",2";
				else inputFile << ",1";
			}

			for(int t=0; t<types.size(); t++) {
				if(isType(object_types[j],types[t])) inputFile << ",2";
				else inputFile << ",1";
			}

			// write known truths
			for(int k=0; k<columns.size(); k++) {


				if(columns[k].typed_parameters.size()==1) {

					// unary predicate
					if((!isType(object_types[i], columns[k].typed_parameters[0].value))) {
						// type check
						inputFile << ",";
					} else if(propFalse(i,k)) {
						inputFile << ",1";
					} else if(propTrue(i,k)) {
						inputFile << ",2";
					} else {
						inputFile << ",";
					}
				
				} else {

					// binary predicate
					if((!isType(object_types[i], columns[k].typed_parameters[0].value)) ||
						(!isType(object_types[j], columns[k].typed_parameters[1].value))) {
						// type check
						inputFile << ",";
					} else if(propFalse(i,j,k)) {
						inputFile << ",1";
					} else if(propTrue(i,j,k)) {
						inputFile << ",2";
					} else {
						inputFile << ",";
					}
				}
			}

			inputFile << std::endl;
		}};

		// close file writer
		inputFile.close();
	}

	/**
	 * Returns true iff the proposition exists in the KB, and is not negative UNARY
	 */
	bool RPPrediction::propTrue(int a, int column) {

		// type check
		if(!isType(object_types[a], columns[column].typed_parameters[0].value)) return false;

		std::map<std::string, std::vector<rosplan_knowledge_msgs::KnowledgeItem> >::iterator mit = props.find(columns[column].name);
		if(mit == props.end()) return false;

		// value check
		std::vector<rosplan_knowledge_msgs::KnowledgeItem>::iterator kit = props[columns[column].name].begin();
		for(; kit != props[columns[column].name].end(); kit++) {
			if(kit->values[0].value==objects[a]) return !kit->is_negative;
		}

		return false;
	}

	/**
	 * Returns true iff the proposition exists in the KB, and is not negative BINARY
	 */
	bool RPPrediction::propTrue(int a, int b, int column) {

		// type check
		if(!isType(object_types[a], columns[column].typed_parameters[0].value)) return false;
		if(!isType(object_types[b], columns[column].typed_parameters[1].value)) return false;

		std::map<std::string, std::vector<rosplan_knowledge_msgs::KnowledgeItem> >::iterator mit = props.find(columns[column].name);
		if(mit == props.end()) return false;

		// value check
		std::vector<rosplan_knowledge_msgs::KnowledgeItem>::iterator kit = props[columns[column].name].begin();
		for(; kit != props[columns[column].name].end(); kit++) {
			if(kit->values[0].value==objects[a] && kit->values[1].value==objects[b]) return !kit->is_negative;
		}

		return false;
	}

	/**
	 * Returns true iff the proposition exists in the KB, and is_negative BINARY
	 */
	bool RPPrediction::propFalse(int a, int b, int column) {

		// type check
		if(!isType(object_types[a], columns[column].typed_parameters[0].value)) return false;
		if(!isType(object_types[b], columns[column].typed_parameters[1].value)) return false;

		std::map<std::string, std::vector<rosplan_knowledge_msgs::KnowledgeItem> >::iterator mit = props.find(columns[column].name);
		if(mit == props.end()) return false;

		// value check
		std::vector<rosplan_knowledge_msgs::KnowledgeItem>::iterator kit = props[columns[column].name].begin();
		for(; kit != props[columns[column].name].end(); kit++) {
			if(kit->values[0].value==objects[a] && kit->values[1].value==objects[b]) return kit->is_negative;
		}

		return false;
	}

	/**
	 * Returns true iff the proposition exists in the KB, and is_negative UNARY
	 */
	bool RPPrediction::propFalse(int a, int column) {

		// type check
		if(!isType(object_types[a], columns[column].typed_parameters[0].value)) return false;

		std::map<std::string, std::vector<rosplan_knowledge_msgs::KnowledgeItem> >::iterator mit = props.find(columns[column].name);
		if(mit == props.end()) return false;

		// value check
		std::vector<rosplan_knowledge_msgs::KnowledgeItem>::iterator kit = props[columns[column].name].begin();
		for(; kit != props[columns[column].name].end(); kit++) {
			if(kit->values[0].value==objects[a]) return kit->is_negative;
		}

		return false;
	}

	/**
	 * Returns true iff type "a" is, or is a subtype of type "b"
	 */
	bool RPPrediction::isType(std::string a, std::string b) {

		if(a==b) return true;

		for(int i=0; i<types.size(); i++) {
			if(types[i] == a && super_types[i] != a)
				return isType(super_types[i], b);
		}

		return false;
	}

	/*--------------------*/
	/* READING PREDICTION */
	/*--------------------*/

	/**
	 * Reads the prediction file and writes new propositions to the KB
	 */
	void RPPrediction::readPrediction() {

		ROS_INFO("KCL: (prediction) Reading data from csv file");

		rosplan_knowledge_msgs::KnowledgeUpdateServiceArray srv;
		srv.request.update_type = rosplan_knowledge_msgs::KnowledgeUpdateServiceArray::Request::ADD_KNOWLEDGE;

		std::stringstream ss;
		ss << data_path << "/predicted_missing_known_full.csv";
		std::ifstream preFile(ss.str().c_str());

		std::string line;
		int curr = 0;
		int next = 0;

		if (preFile.is_open()) {

			// ignore header
			getline(preFile,line);

			rosplan_knowledge_msgs::KnowledgeItem msg;
			msg.knowledge_type = rosplan_knowledge_msgs::KnowledgeItem::FACT;
			std::string obj1, obj2;
			int a, b;

			// 'obj1','obj2',[features],[confidences]
			while (getline(preFile,line)) {

				// objects
				curr = 0;
				next=line.find(",",curr);
				obj1 = line.substr(curr+1, next-curr-2);
				curr=next+1;

				next=line.find(",",curr);
				obj2 = line.substr(curr+1, next-curr-2);
				curr=next+1;

				// skip the category columns
				next=line.find(",",curr);
				curr=next+1;

				next=line.find(",",curr);
				curr=next+1;

				// find object indexes
				int a = 0;
				while(a<objects.size() && objects[a]!=obj1) a++;
				int b = 0;
				while(b<objects.size() && objects[b]!=obj2) b++;

				if(a >= objects.size() || b >= objects.size()) {
					ROS_ERROR("KCL: (prediction) unrecognised objects.");
					return;
				}

				// features
				for(int i=0; i<columns.size(); i++) {

					next=line.find(",",curr);

					/*/ only add predictions on affordances
					if(columns[i].name.substr(0,3)!="can") {
						curr=next+1;
						continue;
					}*/

					// proposition is prediced to be false
					if("1" == line.substr(curr,next-curr)
						&& (isType(object_types[a], columns[i].typed_parameters[0].value))
						&& ((columns[i].typed_parameters.size()==1 && !propFalse(a,i)) || (columns[i].typed_parameters.size()==2 && !propFalse(a,b,i))
							)) {
						msg.values.clear();
						msg.attribute_name = columns[i].name;
						diagnostic_msgs::KeyValue pair;
						pair.key = columns[i].typed_parameters[0].key;
						pair.value = obj1;
						msg.values.push_back(pair);
						if(columns[i].typed_parameters.size()==2) {
							pair.key = columns[i].typed_parameters[1].key;
							pair.value = obj2;
							msg.values.push_back(pair);
						}
						srv.request.knowledge.push_back(msg);
					}

					// proposition is prediced to be true
					if("2" == line.substr(curr,next-curr)
						&& (isType(object_types[a], columns[i].typed_parameters[0].value))
						&& (isType(object_types[b], columns[i].typed_parameters[1].value))
						&& ((columns[i].typed_parameters.size()==1 && !propTrue(a,i)) || (columns[i].typed_parameters.size()==2 && !propTrue(a,b,i))
							)) {
						msg.values.clear();
						msg.attribute_name = columns[i].name;
						diagnostic_msgs::KeyValue pair;
						pair.key = columns[i].typed_parameters[0].key;
						pair.value = obj1;
						msg.values.push_back(pair);
						if(columns[i].typed_parameters.size()==2) {
							pair.key = columns[i].typed_parameters[1].key;
							pair.value = obj2;
							msg.values.push_back(pair);
						}
						srv.request.knowledge.push_back(msg);
					}
					curr=next+1;
				}
				// confidence
				for(int i=0; i<columns.size(); i++) { }
			}
			preFile.close();
		} else {
			ROS_ERROR("KCL: (prediction) Could not open prediction file: predicted_missing_known_full.csv");
			return;
		}

		// add to KB
		if (!knowledge_update_array_client.call(srv)) {
			ROS_ERROR("KCL: (prediction) Could not add new predictions to knowledge base.");
		}
	}
}

int main(int argc, char **argv) {

	// init
	ros::init(argc, argv, "rosplan_prediction_service");
	ros::NodeHandle nh("~");

	KCL_rosplan::RPPrediction rpp(nh);

	// start the prediction service
	ros::ServiceServer service1 = nh.advertiseService("/kcl_rosplan/state_prediction_server", &KCL_rosplan::RPPrediction::makePrediction, &rpp);

	ROS_INFO("KCL: (prediction) Ready to receive");
	ros::spin();

	return 0;
}
