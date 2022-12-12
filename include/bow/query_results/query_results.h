#ifndef QURY_RESULTS_H_
#define QURY_RESULTS_H_

#include <vector>

#include "bow/query_results/result.h"

namespace place_recognition
{
// Multiple results from a query
class QueryResults : public std::vector<Result>
{
public:
	void scale_scores(double factor)
	{
		for(QueryResults::iterator qit = begin(); qit != end(); qit++) qit->score *= factor;
	}

	void save_M(std::string &file_name);

	friend std::ostream& operator<<(std::ostream& os,const QueryResults& ret)
	{
		if(ret.size() == 1) os << "1 result:" << std::endl;
		else os << ret.size() << " results:" << std::endl;
		
		QueryResults::const_iterator rit;
		for(rit = ret.begin(); rit != ret.end(); rit++){
			os << *rit;
			if(rit + 1 != ret.end()) os << std::endl;
  		}
  		return os;
	}
	
};
}

#endif	// QURY_RESULTS_H_