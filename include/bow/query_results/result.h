#ifndef RESULT_H_
#define RESULT_H_

#include <ostream>

namespace place_recognition
{
// Single result of a query
class Result
{
public: 
  	Result() {}
	Result(unsigned int _id, double _score) : 
		id(_id), score(_score) {}

	// operator
	bool operator<(Result& result) { return this->score < result.score; }
	bool operator>(Result& result) { return this->score > result.score; }
	bool operator==(unsigned int id) { return this->id == id; }
	bool operator<(double score) { return this->score < score; }
	bool operator>(double score) { return this->score > score; }

	static inline bool gt(Result& a,Result& b) { return a.score > b.score; }
	static inline bool ge(Result& a,Result& b) { return a.score > b.score; }
	static inline bool geq(Result& a,Result& b) { return a.score >= b.score; }
	static inline bool geqv(Result& a,double s) { return a.score >= s; }
	static inline bool lt_id(Result& a,Result& b) { return a.id < b.id; }

  	friend std::ostream & operator<<(std::ostream& os,const Result& ret)
  	{
		os << "<EntryId: " << ret.id << ", Score: " << ret.score << ">";
		return os;
  	}
  
	unsigned int id;    // Entry id
  	double score;       // Score obtained
	int nWords;         // words in common
	double bhatScore;
	double chiScore;
	
	// only done by ChiSq and BCThresholding 
  	double sumCommonVi;
  	double sumCommonWi;
  	double expectedChiScore;
  
private:
};
}

#endif	// RESULT_H_