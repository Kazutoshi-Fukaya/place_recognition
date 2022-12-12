#ifndef IF_PAIR_H_
#define IF_PAIR_H

// Inverted file declaration  
// Item of IFRow
class IFPair
{
public:
	IFPair() {}
	IFPair(unsigned int eid,double ww) : 
		entry_id(eid), word_weight(ww) {}

	inline bool operator==(unsigned eid) const { return entry_id == eid; }

	unsigned int entry_id;	// Entry id
    double word_weight;		// Word weight in this entry
private:
};

#endif	// IF_PAIR_H