#ifndef OBJECT_NAME_LIST_H_
#define OBJECT_NAME_LIST_H_

#include <iostream>
#include <vector>

namespace place_recognition
{
class ObjectNameList : public std::vector<std::string>
{
public:
	ObjectNameList() {}

	void add_name(std::string name)
	{
		for(size_t i = 0; i < this->size(); i++){
			if(this->at(i) == name) return;
		}
		this->emplace_back(name);
	}

private:
};
} // namespace place_recognition

#endif	// OBJECT_NAME_LIST_H_