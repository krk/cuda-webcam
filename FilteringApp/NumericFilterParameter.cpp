#include "NumericFilterParameter.h"

char NumericFilterParameter::getMinValue() { return minValue; }
	
char NumericFilterParameter::getMaxValue() { return maxValue; }
	
void NumericFilterParameter::setValue(char value) { this->value = value; }
	
char NumericFilterParameter::getValue() { return value; }
	
QString NumericFilterParameter::getCaption() { return caption; }