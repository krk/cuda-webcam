#pragma once
#ifndef NUMERICFILTERPARAMETER_H
#define NUMERICFILTERPARAMETER_H

#include <QObject>
#include <QString>

class NumericFilterParameter
{
protected:
	char minValue;
	char maxValue;
	QString caption;
	char value;
public:

	NumericFilterParameter() {  }

	NumericFilterParameter(const NumericFilterParameter &other)
	{
		minValue = other.minValue; 
		maxValue = other.maxValue; 
		caption = other.caption;
		value = other.value; 
	}

	NumericFilterParameter(char minValue, char maxValue, QString caption)
	{
		this->minValue = minValue;
		this->maxValue = maxValue;
		this->caption = caption;
	}

	char getMinValue();
	char getMaxValue();
	void setValue(char value);
	char getValue();
	QString getCaption();
};

#endif NUMERICFILTERPARAMETER_H