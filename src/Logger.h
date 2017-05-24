/*
 * Logger.h
 *
 */

#include <iostream>
#include <fstream>

using namespace std;

#ifndef LOGGER_H_
#define LOGGER_H_

class Logger {
public:
  Logger(ofstream& _log_out, ostream& _std_out):log_out(_log_out), std_out(_std_out){}

  template<typename T>
  const Logger& operator<<(const T& v) const {
	  log_out << v;
	  log_out.flush();
	  std_out << v;
	  return *this;
  }

protected:
  ofstream& log_out;
  ostream& std_out;
};

#endif /* LOGGER_H_ */
