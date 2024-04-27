#include "verify.hpp"

#include <iostream>

using namespace std;

int main() {
  auto report = verify();
  auto total = report->failed.size() + report->passed.size();

  cout << report->passed.size() << "/" << total << " tests passed" << endl;

  delete report;
}
