#include "table.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

#define WIDTH 80
#define SIDE '|'
#define BLANK ' '

namespace table {

using namespace std;

void hsep() { cout << string(WIDTH, '-') << endl; }

void htitle(const char *title) {
  size_t len = strlen(title);
  uint32_t w = ((WIDTH - len) / 2) - 2 + (len % 2 ? 0 : 1);
  cout << SIDE << string(w, BLANK) << title << string(w, BLANK) << SIDE << endl;
}

void row(std::vector<const char *> columns, std::vector<size_t> widths) {
  assert(widths.size() == 0 || columns.size() == widths.size());
  size_t spacers = columns.size() + 1;
  size_t n = columns.size();
  if (widths.size() == 0) {
    uint32_t col_width = (WIDTH - spacers) / n;
    size_t nn = n;
    while (nn--)
      widths.push_back(col_width);

    // leave any extra space to the last column
    if (col_width * n + spacers < WIDTH)
      widths[n - 1] += WIDTH - (col_width * n + spacers);
  }
  assert(std::accumulate(widths.begin(), widths.end(), 0) ==
         (int)(WIDTH - spacers));

  cout << SIDE;
  for (size_t i = 0; i < n; ++i) {
    auto text = columns[i];
    auto width = widths[i];

    cout << setw(width) << text << SIDE;
  }
  cout << endl;
}

} // namespace table
