#ifndef _TABLE_HPP
#define _TABLE_HPP

#include <cstddef>
#include <vector>

namespace table {

// print an horizontal line
void hsep();

// print a centered title
void htitle(const char *title);

// print a row
void row(std::vector<const char *> columns, std::vector<size_t> widths);

} // namespace table

#endif // _TABLE_HPP
