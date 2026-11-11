#ifndef SCI_CROWN_MLP_FILE_H__
#define SCI_CROWN_MLP_FILE_H__

#include "crown_plaintext.h"

#include <string>
#include <vector>

namespace sci::crown::io {

/// Load a packed MLP written by ``utils/format_convert.export_crown_mlp_bin`` (version 1, little-endian).
std::vector<sci::crown::plain::DenseLayerPublic> load_crown_mlp_file(const std::string &path);

} // namespace sci::crown::io

#endif
