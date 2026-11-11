#include "crown_mlp_file.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sci::crown::io {

namespace {

constexpr char kMagic[8] = {'C', 'R', 'O', 'W', 'N', 'M', 'L', 'P'};
constexpr uint32_t kVersion = 1;

void read_exact(std::istream &is, void *dst, std::size_t n) {
  is.read(static_cast<char *>(dst), static_cast<std::streamsize>(n));
  if (!is || static_cast<std::size_t>(is.gcount()) != n) {
    throw std::runtime_error("Unexpected EOF while reading CROWN MLP file");
  }
}

uint32_t read_u32_le(std::istream &is) {
  uint32_t v = 0;
  read_exact(is, &v, sizeof(v));
  return v;
}

double read_f64_le(std::istream &is) {
  double v = 0.0;
  read_exact(is, &v, sizeof(v));
  return v;
}

} // namespace

std::vector<sci::crown::plain::DenseLayerPublic> load_crown_mlp_file(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(
        "CROWN MLP file not found (wrong path or cwd): " + path +
        "\nNote: `mlp=` must be the packed .crownmlp from Python export_crown_mlp_bin(), "
        "not an .onnx file. Use an absolute path if you run cryptflow2_crown from build-ot/.");
  }
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Cannot open CROWN MLP file for reading: " + path);
  }

  char magic[8];
  read_exact(is, magic, sizeof(magic));
  for (int i = 0; i < 8; ++i) {
    if (magic[i] != kMagic[i]) {
      throw std::runtime_error(
          "Invalid CROWN MLP magic (expected ASCII CROWNMLP at byte 0). "
          "This file is not export_crown_mlp_bin output — did you pass an .onnx path by mistake?");
    }
  }

  const uint32_t version = read_u32_le(is);
  if (version != kVersion) {
    std::ostringstream oss;
    oss << "Unsupported CROWN MLP version " << version << " (expected " << kVersion << ")";
    throw std::runtime_error(oss.str());
  }

  const uint32_t num_layers = read_u32_le(is);
  const uint32_t input_dim = read_u32_le(is);
  if (num_layers < 1) {
    throw std::runtime_error("CROWN MLP num_layers must be >= 1");
  }

  std::vector<sci::crown::plain::DenseLayerPublic> net;
  net.reserve(num_layers);

  uint32_t expect_next_in = UINT32_MAX;
  for (uint32_t li = 0; li < num_layers; ++li) {
    const uint32_t in_d = read_u32_le(is);
    const uint32_t out_d = read_u32_le(is);

    if (li == 0) {
      if (in_d != input_dim) {
        std::ostringstream oss;
        oss << "CROWN MLP input_dim field " << input_dim << " != first layer in_dim " << in_d;
        throw std::runtime_error(oss.str());
      }
    } else {
      if (in_d != expect_next_in) {
        std::ostringstream oss;
        oss << "CROWN MLP layer " << li << ": expected in_dim " << expect_next_in << ", got " << in_d;
        throw std::runtime_error(oss.str());
      }
    }

    sci::crown::plain::DenseLayerPublic layer;
    layer.in_dim = static_cast<int>(in_d);
    layer.out_dim = static_cast<int>(out_d);
    layer.W.assign(static_cast<size_t>(out_d), sci::crown::plain::Vec(static_cast<size_t>(in_d)));
    for (uint32_t r = 0; r < out_d; ++r) {
      for (uint32_t c = 0; c < in_d; ++c) {
        layer.W[(size_t)r][(size_t)c] = read_f64_le(is);
      }
    }
    layer.b.resize(static_cast<size_t>(out_d));
    for (uint32_t j = 0; j < out_d; ++j) {
      layer.b[(size_t)j] = read_f64_le(is);
    }

    expect_next_in = out_d;
    net.push_back(std::move(layer));
  }

  char trailing = '\0';
  if (is.read(&trailing, 1)) {
    throw std::runtime_error("Trailing data after CROWN MLP payload");
  }

  return net;
}

} // namespace sci::crown::io
