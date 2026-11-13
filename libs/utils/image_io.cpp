
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include "utils/image_io.h"

#include <fstream>
#include <iostream>
#include <memory>

#include <jpeglib.h>
#include <png.h>

#include "common/log.h"
#include "common/tga.h"

#define CHECK(condition, return_code, ...) \
  if (!(condition)) {                      \
    TraceError(__VA_ARGS__);               \
    return (return_code);                  \
  }

namespace cuvslam {

namespace {

// A set of functions for saving/loading images (png/jpg),
// borrowed from Isaac sdk/gems/image/io.cpp.
// Tested with grayscale image representations, support for RGB(A) is present but not tested.

// Struct to hold PNG file information including file pointer and metadata
struct PngImpl {
  std::unique_ptr<std::istream> fp;
  png_structp png = nullptr;
  png_infop info = nullptr;

  ~PngImpl() { png_destroy_read_struct(&png, &info, nullptr); }
};

// Struct to hold JPEG file information including file pointer and metadata
struct JpegImpl {
  std::FILE* fp;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  ~JpegImpl() {
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(fp);
  }
};

// Custom function to read PNG from a std::istream with libpng.
void ReadPng(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::istream* in = reinterpret_cast<std::istream*>(png_get_io_ptr(png_ptr));
  assert(in);
  if (!in->read(reinterpret_cast<char*>(data), length)) {
    png_error(png_ptr, "Failed to read from input stream.");
  }
}

// Helper function to load metadata of a PNG from a std::istream.
bool LoadPngInfo(const std::string& filename, PngImpl& impl) {
  // Open file
  const std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;
  impl.fp = std::make_unique<std::ifstream>(filename, mode);
  if (!impl.fp->good()) {
    return false;
  }

  impl.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!impl.png) {
    TraceError("png_create_read_struct() failed");
    return false;
  }

  impl.info = png_create_info_struct(impl.png);
  if (!impl.info) {
    TraceError("png_create_info_struct() failed");
    return false;
  }

  if (setjmp(png_jmpbuf(impl.png)) != 0) {
    TraceError("Error during png header loading");
    return false;
  }

  png_set_read_fn(impl.png, impl.fp.get(), ReadPng);
  png_read_info(impl.png, impl.info);
  return true;
}

// Helper function to read an open PNG file's data contents into a buffer
template <typename K, int N, typename SizeT>
bool LoadPngImpl(PngImpl& impl, K* data, SizeT rows, SizeT cols) {
  if (setjmp(png_jmpbuf(impl.png)) != 0) {
    TraceError("Error during png body loading");
    return false;
  }

  const int width = png_get_image_width(impl.png, impl.info);
  const int height = png_get_image_height(impl.png, impl.info);
  const png_byte color_type = png_get_color_type(impl.png, impl.info);
  const png_byte bit_depth = png_get_bit_depth(impl.png, impl.info);

  if (height != rows || width != cols) {
    TraceError(
        "Buffer dimensions must match the image dimensions."
        "Expected %d x %d, received buffer with %d x %d",
        height, width, static_cast<int>(rows), static_cast<int>(cols));
    return false;
  }

  const bool out_color = N > 2;
  const bool in_color = color_type & PNG_COLOR_MASK_COLOR;
  const bool out_alpha = N == 2 || N == 4;
  const bool in_alpha = color_type & PNG_COLOR_MASK_ALPHA;

  if (out_color != in_color) {
    if (out_color) {
      png_set_gray_to_rgb(impl.png);
    } else {
      png_set_rgb_to_gray_fixed(impl.png, 1, 33000, 33000);
    }
  }
  if (out_alpha != in_alpha) {
    if (out_alpha) {
      TraceError("The loaded PNG does not have an alpha channel, but one was requested");
      return false;
    } else {
      png_set_strip_alpha(impl.png);
    }
  }

  if (bit_depth == 16) {
    if (sizeof(K) == 1) {
      png_set_strip_16(impl.png);
    } else if (typeid(K) != typeid(uint16_t)) {
      TraceError("16-bit images can only be loaded into either 8ub or ui16 images");
      return false;
    }
    // TODO Should swap always be used?
    png_set_swap(impl.png);
  }

  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(impl.png);
  }
  if (bit_depth < 8) {
    png_set_packing(impl.png);
  }

  png_read_update_info(impl.png, impl.info);

  const size_t png_row_bytes = png_get_rowbytes(impl.png, impl.info);
  const size_t image_row_bytes = width * N * sizeof(K);
  if (png_row_bytes != image_row_bytes) {
    TraceError("Dimensions mismatch %zu != %zu", png_row_bytes, image_row_bytes);
    return false;
  }

  std::vector<png_bytep> row_pointers(height);
  for (int y = 0; y < height; y++) {
    row_pointers[y] = reinterpret_cast<png_bytep>(data) + y * width * N * sizeof(K);
  }
  png_read_image(impl.png, row_pointers.data());
  return true;
}

// Helper to save a png file
template <int N>
bool SavePngImpl(const std::string& filename, size_t rows, size_t cols, const uint8_t* img_ptr, size_t pitch) {
  struct Impl {
    std::FILE* fp;
    png_structp png;
    png_infop info;

    ~Impl() {
      png_destroy_write_struct(&png, &info);
      std::fclose(fp);
    }
  };

  CHECK(N == 1 || N == 3 || N == 4, false, "Cannot save an image with %d channels", N);

  std::FILE* fp = std::fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    TraceError("Could not open file for writing: %s. Errno: %i", filename.c_str(), errno);
    return false;
  }

  // Store resources in a smart object so that they get automatically deleted
  auto impl = std::make_unique<Impl>();
  impl->fp = fp;

  impl->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  CHECK(impl->png != nullptr, false, "png_create_write_struct() failed");
  impl->info = png_create_info_struct(impl->png);
  CHECK(impl->info != nullptr, false, "png_create_info_struct() failed");

  if (setjmp(png_jmpbuf(impl->png)) != 0) {
    TraceError("setjmp(png_jmpbuf(png)) failed");
    return false;
  }

  png_init_io(impl->png, impl->fp);

  png_set_IHDR(impl->png, impl->info, cols, rows, sizeof(uint8_t) * 8,
               (N == 1 ? PNG_COLOR_TYPE_GRAY : (N == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA)), PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(impl->png, impl->info);

  // in case uint16_t data type is supported
  //   if (std::is_same_v<K, uint16_t>) {
  //     png_set_swap(impl->png);
  //   }

  png_set_compression_level(impl->png, 1);

  for (size_t row = 0; row < rows; ++row) {
    png_write_row(impl->png, reinterpret_cast<png_const_bytep>(img_ptr + pitch * row));
  }

  png_write_end(impl->png, nullptr);
  return true;
}

// Helper to get JPEG header information
bool LoadJpegInfo(const std::string& filename, std::unique_ptr<JpegImpl>& impl) {
  // Open file
  std::FILE* fp = std::fopen(filename.c_str(), "rb");
  if (fp == nullptr) {
    TraceError("Could not open file: '%s'", filename.c_str());
    return false;
  }
  impl = std::make_unique<JpegImpl>();
  impl->fp = fp;

  // create structures, read .jpeg header
  impl->cinfo.err = jpeg_std_error(&impl->jerr);

  jpeg_create_decompress(&impl->cinfo);
  jpeg_stdio_src(&impl->cinfo, fp);
  jpeg_read_header(&impl->cinfo, TRUE);

  // Check the image size from the JPEG header.
  if (!(impl->cinfo.image_height > 0 && impl->cinfo.image_height < 65536)) {
    TraceError("Invalid JPEG image height.");
    return false;
  }
  if (!(impl->cinfo.image_width > 0 && impl->cinfo.image_width < 65536)) {
    TraceError("Invalid JPEG image width.");
    return false;
  }
  return true;
}

// Helper to load a JPEG into a buffer
template <int N, typename SizeT>
bool LoadJpegImpl(std::unique_ptr<JpegImpl>& impl, uint8_t* data, const std::array<SizeT, 2>& dimensions) {
  const bool out_color = N > 2;
  const bool out_alpha = N == 2 || N == 4;
  if (out_alpha) {
    TraceError("The JPEG reader does not support an alpha channel, but one was requested");
    return false;
  }

  // Set parameters for decompression, defaults are from jpeg_read_header()
  impl->cinfo.out_color_space = out_color ? JCS_RGB : JCS_GRAYSCALE;

  // Check image dimensions.
  if (static_cast<int>(impl->cinfo.image_height) != dimensions[0] ||
      static_cast<int>(impl->cinfo.image_width) != dimensions[1]) {
    TraceError(
        "Buffer dimensions must match the image dimensions."
        "Expected %u x %u, received buffer with %d x %d",
        impl->cinfo.image_height, impl->cinfo.image_width, static_cast<int>(dimensions[0]),
        static_cast<int>(dimensions[1]));
    return false;
  }

  // decompress
  jpeg_start_decompress(&impl->cinfo);

  while (impl->cinfo.output_scanline < impl->cinfo.image_height) {
    JSAMPROW samp = const_cast<uint8_t*>(data + impl->cinfo.output_scanline * dimensions[1] * N);
    jpeg_read_scanlines(&impl->cinfo, &samp, 1);
  }
  return true;
}

// Helper to save a jpg file
bool SaveJpegImpl(const std::string& filename, J_COLOR_SPACE type, int quality, const ImageMatrix<uint8_t>& image) {
  std::FILE* fp = std::fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    TraceError("Could not open the file: %s", filename.c_str());
    return false;
  }

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  cinfo.image_width = image.cols();
  cinfo.image_height = image.rows();
  cinfo.input_components = type == JCS_GRAYSCALE ? 1 : 3;
  cinfo.in_color_space = type;

  jpeg_stdio_dest(&cinfo, fp);

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW samp = const_cast<uint8_t*>(image.row(cinfo.next_scanline).data());
    jpeg_write_scanlines(&cinfo, &samp, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  std::fclose(fp);
  return true;
}

}  // namespace

template <typename K>
bool LoadPng(const std::string& filename, ImageMatrix<K>& image) {
  assert(image.IsRowMajor);

  PngImpl impl;

  if (!LoadPngInfo(filename, impl)) {
    return false;
  }
  image.resize(png_get_image_height(impl.png, impl.info), png_get_image_width(impl.png, impl.info));
  return LoadPngImpl<K, 1>(impl, image.data(), image.rows(), image.cols());
}

bool LoadPng(const std::string& filename, ImageMatrix<uint8_t>& image) { return LoadPng<uint8_t>(filename, image); }

bool LoadPng(const std::string& filename, ImageMatrix<uint16_t>& image) { return LoadPng<uint16_t>(filename, image); }

bool LoadPngCached(const std::string& filename, ImageMatrix<uint8_t>& image) {
  const std::string cache = filename + ".tga";
  if (LoadTga(cache, image)) return true;
  if (!LoadPng(filename, image)) return false;
  SaveTga(image, cache);
  return true;
}

bool SavePng(const ImageMatrix<uint8_t>& image, const std::string& filename) {
  return SavePngImpl<1>(filename, image.rows(), image.cols(), image.data(), image.cols() * sizeof(uint8_t));
}

bool SaveRGB(const std::string& filename, size_t rows, size_t cols, const uint8_t* img_ptr, size_t pitch) {
  return SavePngImpl<3>(filename, rows, cols, img_ptr, pitch);
}

bool LoadJpeg(const std::string& filename, ImageMatrix<uint8_t>& image) {
  std::unique_ptr<JpegImpl> impl;
  if (!LoadJpegInfo(filename, impl)) return false;
  image.resize(static_cast<size_t>(impl->cinfo.image_height), static_cast<size_t>(impl->cinfo.image_width));
  return LoadJpegImpl<1>(impl, image.data(), std::array{image.rows(), image.cols()});
}

bool SaveJpeg(const ImageMatrix<uint8_t>& image, const std::string& filename, const int quality) {
  return SaveJpegImpl(filename, JCS_GRAYSCALE, quality, image);
}

}  // namespace cuvslam
