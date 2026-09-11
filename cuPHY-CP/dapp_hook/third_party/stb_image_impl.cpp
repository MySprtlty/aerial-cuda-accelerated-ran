/*
 * Single translation unit for the stb_image implementation. Compiled with
 * warnings disabled (see CMakeLists.txt) so the vendored header does not have
 * to satisfy the project's -Werror flags; dapp_yolo.hpp only sees the
 * declarations.
 */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "stb_image.h"
