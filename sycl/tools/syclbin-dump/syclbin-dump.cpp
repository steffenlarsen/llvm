//==----------- syclbin-dump.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "syclbin-dump" utility lists the contents of a SYCLBIN file in a
// human-readable format.
//

#include "detail/syclbin.hpp"
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace llvm;

std::string_view StateToString(uint8_t State) {
  switch (State) {
  case 0:
    return "input";
  case 1:
    return "object";
  case 2:
    return "executable";
  default:
    return "UNKNOWN";
  }
}

int main(int argc, char **argv, char *env[]) {
  cl::opt<std::string> TargetSYCLBIN(
      cl::Positional, cl::desc("<target syclbin>"), cl::Required);

  cl::ParseCommandLineOptions(argc, argv);

  std::string TargetFilename{TargetSYCLBIN};
  std::ifstream InputStream(TargetFilename.c_str(), std::ios::binary);
  std::vector<char> RawSYCLBINData{std::istreambuf_iterator<char>(InputStream),
                                   std::istreambuf_iterator<char>()};
  InputStream.close();

  sycl::detail::SYCLBIN ParsedSYCLBIN;
  try {
    ParsedSYCLBIN = sycl::detail::ParseSYCLBIN(RawSYCLBINData.data(),
                                               RawSYCLBINData.size());
  } catch (sycl::exception &e) {
    std::cerr << "Failed to parse SYCLBIN file: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Version:                    " << ParsedSYCLBIN.Header.Version
            << "\n";
  std::cout << "State:                      "
            << StateToString(ParsedSYCLBIN.Header.State) << "\n";
  std::cout << "Number of Abstract Modules: "
            << ParsedSYCLBIN.AbstractModules.size() << "\n";

  for (size_t I = 0; I < ParsedSYCLBIN.AbstractModules.size(); ++I) {
    const sycl::detail::SYCLBIN::AbstractModule &AM =
        ParsedSYCLBIN.AbstractModules[I];

    std::cout << "Abstract Module " << I << ":\n";

    // Metadata.
    std::cout << "  Metadata:\n";
    std::cout << "    Kernel names:\n";
    for (const std::string &KernelName : AM.KernelNames)
      std::cout << "      " << KernelName << "\n";
    std::cout << "    Imported symbols:\n";
    for (const std::string &ImportedSymbol : AM.ImportedSymbols)
      std::cout << "      " << ImportedSymbol << "\n";
    std::cout << "    Exported symbols:\n";
    for (const std::string &ExportedSymbol : AM.ExportedSymbols)
      std::cout << "      " << ExportedSymbol << "\n";
    std::cout << "    Properties: <Binary blob of " << AM.Properties.size()
              << " bytes>\n";

    // IR Modules.
    std::cout << "  Number of IR Modules: " << AM.IRModules.size() << "\n";
    for (size_t J = 0; J < AM.IRModules.size(); ++J) {
      const sycl::detail::SYCLBIN::IRModule &IRM = AM.IRModules[J];
      std::cout << "  IR module " << J << ":\n";
      std::cout << "    IR type: " << IRM.Type << "\n";
      std::cout << "    Raw IR bytes: <Binary blob of " << IRM.RawIRBytes.size()
                << " bytes>\n";
    }

    // Native device code images.
    std::cout << "  Number of Native Device Code Images: "
              << AM.NativeDeviceCodeImages.size() << "\n";
    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const sycl::detail::SYCLBIN::NativeDeviceCodeImage &NDCI =
          AM.NativeDeviceCodeImages[J];
      std::cout << "  Native device code image " << J << ":\n";
      std::cout << "    Architecture: " << NDCI.ArchString << "\n";
      std::cout << "    Raw native device code image bytes: <Binary blob of "
                << NDCI.RawDeviceCodeImageBytes.size() << " bytes>\n";
    }
  }

  std::cout << std::flush;
  return 0;
}
