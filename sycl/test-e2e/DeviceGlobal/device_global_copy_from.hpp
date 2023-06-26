#pragma once

#include "common.hpp"

struct StructWithMembers {
  int x;
  float y;
};

device_global<int[4], TestProperties> DeviceGlobalVar1;
device_global<StructWithMembers, TestProperties> DeviceGlobalVar2;

constexpr int RefArr[4] = {104, 102, 108, 106};
constexpr StructWithMembers RefStruct{42, 3.14f};

int test() {
  queue Q;

  Q.submit([&](handler &CGH) {
     CGH.single_task([=]() {
       for (size_t I = 0; I < 4; ++I)
         DeviceGlobalVar1[I] = RefArr[I];
       DeviceGlobalVar2 = RefStruct;
     });
   }).wait_and_throw();

  int Out1[4] = {0};
  int Out2[4] = {0};
  StructWithMembers Out3;

  Q.memcpy(Out1, DeviceGlobalVar1);
  Q.memcpy(Out2, DeviceGlobalVar1, 2 * sizeof(int));
  Q.memcpy(Out2 + 3, DeviceGlobalVar1, sizeof(int), 3 * sizeof(int));
  Q.memcpy(&Out3, DeviceGlobalVar2);
  Q.wait_and_throw();

  for (size_t I = 0; I < 4; ++I) {
    assert(Out1[I] == RefArr[I] &&
           "Value in fully copied array does not match.");
    assert(Out2[I] == (I != 2 ? RefArr[I] : 0) &&
           "Value in partially copied array does not match.");
  }
  assert(Out3.x == RefStruct.x && "x in copied struct does not match.");
  assert(Out3.y == RefStruct.y && "y in copied struct does not match.");
  return 0;
}
