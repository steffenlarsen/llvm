#pragma once

#include "common.hpp"

struct StructWithMembers {
  int x;
  float y;
};

device_global<int[4], TestProperties> DeviceGlobalVar1;
device_global<int[4], TestProperties> DeviceGlobalVar2;
device_global<StructWithMembers, TestProperties> DeviceGlobalVar3;

int test() {
  queue Q;

  int RefArr[4] = {104, 102, 108, 106};
  StructWithMembers RefStruct{42, 3.14f};

  // Note: For DeviceGlobalVar2 we only copy the first two and the last element,
  // so the third element should be zero.
  Q.copy(RefArr, DeviceGlobalVar1);
  Q.copy(RefArr, DeviceGlobalVar2, 2);
  Q.copy(RefArr + 3, DeviceGlobalVar2, 1, 3);
  Q.copy(&RefStruct, DeviceGlobalVar3);
  Q.wait_and_throw();

  int Out1[4] = {0};
  int Out2[4] = {0};
  StructWithMembers Out3;
  {
    buffer<int, 1> OutBuf1{Out1, 4};
    buffer<int, 1> OutBuf2{Out2, 4};
    buffer<StructWithMembers, 1> OutBuf3{&Out3, 1};
    Q.submit([&](handler &CGH) {
      auto OutAcc1 = OutBuf1.get_access<access::mode::write>(CGH);
      auto OutAcc2 = OutBuf2.get_access<access::mode::write>(CGH);
      auto OutAcc3 = OutBuf3.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() {
        for (size_t I = 0; I < 4; ++I) {
          OutAcc1[I] = DeviceGlobalVar1[I];
          OutAcc2[I] = DeviceGlobalVar2[I];
        }
        OutAcc3[0] = DeviceGlobalVar3;
      });
    });
  }

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
