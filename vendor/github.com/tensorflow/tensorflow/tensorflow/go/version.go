/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

// #include <string.h>
// #include "tensorflow/c/c_api.h"
// #include "tensorflow/core/public/version.h"
//
// const char *APIVersion() { return TF_VERSION_STRING; }
//
import "C"

// Version returns a string describing the version of the underlying TensorFlow
// runtime.
func Version() string { return C.GoString(C.TF_Version()) }

// APIVersion returns a string describing the version of the libtensorflow API
// used to compile golang binding. Normally Version() and APIVersion() should
// return the same string.
func APIVersion() string { return C.GoString(C.APIVersion()) }
