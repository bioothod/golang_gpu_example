// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/protobuf/worker_service.proto

package protobuf

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/worker_service.proto", fileDescriptorWorkerService)
}

var fileDescriptorWorkerService = []byte{
	// 389 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x94, 0x41, 0x6b, 0xe2, 0x40,
	0x14, 0xc7, 0x77, 0x2f, 0xeb, 0xee, 0xb0, 0x22, 0x8c, 0xb7, 0xa8, 0xbb, 0xae, 0x4b, 0xdb, 0x53,
	0x13, 0x68, 0xaf, 0xbd, 0x18, 0x2d, 0x16, 0xda, 0x82, 0x8d, 0x42, 0xc1, 0x4b, 0x89, 0xe9, 0x73,
	0x0c, 0x8d, 0x99, 0xf8, 0x66, 0xa2, 0x9f, 0xa5, 0xdf, 0xb6, 0x34, 0x99, 0xe8, 0x0c, 0x1d, 0xf5,
	0x16, 0xfe, 0xff, 0xdf, 0xfb, 0xbd, 0x81, 0x30, 0x43, 0x2e, 0x25, 0xa4, 0x82, 0xe3, 0x22, 0xe1,
	0x5b, 0x2f, 0xe2, 0x08, 0x5e, 0x86, 0x5c, 0xf2, 0x79, 0xbe, 0xf0, 0xb6, 0x1c, 0xdf, 0x00, 0x5f,
	0x04, 0xe0, 0x26, 0x8e, 0xc0, 0x2d, 0x72, 0xda, 0xd8, 0xe3, 0x2e, 0xc3, 0x2c, 0x72, 0xce, 0x4e,
	0xcc, 0x97, 0x73, 0x57, 0xef, 0x35, 0x52, 0x7f, 0x2e, 0x82, 0x49, 0xe9, 0xa3, 0x77, 0xe4, 0xd7,
	0x08, 0xe4, 0x44, 0x86, 0x32, 0x17, 0xb4, 0xed, 0x6a, 0xde, 0x5d, 0x1c, 0xc0, 0x3a, 0x07, 0x21,
	0x9d, 0xce, 0x81, 0x56, 0x64, 0x3c, 0x15, 0x40, 0x17, 0xa4, 0x39, 0x40, 0x08, 0x25, 0x54, 0x0b,
	0x84, 0x88, 0x79, 0x4a, 0xcf, 0xf5, 0x29, 0x0b, 0x50, 0xd9, 0x2f, 0x4e, 0x72, 0xfb, 0x3d, 0x43,
	0x48, 0xe0, 0xe8, 0x1e, 0x0b, 0x60, 0xdd, 0x63, 0xe5, 0xd4, 0x9e, 0x29, 0xa9, 0x07, 0xc0, 0x62,
	0x21, 0x01, 0x47, 0x18, 0x66, 0x4b, 0xda, 0xd5, 0x27, 0x8d, 0xaa, 0x72, 0xff, 0x3b, 0x42, 0x28,
	0xeb, 0x8c, 0x34, 0x86, 0x80, 0x86, 0xb7, 0x67, 0x9e, 0x08, 0x6d, 0xe6, 0xff, 0x47, 0x19, 0xe5,
	0xbe, 0x25, 0x3f, 0x83, 0x3c, 0x2d, 0xa5, 0x2d, 0xe3, 0x28, 0x2a, 0xad, 0x6c, 0x6d, 0x7b, 0xa9,
	0x34, 0x4f, 0xe4, 0xf7, 0x20, 0x81, 0x30, 0xcd, 0xb3, 0x52, 0xf5, 0xd7, 0xf8, 0x33, 0x5a, 0x53,
	0xe9, 0xba, 0x87, 0x01, 0xa5, 0xbc, 0x27, 0x44, 0xe5, 0xfd, 0x24, 0xa1, 0x1d, 0x0b, 0xdf, 0x4f,
	0x92, 0x4a, 0xf7, 0xe7, 0x50, 0xad, 0x64, 0x8f, 0x84, 0x04, 0x10, 0x6d, 0xa6, 0x05, 0x64, 0xca,
	0xf6, 0xb9, 0x55, 0xa6, 0xd7, 0xa5, 0xac, 0xf7, 0x8d, 0xfa, 0xa4, 0xf6, 0xc0, 0x19, 0x8b, 0x53,
	0x46, 0x1d, 0x1d, 0x56, 0x61, 0x25, 0x6a, 0x59, 0x3b, 0x75, 0x24, 0x9f, 0xd4, 0xa6, 0x18, 0x46,
	0x5f, 0x1c, 0x2a, 0xb4, 0x3a, 0x76, 0x5d, 0xe9, 0xf0, 0xd7, 0xc4, 0xe1, 0xc8, 0x74, 0xe2, 0x35,
	0x16, 0x12, 0xf3, 0x54, 0xc6, 0x2b, 0xf0, 0x9b, 0xc6, 0xb5, 0x1d, 0x7f, 0xde, 0x66, 0x31, 0xfe,
	0x3e, 0xbb, 0x61, 0xb1, 0x5c, 0xe6, 0x73, 0x37, 0xe2, 0x2b, 0x4f, 0x7b, 0x02, 0xec, 0x9f, 0x8c,
	0x9b, 0x6f, 0xc3, 0xfc, 0x47, 0xf1, 0x75, 0xfd, 0x11, 0x00, 0x00, 0xff, 0xff, 0xa7, 0x76, 0x50,
	0xfb, 0x7e, 0x04, 0x00, 0x00,
}
