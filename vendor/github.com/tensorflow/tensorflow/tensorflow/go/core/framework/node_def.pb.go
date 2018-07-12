// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/node_def.proto

package framework

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

type NodeDef struct {
	// The name given to this operator. Used for naming inputs,
	// logging, visualization, etc.  Unique within a single GraphDef.
	// Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_./]*".
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The operation name.  There may be custom parameters in attrs.
	// Op names starting with an underscore are reserved for internal use.
	Op string `protobuf:"bytes,2,opt,name=op,proto3" json:"op,omitempty"`
	// Each input is "node:src_output" with "node" being a string name and
	// "src_output" indicating which output tensor to use from "node". If
	// "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
	// may optionally be followed by control inputs that have the format
	// "^node".
	Input []string `protobuf:"bytes,3,rep,name=input" json:"input,omitempty"`
	// A (possibly partial) specification for the device on which this
	// node should be placed.
	// The expected syntax for this string is as follows:
	//
	// DEVICE_SPEC ::= PARTIAL_SPEC
	//
	// PARTIAL_SPEC ::= ("/" CONSTRAINT) *
	// CONSTRAINT ::= ("job:" JOB_NAME)
	//              | ("replica:" [1-9][0-9]*)
	//              | ("task:" [1-9][0-9]*)
	//              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
	//
	// Valid values for this string include:
	// * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
	// * "/job:worker/device:GPU:3"                   (partial specification)
	// * ""                                    (no specification)
	//
	// If the constraints do not resolve to a single device (or if this
	// field is empty or not present), the runtime will attempt to
	// choose a device automatically.
	Device string `protobuf:"bytes,4,opt,name=device,proto3" json:"device,omitempty"`
	// Operation-specific graph-construction-time configuration.
	// Note that this should include all attrs defined in the
	// corresponding OpDef, including those with a value matching
	// the default -- this allows the default to change and makes
	// NodeDefs easier to interpret on their own.  However, if
	// an attr with a default is not specified in this list, the
	// default will be used.
	// The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
	// one of the names from the corresponding OpDef's attr field).
	// The values must have a type matching the corresponding OpDef
	// attr's type field.
	// TODO(josh11b): Add some examples here showing best practices.
	Attr map[string]*AttrValue `protobuf:"bytes,5,rep,name=attr" json:"attr,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value"`
}

func (m *NodeDef) Reset()                    { *m = NodeDef{} }
func (m *NodeDef) String() string            { return proto.CompactTextString(m) }
func (*NodeDef) ProtoMessage()               {}
func (*NodeDef) Descriptor() ([]byte, []int) { return fileDescriptorNodeDef, []int{0} }

func (m *NodeDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *NodeDef) GetOp() string {
	if m != nil {
		return m.Op
	}
	return ""
}

func (m *NodeDef) GetInput() []string {
	if m != nil {
		return m.Input
	}
	return nil
}

func (m *NodeDef) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *NodeDef) GetAttr() map[string]*AttrValue {
	if m != nil {
		return m.Attr
	}
	return nil
}

func init() {
	proto.RegisterType((*NodeDef)(nil), "tensorflow.NodeDef")
}

func init() { proto.RegisterFile("tensorflow/core/framework/node_def.proto", fileDescriptorNodeDef) }

var fileDescriptorNodeDef = []byte{
	// 282 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x90, 0x4f, 0x4b, 0xc3, 0x40,
	0x10, 0xc5, 0xd9, 0xfc, 0xa9, 0x64, 0x0a, 0x22, 0x8b, 0xca, 0x52, 0x10, 0x82, 0xa7, 0xa0, 0x90,
	0x60, 0xbd, 0x88, 0xe0, 0xc1, 0xa2, 0xd7, 0x52, 0x72, 0xf0, 0xe0, 0xa5, 0xa4, 0xc9, 0x24, 0x86,
	0x36, 0x99, 0xb0, 0x6e, 0x5a, 0xfa, 0x65, 0xfd, 0x1c, 0x1e, 0x65, 0x37, 0xa1, 0x0d, 0xa2, 0xb7,
	0x37, 0xb3, 0xef, 0xed, 0x3c, 0x7e, 0x10, 0x28, 0xac, 0x3f, 0x49, 0xe6, 0x1b, 0xda, 0x45, 0x29,
	0x49, 0x8c, 0x72, 0x99, 0x54, 0xb8, 0x23, 0xb9, 0x8e, 0x6a, 0xca, 0x70, 0x99, 0x61, 0x1e, 0x36,
	0x92, 0x14, 0x71, 0x38, 0x3a, 0x27, 0x37, 0xff, 0xa7, 0x12, 0xa5, 0xe4, 0x72, 0x9b, 0x6c, 0x5a,
	0xec, 0x72, 0xd7, 0x5f, 0x0c, 0x4e, 0xe6, 0x94, 0xe1, 0x0b, 0xe6, 0x9c, 0x83, 0x53, 0x27, 0x15,
	0x0a, 0xe6, 0xb3, 0xc0, 0x8b, 0x8d, 0xe6, 0xa7, 0x60, 0x51, 0x23, 0x2c, 0xb3, 0xb1, 0xa8, 0xe1,
	0xe7, 0xe0, 0x96, 0x75, 0xd3, 0x2a, 0x61, 0xfb, 0x76, 0xe0, 0xc5, 0xdd, 0xc0, 0x2f, 0x61, 0x94,
	0xe1, 0xb6, 0x4c, 0x51, 0x38, 0xc6, 0xd9, 0x4f, 0xfc, 0x0e, 0x1c, 0x7d, 0x51, 0xb8, 0xbe, 0x1d,
	0x8c, 0xa7, 0x57, 0xe1, 0xb1, 0x58, 0xd8, 0x1f, 0x0d, 0x9f, 0x95, 0x92, 0xaf, 0xb5, 0x92, 0xfb,
	0xd8, 0x58, 0x27, 0x73, 0xf0, 0x0e, 0x2b, 0x7e, 0x06, 0xf6, 0x1a, 0xf7, 0x7d, 0x21, 0x2d, 0xf9,
	0x2d, 0xb8, 0xa6, 0xbe, 0xa9, 0x34, 0x9e, 0x5e, 0x0c, 0xbf, 0xd4, 0xb9, 0x37, 0xfd, 0x18, 0x77,
	0x9e, 0x47, 0xeb, 0x81, 0xcd, 0x4a, 0x10, 0x24, 0x8b, 0xa1, 0xed, 0x40, 0x63, 0xe6, 0xe9, 0x12,
	0x0b, 0xcd, 0x61, 0xc1, 0xde, 0x9f, 0x8a, 0x52, 0x7d, 0xb4, 0xab, 0x30, 0xa5, 0x2a, 0x1a, 0x00,
	0xfc, 0x5b, 0x16, 0xf4, 0x8b, 0xec, 0x37, 0x63, 0xab, 0x91, 0x41, 0x7a, 0xff, 0x13, 0x00, 0x00,
	0xff, 0xff, 0xd8, 0xfa, 0x75, 0x83, 0xb6, 0x01, 0x00, 0x00,
}
