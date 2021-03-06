// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/op_def.proto

package framework

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// Defines an operation. A NodeDef in a GraphDef specifies an Op by
// using the "op" field which should match the name of a OpDef.
// LINT.IfChange
type OpDef struct {
	// Op names starting with an underscore are reserved for internal use.
	// Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9_]*".
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Description of the input(s).
	InputArg []*OpDef_ArgDef `protobuf:"bytes,2,rep,name=input_arg,json=inputArg" json:"input_arg,omitempty"`
	// Description of the output(s).
	OutputArg []*OpDef_ArgDef  `protobuf:"bytes,3,rep,name=output_arg,json=outputArg" json:"output_arg,omitempty"`
	Attr      []*OpDef_AttrDef `protobuf:"bytes,4,rep,name=attr" json:"attr,omitempty"`
	// Optional deprecation based on GraphDef versions.
	Deprecation *OpDeprecation `protobuf:"bytes,8,opt,name=deprecation" json:"deprecation,omitempty"`
	// One-line human-readable description of what the Op does.
	Summary string `protobuf:"bytes,5,opt,name=summary,proto3" json:"summary,omitempty"`
	// Additional, longer human-readable description of what the Op does.
	Description string `protobuf:"bytes,6,opt,name=description,proto3" json:"description,omitempty"`
	// True if the operation is commutative ("op(a,b) == op(b,a)" for all inputs)
	IsCommutative bool `protobuf:"varint,18,opt,name=is_commutative,json=isCommutative,proto3" json:"is_commutative,omitempty"`
	// If is_aggregate is true, then this operation accepts N >= 2
	// inputs and produces 1 output all of the same type.  Should be
	// associative and commutative, and produce output with the same
	// shape as the input.  The optimizer may replace an aggregate op
	// taking input from multiple devices with a tree of aggregate ops
	// that aggregate locally within each device (and possibly within
	// groups of nearby devices) before communicating.
	// TODO(josh11b): Implement that optimization.
	IsAggregate bool `protobuf:"varint,16,opt,name=is_aggregate,json=isAggregate,proto3" json:"is_aggregate,omitempty"`
	// By default Ops may be moved between devices.  Stateful ops should
	// either not be moved, or should only be moved if that state can also
	// be moved (e.g. via some sort of save / restore).
	// Stateful ops are guaranteed to never be optimized away by Common
	// Subexpression Elimination (CSE).
	IsStateful bool `protobuf:"varint,17,opt,name=is_stateful,json=isStateful,proto3" json:"is_stateful,omitempty"`
	// By default, all inputs to an Op must be initialized Tensors.  Ops
	// that may initialize tensors for the first time should set this
	// field to true, to allow the Op to take an uninitialized Tensor as
	// input.
	AllowsUninitializedInput bool `protobuf:"varint,19,opt,name=allows_uninitialized_input,json=allowsUninitializedInput,proto3" json:"allows_uninitialized_input,omitempty"`
}

func (m *OpDef) Reset()                    { *m = OpDef{} }
func (m *OpDef) String() string            { return proto.CompactTextString(m) }
func (*OpDef) ProtoMessage()               {}
func (*OpDef) Descriptor() ([]byte, []int) { return fileDescriptorOpDef, []int{0} }

func (m *OpDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *OpDef) GetInputArg() []*OpDef_ArgDef {
	if m != nil {
		return m.InputArg
	}
	return nil
}

func (m *OpDef) GetOutputArg() []*OpDef_ArgDef {
	if m != nil {
		return m.OutputArg
	}
	return nil
}

func (m *OpDef) GetAttr() []*OpDef_AttrDef {
	if m != nil {
		return m.Attr
	}
	return nil
}

func (m *OpDef) GetDeprecation() *OpDeprecation {
	if m != nil {
		return m.Deprecation
	}
	return nil
}

func (m *OpDef) GetSummary() string {
	if m != nil {
		return m.Summary
	}
	return ""
}

func (m *OpDef) GetDescription() string {
	if m != nil {
		return m.Description
	}
	return ""
}

func (m *OpDef) GetIsCommutative() bool {
	if m != nil {
		return m.IsCommutative
	}
	return false
}

func (m *OpDef) GetIsAggregate() bool {
	if m != nil {
		return m.IsAggregate
	}
	return false
}

func (m *OpDef) GetIsStateful() bool {
	if m != nil {
		return m.IsStateful
	}
	return false
}

func (m *OpDef) GetAllowsUninitializedInput() bool {
	if m != nil {
		return m.AllowsUninitializedInput
	}
	return false
}

// For describing inputs and outputs.
type OpDef_ArgDef struct {
	// Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Human readable description.
	Description string `protobuf:"bytes,2,opt,name=description,proto3" json:"description,omitempty"`
	// Describes the type of one or more tensors that are accepted/produced
	// by this input/output arg.  The only legal combinations are:
	// * For a single tensor: either the "type" field is set or the
	//   "type_attr" field is set to the name of an attr with type "type".
	// * For a sequence of tensors with the same type: the "number_attr"
	//   field will be set to the name of an attr with type "int", and
	//   either the "type" or "type_attr" field will be set as for
	//   single tensors.
	// * For a sequence of tensors, the "type_list_attr" field will be set
	//   to the name of an attr with type "list(type)".
	Type       DataType `protobuf:"varint,3,opt,name=type,proto3,enum=tensorflow.DataType" json:"type,omitempty"`
	TypeAttr   string   `protobuf:"bytes,4,opt,name=type_attr,json=typeAttr,proto3" json:"type_attr,omitempty"`
	NumberAttr string   `protobuf:"bytes,5,opt,name=number_attr,json=numberAttr,proto3" json:"number_attr,omitempty"`
	// If specified, attr must have type "list(type)", and none of
	// type, type_attr, and number_attr may be specified.
	TypeListAttr string `protobuf:"bytes,6,opt,name=type_list_attr,json=typeListAttr,proto3" json:"type_list_attr,omitempty"`
	// For inputs: if true, the inputs are required to be refs.
	//   By default, inputs can be either refs or non-refs.
	// For outputs: if true, outputs are refs, otherwise they are not.
	IsRef bool `protobuf:"varint,16,opt,name=is_ref,json=isRef,proto3" json:"is_ref,omitempty"`
}

func (m *OpDef_ArgDef) Reset()                    { *m = OpDef_ArgDef{} }
func (m *OpDef_ArgDef) String() string            { return proto.CompactTextString(m) }
func (*OpDef_ArgDef) ProtoMessage()               {}
func (*OpDef_ArgDef) Descriptor() ([]byte, []int) { return fileDescriptorOpDef, []int{0, 0} }

func (m *OpDef_ArgDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *OpDef_ArgDef) GetDescription() string {
	if m != nil {
		return m.Description
	}
	return ""
}

func (m *OpDef_ArgDef) GetType() DataType {
	if m != nil {
		return m.Type
	}
	return DataType_DT_INVALID
}

func (m *OpDef_ArgDef) GetTypeAttr() string {
	if m != nil {
		return m.TypeAttr
	}
	return ""
}

func (m *OpDef_ArgDef) GetNumberAttr() string {
	if m != nil {
		return m.NumberAttr
	}
	return ""
}

func (m *OpDef_ArgDef) GetTypeListAttr() string {
	if m != nil {
		return m.TypeListAttr
	}
	return ""
}

func (m *OpDef_ArgDef) GetIsRef() bool {
	if m != nil {
		return m.IsRef
	}
	return false
}

// Description of the graph-construction-time configuration of this
// Op.  That is to say, this describes the attr fields that will
// be specified in the NodeDef.
type OpDef_AttrDef struct {
	// A descriptive name for the argument.  May be used, e.g. by the
	// Python client, as a keyword argument name, and so should match
	// the regexp "[a-z][a-z0-9_]+".
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// One of the type names from attr_value.proto ("string", "list(string)",
	// "int", etc.).
	Type string `protobuf:"bytes,2,opt,name=type,proto3" json:"type,omitempty"`
	// A reasonable default for this attribute if the user does not supply
	// a value.  If not specified, the user must supply a value.
	DefaultValue *AttrValue `protobuf:"bytes,3,opt,name=default_value,json=defaultValue" json:"default_value,omitempty"`
	// Human-readable description.
	Description string `protobuf:"bytes,4,opt,name=description,proto3" json:"description,omitempty"`
	// For type == "int", this is a minimum value.  For "list(___)"
	// types, this is the minimum length.
	HasMinimum bool  `protobuf:"varint,5,opt,name=has_minimum,json=hasMinimum,proto3" json:"has_minimum,omitempty"`
	Minimum    int64 `protobuf:"varint,6,opt,name=minimum,proto3" json:"minimum,omitempty"`
	// The set of allowed values.  Has type that is the "list" version
	// of the "type" field above (uses the "list" field of AttrValue).
	// If type == "type" or "list(type)" above, then the "type" field
	// of "allowed_values.list" has the set of allowed DataTypes.
	// If type == "string" or "list(string)", then the "s" field of
	// "allowed_values.list" has the set of allowed strings.
	AllowedValues *AttrValue `protobuf:"bytes,7,opt,name=allowed_values,json=allowedValues" json:"allowed_values,omitempty"`
}

func (m *OpDef_AttrDef) Reset()                    { *m = OpDef_AttrDef{} }
func (m *OpDef_AttrDef) String() string            { return proto.CompactTextString(m) }
func (*OpDef_AttrDef) ProtoMessage()               {}
func (*OpDef_AttrDef) Descriptor() ([]byte, []int) { return fileDescriptorOpDef, []int{0, 1} }

func (m *OpDef_AttrDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *OpDef_AttrDef) GetType() string {
	if m != nil {
		return m.Type
	}
	return ""
}

func (m *OpDef_AttrDef) GetDefaultValue() *AttrValue {
	if m != nil {
		return m.DefaultValue
	}
	return nil
}

func (m *OpDef_AttrDef) GetDescription() string {
	if m != nil {
		return m.Description
	}
	return ""
}

func (m *OpDef_AttrDef) GetHasMinimum() bool {
	if m != nil {
		return m.HasMinimum
	}
	return false
}

func (m *OpDef_AttrDef) GetMinimum() int64 {
	if m != nil {
		return m.Minimum
	}
	return 0
}

func (m *OpDef_AttrDef) GetAllowedValues() *AttrValue {
	if m != nil {
		return m.AllowedValues
	}
	return nil
}

// Information about version-dependent deprecation of an op
type OpDeprecation struct {
	// First GraphDef version at which the op is disallowed.
	Version int32 `protobuf:"varint,1,opt,name=version,proto3" json:"version,omitempty"`
	// Explanation of why it was deprecated and what to use instead.
	Explanation string `protobuf:"bytes,2,opt,name=explanation,proto3" json:"explanation,omitempty"`
}

func (m *OpDeprecation) Reset()                    { *m = OpDeprecation{} }
func (m *OpDeprecation) String() string            { return proto.CompactTextString(m) }
func (*OpDeprecation) ProtoMessage()               {}
func (*OpDeprecation) Descriptor() ([]byte, []int) { return fileDescriptorOpDef, []int{1} }

func (m *OpDeprecation) GetVersion() int32 {
	if m != nil {
		return m.Version
	}
	return 0
}

func (m *OpDeprecation) GetExplanation() string {
	if m != nil {
		return m.Explanation
	}
	return ""
}

// A collection of OpDefs
type OpList struct {
	Op []*OpDef `protobuf:"bytes,1,rep,name=op" json:"op,omitempty"`
}

func (m *OpList) Reset()                    { *m = OpList{} }
func (m *OpList) String() string            { return proto.CompactTextString(m) }
func (*OpList) ProtoMessage()               {}
func (*OpList) Descriptor() ([]byte, []int) { return fileDescriptorOpDef, []int{2} }

func (m *OpList) GetOp() []*OpDef {
	if m != nil {
		return m.Op
	}
	return nil
}

func init() {
	proto.RegisterType((*OpDef)(nil), "tensorflow.OpDef")
	proto.RegisterType((*OpDef_ArgDef)(nil), "tensorflow.OpDef.ArgDef")
	proto.RegisterType((*OpDef_AttrDef)(nil), "tensorflow.OpDef.AttrDef")
	proto.RegisterType((*OpDeprecation)(nil), "tensorflow.OpDeprecation")
	proto.RegisterType((*OpList)(nil), "tensorflow.OpList")
}

func init() { proto.RegisterFile("tensorflow/core/framework/op_def.proto", fileDescriptorOpDef) }

var fileDescriptorOpDef = []byte{
	// 647 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x94, 0xdd, 0x6e, 0xd3, 0x30,
	0x14, 0xc7, 0x95, 0x7e, 0xf7, 0x64, 0xad, 0x98, 0x61, 0x92, 0x29, 0x17, 0xeb, 0x26, 0x86, 0x2a,
	0x10, 0xad, 0x34, 0x84, 0x90, 0x60, 0x5c, 0x6c, 0xec, 0x06, 0x01, 0xda, 0x14, 0x3e, 0x2e, 0xb8,
	0x89, 0xbc, 0xd6, 0xc9, 0xac, 0x25, 0x71, 0x64, 0x3b, 0x1b, 0xe3, 0x09, 0x78, 0x39, 0x9e, 0x82,
	0x97, 0xe0, 0x12, 0xf9, 0x38, 0xd9, 0xc2, 0xda, 0xc1, 0x55, 0xe3, 0x73, 0x7e, 0x7f, 0xd7, 0xff,
	0xbf, 0x3f, 0xe0, 0x91, 0xe1, 0x99, 0x96, 0x2a, 0x4a, 0xe4, 0xc5, 0x6c, 0x2e, 0x15, 0x9f, 0x45,
	0x8a, 0xa5, 0xfc, 0x42, 0xaa, 0xb3, 0x99, 0xcc, 0xc3, 0x05, 0x8f, 0xa6, 0xb9, 0x92, 0x46, 0x12,
	0xb8, 0xe6, 0x46, 0x8f, 0x6f, 0xd7, 0x30, 0x63, 0x54, 0x78, 0xce, 0x92, 0x82, 0x3b, 0xdd, 0x68,
	0xe7, 0x76, 0xd6, 0x5c, 0xe6, 0x5c, 0x3b, 0x6c, 0xfb, 0x67, 0x17, 0xda, 0x47, 0xf9, 0x21, 0x8f,
	0x08, 0x81, 0x56, 0xc6, 0x52, 0x4e, 0xbd, 0xb1, 0x37, 0xe9, 0x07, 0xf8, 0x4d, 0x9e, 0x43, 0x5f,
	0x64, 0x79, 0x61, 0x42, 0xa6, 0x62, 0xda, 0x18, 0x37, 0x27, 0xfe, 0x2e, 0x9d, 0x5e, 0x4f, 0x3c,
	0x45, 0xe5, 0x74, 0x5f, 0xc5, 0x87, 0x3c, 0x0a, 0x7a, 0x88, 0xee, 0xab, 0x98, 0xbc, 0x00, 0x90,
	0x85, 0xa9, 0x74, 0xcd, 0xff, 0xe8, 0xfa, 0x8e, 0xb5, 0xc2, 0xa7, 0xd0, 0xb2, 0x46, 0x68, 0x0b,
	0x25, 0xf7, 0x57, 0x48, 0x8c, 0x51, 0x56, 0x83, 0x18, 0x79, 0x05, 0xfe, 0x82, 0xe7, 0x8a, 0xcf,
	0x99, 0x11, 0x32, 0xa3, 0xbd, 0xb1, 0xb7, 0x4a, 0x75, 0x05, 0x04, 0x75, 0x9a, 0x50, 0xe8, 0xea,
	0x22, 0x4d, 0x99, 0xba, 0xa4, 0x6d, 0xb4, 0x5c, 0x0d, 0xc9, 0xd8, 0x4e, 0xab, 0xe7, 0x4a, 0xe4,
	0x38, 0x6d, 0x07, 0xbb, 0xf5, 0x12, 0xd9, 0x81, 0xa1, 0xd0, 0xe1, 0x5c, 0xa6, 0x69, 0x61, 0x98,
	0x11, 0xe7, 0x9c, 0x92, 0xb1, 0x37, 0xe9, 0x05, 0x03, 0xa1, 0xdf, 0x5c, 0x17, 0xc9, 0x16, 0xac,
	0x09, 0x1d, 0xb2, 0x38, 0x56, 0x3c, 0x66, 0x86, 0xd3, 0x3b, 0x08, 0xf9, 0x42, 0xef, 0x57, 0x25,
	0xb2, 0x09, 0xbe, 0xd0, 0xa1, 0x36, 0xcc, 0xf0, 0xa8, 0x48, 0xe8, 0x3a, 0x12, 0x20, 0xf4, 0xc7,
	0xb2, 0x42, 0xf6, 0x60, 0xc4, 0x92, 0x44, 0x5e, 0xe8, 0xb0, 0xc8, 0x44, 0x26, 0x8c, 0x60, 0x89,
	0xf8, 0xce, 0x17, 0x21, 0x86, 0x4d, 0xef, 0x22, 0x4f, 0x1d, 0xf1, 0xb9, 0x0e, 0xbc, 0xb5, 0xfd,
	0xd1, 0x2f, 0x0f, 0x3a, 0x2e, 0xe6, 0x95, 0xfb, 0x7b, 0xc3, 0x69, 0x63, 0xd9, 0xe9, 0x04, 0x5a,
	0xf6, 0xb8, 0xd0, 0xe6, 0xd8, 0x9b, 0x0c, 0x77, 0xef, 0xd5, 0xb3, 0x3d, 0x64, 0x86, 0x7d, 0xba,
	0xcc, 0x79, 0x80, 0x04, 0x79, 0x00, 0x7d, 0xfb, 0x1b, 0x96, 0x1b, 0x68, 0x67, 0xea, 0xd9, 0x82,
	0xdd, 0x32, 0x6b, 0x33, 0x2b, 0xd2, 0x13, 0xae, 0x5c, 0xdb, 0x05, 0x0e, 0xae, 0x84, 0xc0, 0x43,
	0x18, 0xa2, 0x3a, 0x11, 0xda, 0x38, 0xc6, 0xc5, 0xbe, 0x66, 0xab, 0xef, 0x85, 0x36, 0x48, 0x6d,
	0x40, 0x47, 0xe8, 0x50, 0xf1, 0xa8, 0x8c, 0xb2, 0x2d, 0x74, 0xc0, 0xa3, 0xd1, 0x8f, 0x06, 0x74,
	0xcb, 0x93, 0xb1, 0xd2, 0x26, 0x29, 0x4d, 0x38, 0x7f, 0x6e, 0xb9, 0x2f, 0x61, 0xb0, 0xe0, 0x11,
	0x2b, 0x12, 0xe3, 0xae, 0x0d, 0x3a, 0xf4, 0x77, 0x37, 0xea, 0x0e, 0xed, 0x9c, 0x5f, 0x6c, 0x33,
	0x58, 0x2b, 0x59, 0x1c, 0xdd, 0x8c, 0xad, 0xb5, 0x1c, 0xdb, 0x26, 0xf8, 0xa7, 0x4c, 0x87, 0xa9,
	0xc8, 0x44, 0x5a, 0xa4, 0xe8, 0xb7, 0x17, 0xc0, 0x29, 0xd3, 0x1f, 0x5c, 0xc5, 0x9e, 0xbe, 0xaa,
	0x69, 0x8d, 0x36, 0x83, 0x6a, 0x48, 0xf6, 0x60, 0x88, 0xdb, 0xc9, 0x17, 0x6e, 0x61, 0x9a, 0x76,
	0xff, 0xb5, 0xb2, 0x41, 0x09, 0xe3, 0x48, 0x6f, 0xbf, 0x83, 0xc1, 0x5f, 0x67, 0xde, 0xfe, 0xd1,
	0x39, 0x57, 0xda, 0xae, 0xd3, 0x46, 0xd2, 0x0e, 0xaa, 0xa1, 0x75, 0xc1, 0xbf, 0xe5, 0x09, 0xcb,
	0x58, 0x7d, 0xf3, 0x6b, 0xa5, 0xed, 0x27, 0xd0, 0x39, 0xca, 0x6d, 0xf8, 0x64, 0x0b, 0x1a, 0x32,
	0xa7, 0x1e, 0x5e, 0xcb, 0xf5, 0xa5, 0x6b, 0x19, 0x34, 0x64, 0x7e, 0x70, 0x06, 0x54, 0xaa, 0xb8,
	0xde, 0xbb, 0x7a, 0x71, 0x0e, 0x7c, 0xc4, 0x8e, 0xed, 0x8b, 0xa3, 0x8f, 0xbd, 0xaf, 0xaf, 0x63,
	0x61, 0x4e, 0x8b, 0x93, 0xe9, 0x5c, 0xa6, 0xb3, 0xda, 0x33, 0xb5, 0xfa, 0x33, 0x96, 0x37, 0xde,
	0xaf, 0xdf, 0x9e, 0x77, 0xd2, 0xc1, 0xd7, 0xeb, 0xd9, 0x9f, 0x00, 0x00, 0x00, 0xff, 0xff, 0x96,
	0xbd, 0x02, 0xd3, 0x46, 0x05, 0x00, 0x00,
}
