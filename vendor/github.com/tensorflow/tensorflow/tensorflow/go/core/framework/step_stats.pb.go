// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/step_stats.proto

package framework

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// An allocation/de-allocation operation performed by the allocator.
type AllocationRecord struct {
	// The timestamp of the operation.
	AllocMicros int64 `protobuf:"varint,1,opt,name=alloc_micros,json=allocMicros,proto3" json:"alloc_micros,omitempty"`
	// Number of bytes allocated, or de-allocated if negative.
	AllocBytes int64 `protobuf:"varint,2,opt,name=alloc_bytes,json=allocBytes,proto3" json:"alloc_bytes,omitempty"`
}

func (m *AllocationRecord) Reset()                    { *m = AllocationRecord{} }
func (m *AllocationRecord) String() string            { return proto.CompactTextString(m) }
func (*AllocationRecord) ProtoMessage()               {}
func (*AllocationRecord) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{0} }

func (m *AllocationRecord) GetAllocMicros() int64 {
	if m != nil {
		return m.AllocMicros
	}
	return 0
}

func (m *AllocationRecord) GetAllocBytes() int64 {
	if m != nil {
		return m.AllocBytes
	}
	return 0
}

type AllocatorMemoryUsed struct {
	AllocatorName string `protobuf:"bytes,1,opt,name=allocator_name,json=allocatorName,proto3" json:"allocator_name,omitempty"`
	// These are per-node allocator memory stats.
	TotalBytes int64 `protobuf:"varint,2,opt,name=total_bytes,json=totalBytes,proto3" json:"total_bytes,omitempty"`
	PeakBytes  int64 `protobuf:"varint,3,opt,name=peak_bytes,json=peakBytes,proto3" json:"peak_bytes,omitempty"`
	// The bytes that are not deallocated.
	LiveBytes int64 `protobuf:"varint,4,opt,name=live_bytes,json=liveBytes,proto3" json:"live_bytes,omitempty"`
	// The allocation and deallocation timeline.
	AllocationRecords []*AllocationRecord `protobuf:"bytes,6,rep,name=allocation_records,json=allocationRecords" json:"allocation_records,omitempty"`
	// These are snapshots of the overall allocator memory stats.
	// The number of live bytes currently allocated by the allocator.
	AllocatorBytesInUse int64 `protobuf:"varint,5,opt,name=allocator_bytes_in_use,json=allocatorBytesInUse,proto3" json:"allocator_bytes_in_use,omitempty"`
}

func (m *AllocatorMemoryUsed) Reset()                    { *m = AllocatorMemoryUsed{} }
func (m *AllocatorMemoryUsed) String() string            { return proto.CompactTextString(m) }
func (*AllocatorMemoryUsed) ProtoMessage()               {}
func (*AllocatorMemoryUsed) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{1} }

func (m *AllocatorMemoryUsed) GetAllocatorName() string {
	if m != nil {
		return m.AllocatorName
	}
	return ""
}

func (m *AllocatorMemoryUsed) GetTotalBytes() int64 {
	if m != nil {
		return m.TotalBytes
	}
	return 0
}

func (m *AllocatorMemoryUsed) GetPeakBytes() int64 {
	if m != nil {
		return m.PeakBytes
	}
	return 0
}

func (m *AllocatorMemoryUsed) GetLiveBytes() int64 {
	if m != nil {
		return m.LiveBytes
	}
	return 0
}

func (m *AllocatorMemoryUsed) GetAllocationRecords() []*AllocationRecord {
	if m != nil {
		return m.AllocationRecords
	}
	return nil
}

func (m *AllocatorMemoryUsed) GetAllocatorBytesInUse() int64 {
	if m != nil {
		return m.AllocatorBytesInUse
	}
	return 0
}

// Output sizes recorded for a single execution of a graph node.
type NodeOutput struct {
	Slot              int32              `protobuf:"varint,1,opt,name=slot,proto3" json:"slot,omitempty"`
	TensorDescription *TensorDescription `protobuf:"bytes,3,opt,name=tensor_description,json=tensorDescription" json:"tensor_description,omitempty"`
}

func (m *NodeOutput) Reset()                    { *m = NodeOutput{} }
func (m *NodeOutput) String() string            { return proto.CompactTextString(m) }
func (*NodeOutput) ProtoMessage()               {}
func (*NodeOutput) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{2} }

func (m *NodeOutput) GetSlot() int32 {
	if m != nil {
		return m.Slot
	}
	return 0
}

func (m *NodeOutput) GetTensorDescription() *TensorDescription {
	if m != nil {
		return m.TensorDescription
	}
	return nil
}

// For memory tracking.
type MemoryStats struct {
	TempMemorySize                 int64   `protobuf:"varint,1,opt,name=temp_memory_size,json=tempMemorySize,proto3" json:"temp_memory_size,omitempty"`
	PersistentMemorySize           int64   `protobuf:"varint,3,opt,name=persistent_memory_size,json=persistentMemorySize,proto3" json:"persistent_memory_size,omitempty"`
	PersistentTensorAllocIds       []int64 `protobuf:"varint,5,rep,packed,name=persistent_tensor_alloc_ids,json=persistentTensorAllocIds" json:"persistent_tensor_alloc_ids,omitempty"`
	DeviceTempMemorySize           int64   `protobuf:"varint,2,opt,name=device_temp_memory_size,json=deviceTempMemorySize,proto3" json:"device_temp_memory_size,omitempty"`
	DevicePersistentMemorySize     int64   `protobuf:"varint,4,opt,name=device_persistent_memory_size,json=devicePersistentMemorySize,proto3" json:"device_persistent_memory_size,omitempty"`
	DevicePersistentTensorAllocIds []int64 `protobuf:"varint,6,rep,packed,name=device_persistent_tensor_alloc_ids,json=devicePersistentTensorAllocIds" json:"device_persistent_tensor_alloc_ids,omitempty"`
}

func (m *MemoryStats) Reset()                    { *m = MemoryStats{} }
func (m *MemoryStats) String() string            { return proto.CompactTextString(m) }
func (*MemoryStats) ProtoMessage()               {}
func (*MemoryStats) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{3} }

func (m *MemoryStats) GetTempMemorySize() int64 {
	if m != nil {
		return m.TempMemorySize
	}
	return 0
}

func (m *MemoryStats) GetPersistentMemorySize() int64 {
	if m != nil {
		return m.PersistentMemorySize
	}
	return 0
}

func (m *MemoryStats) GetPersistentTensorAllocIds() []int64 {
	if m != nil {
		return m.PersistentTensorAllocIds
	}
	return nil
}

func (m *MemoryStats) GetDeviceTempMemorySize() int64 {
	if m != nil {
		return m.DeviceTempMemorySize
	}
	return 0
}

func (m *MemoryStats) GetDevicePersistentMemorySize() int64 {
	if m != nil {
		return m.DevicePersistentMemorySize
	}
	return 0
}

func (m *MemoryStats) GetDevicePersistentTensorAllocIds() []int64 {
	if m != nil {
		return m.DevicePersistentTensorAllocIds
	}
	return nil
}

// Time/size stats recorded for a single execution of a graph node.
type NodeExecStats struct {
	// TODO(tucker): Use some more compact form of node identity than
	// the full string name.  Either all processes should agree on a
	// global id (cost_id?) for each node, or we should use a hash of
	// the name.
	NodeName         string                   `protobuf:"bytes,1,opt,name=node_name,json=nodeName,proto3" json:"node_name,omitempty"`
	AllStartMicros   int64                    `protobuf:"varint,2,opt,name=all_start_micros,json=allStartMicros,proto3" json:"all_start_micros,omitempty"`
	OpStartRelMicros int64                    `protobuf:"varint,3,opt,name=op_start_rel_micros,json=opStartRelMicros,proto3" json:"op_start_rel_micros,omitempty"`
	OpEndRelMicros   int64                    `protobuf:"varint,4,opt,name=op_end_rel_micros,json=opEndRelMicros,proto3" json:"op_end_rel_micros,omitempty"`
	AllEndRelMicros  int64                    `protobuf:"varint,5,opt,name=all_end_rel_micros,json=allEndRelMicros,proto3" json:"all_end_rel_micros,omitempty"`
	Memory           []*AllocatorMemoryUsed   `protobuf:"bytes,6,rep,name=memory" json:"memory,omitempty"`
	Output           []*NodeOutput            `protobuf:"bytes,7,rep,name=output" json:"output,omitempty"`
	TimelineLabel    string                   `protobuf:"bytes,8,opt,name=timeline_label,json=timelineLabel,proto3" json:"timeline_label,omitempty"`
	ScheduledMicros  int64                    `protobuf:"varint,9,opt,name=scheduled_micros,json=scheduledMicros,proto3" json:"scheduled_micros,omitempty"`
	ThreadId         uint32                   `protobuf:"varint,10,opt,name=thread_id,json=threadId,proto3" json:"thread_id,omitempty"`
	ReferencedTensor []*AllocationDescription `protobuf:"bytes,11,rep,name=referenced_tensor,json=referencedTensor" json:"referenced_tensor,omitempty"`
	MemoryStats      *MemoryStats             `protobuf:"bytes,12,opt,name=memory_stats,json=memoryStats" json:"memory_stats,omitempty"`
}

func (m *NodeExecStats) Reset()                    { *m = NodeExecStats{} }
func (m *NodeExecStats) String() string            { return proto.CompactTextString(m) }
func (*NodeExecStats) ProtoMessage()               {}
func (*NodeExecStats) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{4} }

func (m *NodeExecStats) GetNodeName() string {
	if m != nil {
		return m.NodeName
	}
	return ""
}

func (m *NodeExecStats) GetAllStartMicros() int64 {
	if m != nil {
		return m.AllStartMicros
	}
	return 0
}

func (m *NodeExecStats) GetOpStartRelMicros() int64 {
	if m != nil {
		return m.OpStartRelMicros
	}
	return 0
}

func (m *NodeExecStats) GetOpEndRelMicros() int64 {
	if m != nil {
		return m.OpEndRelMicros
	}
	return 0
}

func (m *NodeExecStats) GetAllEndRelMicros() int64 {
	if m != nil {
		return m.AllEndRelMicros
	}
	return 0
}

func (m *NodeExecStats) GetMemory() []*AllocatorMemoryUsed {
	if m != nil {
		return m.Memory
	}
	return nil
}

func (m *NodeExecStats) GetOutput() []*NodeOutput {
	if m != nil {
		return m.Output
	}
	return nil
}

func (m *NodeExecStats) GetTimelineLabel() string {
	if m != nil {
		return m.TimelineLabel
	}
	return ""
}

func (m *NodeExecStats) GetScheduledMicros() int64 {
	if m != nil {
		return m.ScheduledMicros
	}
	return 0
}

func (m *NodeExecStats) GetThreadId() uint32 {
	if m != nil {
		return m.ThreadId
	}
	return 0
}

func (m *NodeExecStats) GetReferencedTensor() []*AllocationDescription {
	if m != nil {
		return m.ReferencedTensor
	}
	return nil
}

func (m *NodeExecStats) GetMemoryStats() *MemoryStats {
	if m != nil {
		return m.MemoryStats
	}
	return nil
}

type DeviceStepStats struct {
	Device    string           `protobuf:"bytes,1,opt,name=device,proto3" json:"device,omitempty"`
	NodeStats []*NodeExecStats `protobuf:"bytes,2,rep,name=node_stats,json=nodeStats" json:"node_stats,omitempty"`
}

func (m *DeviceStepStats) Reset()                    { *m = DeviceStepStats{} }
func (m *DeviceStepStats) String() string            { return proto.CompactTextString(m) }
func (*DeviceStepStats) ProtoMessage()               {}
func (*DeviceStepStats) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{5} }

func (m *DeviceStepStats) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *DeviceStepStats) GetNodeStats() []*NodeExecStats {
	if m != nil {
		return m.NodeStats
	}
	return nil
}

type StepStats struct {
	DevStats []*DeviceStepStats `protobuf:"bytes,1,rep,name=dev_stats,json=devStats" json:"dev_stats,omitempty"`
}

func (m *StepStats) Reset()                    { *m = StepStats{} }
func (m *StepStats) String() string            { return proto.CompactTextString(m) }
func (*StepStats) ProtoMessage()               {}
func (*StepStats) Descriptor() ([]byte, []int) { return fileDescriptorStepStats, []int{6} }

func (m *StepStats) GetDevStats() []*DeviceStepStats {
	if m != nil {
		return m.DevStats
	}
	return nil
}

func init() {
	proto.RegisterType((*AllocationRecord)(nil), "tensorflow.AllocationRecord")
	proto.RegisterType((*AllocatorMemoryUsed)(nil), "tensorflow.AllocatorMemoryUsed")
	proto.RegisterType((*NodeOutput)(nil), "tensorflow.NodeOutput")
	proto.RegisterType((*MemoryStats)(nil), "tensorflow.MemoryStats")
	proto.RegisterType((*NodeExecStats)(nil), "tensorflow.NodeExecStats")
	proto.RegisterType((*DeviceStepStats)(nil), "tensorflow.DeviceStepStats")
	proto.RegisterType((*StepStats)(nil), "tensorflow.StepStats")
}

func init() { proto.RegisterFile("tensorflow/core/framework/step_stats.proto", fileDescriptorStepStats) }

var fileDescriptorStepStats = []byte{
	// 826 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x55, 0xdf, 0x6f, 0xdb, 0x36,
	0x10, 0x86, 0xec, 0xc4, 0x8b, 0xcf, 0xf9, 0xe1, 0x30, 0x45, 0xaa, 0x25, 0xcb, 0x9a, 0x1a, 0x18,
	0xe0, 0x6e, 0x98, 0x03, 0xa4, 0xc3, 0xd6, 0x0d, 0xe8, 0xc3, 0x82, 0xe6, 0x21, 0x58, 0x9b, 0x05,
	0x4a, 0xbb, 0x87, 0xbd, 0x08, 0x8a, 0x78, 0x49, 0x84, 0x52, 0xa2, 0x40, 0xd2, 0xee, 0x9a, 0xff,
	0x65, 0x7f, 0xcc, 0xfe, 0xab, 0x3d, 0x0d, 0x03, 0x8f, 0xb4, 0x24, 0xcb, 0xce, 0x9b, 0xf4, 0xdd,
	0x77, 0x1f, 0x8f, 0xc7, 0xef, 0x48, 0xf8, 0xd6, 0x60, 0xa1, 0xa5, 0xba, 0x15, 0xf2, 0xd3, 0x49,
	0x2a, 0x15, 0x9e, 0xdc, 0xaa, 0x24, 0xc7, 0x4f, 0x52, 0x7d, 0x3c, 0xd1, 0x06, 0xcb, 0x58, 0x9b,
	0xc4, 0xe8, 0x49, 0xa9, 0xa4, 0x91, 0x0c, 0x6a, 0xee, 0xc1, 0x8f, 0x8f, 0xe7, 0x25, 0x42, 0xc8,
	0x34, 0x31, 0x99, 0x2c, 0x62, 0x8e, 0x3a, 0x55, 0x59, 0x69, 0xbf, 0x9d, 0xc6, 0xc1, 0xe9, 0xe3,
	0x79, 0x2e, 0xb2, 0x9c, 0x33, 0xfa, 0x03, 0x86, 0xbf, 0x56, 0x9a, 0x11, 0xa6, 0x52, 0x71, 0xf6,
	0x1c, 0x36, 0x69, 0x9d, 0x38, 0xcf, 0x52, 0x25, 0x75, 0x18, 0x1c, 0x07, 0xe3, 0x6e, 0x34, 0x20,
	0xec, 0x1d, 0x41, 0xec, 0x19, 0xb8, 0xdf, 0xf8, 0xe6, 0xb3, 0x41, 0x1d, 0x76, 0x88, 0x01, 0x04,
	0x9d, 0x59, 0x64, 0xf4, 0x77, 0x07, 0xf6, 0xbc, 0xb0, 0x54, 0xef, 0x30, 0x97, 0xea, 0xf3, 0x07,
	0x8d, 0x9c, 0x7d, 0x03, 0xdb, 0xc9, 0x1c, 0x8e, 0x8b, 0x24, 0x47, 0x52, 0xef, 0x47, 0x5b, 0x15,
	0x7a, 0x99, 0xe4, 0x68, 0xf5, 0x8d, 0x34, 0x89, 0x58, 0xd4, 0x27, 0x88, 0xf4, 0xd9, 0x11, 0x40,
	0x89, 0xc9, 0x47, 0x1f, 0xef, 0x52, 0xbc, 0x6f, 0x91, 0x2a, 0x2c, 0xb2, 0x19, 0xfa, 0xf0, 0x9a,
	0x0b, 0x5b, 0xc4, 0x85, 0x7f, 0x03, 0xd6, 0xe8, 0xa4, 0xa2, 0x6d, 0xeb, 0xb0, 0x77, 0xdc, 0x1d,
	0x0f, 0x4e, 0xbf, 0x9a, 0xd4, 0x6d, 0x9c, 0xb4, 0x7b, 0x13, 0xed, 0x26, 0x2d, 0x44, 0xb3, 0x97,
	0xb0, 0x5f, 0x6f, 0x89, 0x16, 0x8c, 0xb3, 0x22, 0x9e, 0x6a, 0x0c, 0xd7, 0x69, 0xdd, 0xbd, 0x2a,
	0x4a, 0x8b, 0x5f, 0x14, 0x1f, 0x34, 0x8e, 0x0a, 0x80, 0x4b, 0xc9, 0xf1, 0xf7, 0xa9, 0x29, 0xa7,
	0x86, 0x31, 0x58, 0xd3, 0x42, 0x1a, 0xea, 0xc5, 0x7a, 0x44, 0xdf, 0xec, 0x2d, 0xb0, 0xe5, 0x53,
	0xa3, 0x9d, 0x0e, 0x4e, 0x8f, 0x9a, 0x35, 0xbe, 0xa7, 0xcf, 0x37, 0x35, 0x29, 0xda, 0x35, 0x6d,
	0x68, 0xf4, 0x5f, 0x07, 0x06, 0xee, 0x18, 0xae, 0xad, 0xeb, 0xd8, 0x18, 0x86, 0x06, 0xf3, 0x32,
	0xce, 0x09, 0x8b, 0x75, 0xf6, 0x80, 0xfe, 0x9c, 0xb7, 0x2d, 0xee, 0xa9, 0xd9, 0x03, 0xb2, 0x1f,
	0x60, 0xbf, 0x44, 0xa5, 0x33, 0x6d, 0xb0, 0x30, 0x0b, 0x7c, 0xd7, 0xf5, 0x27, 0x75, 0xb4, 0x91,
	0xf5, 0x1a, 0x0e, 0x1b, 0x59, 0x7e, 0x23, 0xce, 0x32, 0x19, 0xd7, 0xe1, 0xfa, 0x71, 0x77, 0xdc,
	0x8d, 0xc2, 0x9a, 0xe2, 0x36, 0x41, 0xed, 0xbe, 0xe0, 0x9a, 0xfd, 0x0c, 0x4f, 0x39, 0xce, 0xb2,
	0x14, 0xe3, 0xa5, 0x2a, 0xc9, 0x0b, 0x67, 0x9d, 0x30, 0x88, 0x9e, 0x38, 0xca, 0xfb, 0xc5, 0x7a,
	0xcf, 0xe1, 0xc8, 0xa7, 0x3e, 0x52, 0xf6, 0x5a, 0x25, 0x70, 0xe0, 0x88, 0x57, 0xab, 0x36, 0x70,
	0x09, 0xa3, 0x65, 0x99, 0xa5, 0x7d, 0x58, 0xcb, 0x38, 0xad, 0xaf, 0xdb, 0x5a, 0x8b, 0x3b, 0x1a,
	0xfd, 0xb3, 0x06, 0x5b, 0xf6, 0xc4, 0xcf, 0xff, 0xc2, 0xd4, 0x1d, 0xc1, 0x21, 0xf4, 0x0b, 0xc9,
	0xb1, 0x39, 0x05, 0x1b, 0x16, 0xa0, 0x01, 0x18, 0xc3, 0x30, 0x11, 0xc2, 0x5e, 0x11, 0xca, 0xcc,
	0xe7, 0xd0, 0x4d, 0x81, 0x9d, 0x9f, 0x6b, 0x0b, 0xfb, 0x51, 0xfc, 0x1e, 0xf6, 0x64, 0xe9, 0x89,
	0x0a, 0xc5, 0x9c, 0xec, 0x0e, 0x67, 0x28, 0x4b, 0xe2, 0x46, 0x28, 0x3c, 0xfd, 0x05, 0xec, 0xca,
	0x32, 0xc6, 0x82, 0x37, 0xc9, 0x6e, 0x40, 0xb6, 0x65, 0x79, 0x5e, 0xf0, 0x9a, 0xfa, 0x1d, 0x4d,
	0x49, 0x9b, 0xeb, 0x4c, 0xbd, 0x93, 0x08, 0xb1, 0x40, 0xfe, 0x09, 0x7a, 0xae, 0xc9, 0x7e, 0x8c,
	0x9e, 0xad, 0x18, 0xa3, 0xe6, 0x4d, 0x10, 0x79, 0x3a, 0x9b, 0x40, 0x4f, 0xd2, 0x14, 0x84, 0x5f,
	0x50, 0xe2, 0x7e, 0x33, 0xb1, 0x9e, 0x91, 0xc8, 0xb3, 0xec, 0x0d, 0x62, 0xb2, 0x1c, 0x45, 0x56,
	0x60, 0x2c, 0x92, 0x1b, 0x14, 0xe1, 0x86, 0xbb, 0x41, 0xe6, 0xe8, 0x5b, 0x0b, 0xb2, 0x17, 0x30,
	0xd4, 0xe9, 0x3d, 0xf2, 0xa9, 0x40, 0x3e, 0x2f, 0xbd, 0xef, 0x4a, 0xaf, 0x70, 0x5f, 0xfa, 0x21,
	0xf4, 0xcd, 0xbd, 0xc2, 0x84, 0xc7, 0x19, 0x0f, 0xe1, 0x38, 0x18, 0x6f, 0x45, 0x1b, 0x0e, 0xb8,
	0xe0, 0xec, 0x12, 0x76, 0x15, 0xde, 0xa2, 0xc2, 0x22, 0x45, 0xee, 0x0d, 0x10, 0x0e, 0xa8, 0xd2,
	0xe7, 0xab, 0x6f, 0x8a, 0xe6, 0x24, 0x0e, 0xeb, 0x5c, 0xe7, 0x07, 0xf6, 0x0b, 0x6c, 0xce, 0xcd,
	0x68, 0x5d, 0x10, 0x6e, 0xd2, 0x40, 0x3f, 0x6d, 0x4a, 0x35, 0xe6, 0x34, 0x1a, 0xe4, 0xf5, 0xcf,
	0x28, 0x85, 0x9d, 0x37, 0xe4, 0xb2, 0x6b, 0x83, 0xa5, 0x33, 0xd1, 0x3e, 0xf4, 0x9c, 0xf1, 0xbc,
	0x83, 0xfc, 0x1f, 0x7b, 0x05, 0x40, 0xe6, 0x72, 0x8b, 0x74, 0xa8, 0xde, 0x2f, 0xdb, 0x9d, 0xad,
	0xbc, 0x18, 0x91, 0x13, 0xdd, 0x22, 0xe7, 0xd0, 0xaf, 0xe5, 0x5f, 0x41, 0x9f, 0xe3, 0xcc, 0xab,
	0x04, 0xa4, 0x72, 0xd8, 0x54, 0x69, 0x95, 0x13, 0x6d, 0x70, 0x9c, 0xd1, 0xd7, 0x99, 0x84, 0x50,
	0xaa, 0xbb, 0x26, 0xb7, 0x7a, 0x8d, 0xce, 0x76, 0xaa, 0x84, 0x2b, 0xfb, 0x08, 0xe9, 0xab, 0xe0,
	0xcf, 0xd7, 0x77, 0x99, 0xb9, 0x9f, 0xde, 0x4c, 0x52, 0x99, 0x9f, 0x34, 0x9e, 0xb1, 0xd5, 0x9f,
	0x77, 0xb2, 0xf5, 0xbe, 0xfd, 0x1b, 0x04, 0x37, 0x3d, 0x7a, 0xd0, 0x5e, 0xfe, 0x1f, 0x00, 0x00,
	0xff, 0xff, 0xeb, 0x46, 0x27, 0x74, 0x76, 0x07, 0x00, 0x00,
}
