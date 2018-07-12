package main

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	tensorflow_config "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf"
	
	"github.com/gogo/protobuf/proto"

	"io/ioutil"
	"log"
	"os"
)

func create_graph(model_path string) (*tf.Graph) {
	model, err := ioutil.ReadFile(model_path)
	if err != nil {
		log.Fatalf("could not read model from %s: %v", model_path, err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatalf("could not load model from %s: %v", model_path, err)
	}

	return graph
}

func session_config() *tf.SessionOptions {
       config := &tensorflow_config.ConfigProto {
	       LogDevicePlacement: true,
	       AllowSoftPlacement: false,
	       GpuOptions: &tensorflow_config.GPUOptions {
	       },
       }
       ser, err := proto.Marshal(config)
       if err != nil {
               log.Fatalf("could not serialize config: %v", err)
       }

       return &tf.SessionOptions {
               Config: ser,
       }
}

func new_session(graph *tf.Graph) *tf.Session {
	so := session_config()
	sess, err := tf.NewSession(graph, so)
	if err != nil {
		log.Fatalf("could not create session: %v", err)
	}
	return sess
}

func main() {
	args := os.Args[1:]
	if len(args) != 1 {
		log.Fatalf("path to serialized graph must be provioded")
	}

	graph := create_graph(args[0])

	session := new_session(graph)

	i0, err := tf.NewTensor([][]int32{{1, 2, 3}})
	if err != nil {
		log.Fatal(err)
	}
	i1, err := tf.NewTensor([][]int32{{3, 4, 5}})
	if err != nil {
		log.Fatal(err)
	}

	res, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input/ph0").Output(0): i0,
			graph.Operation("input/ph1").Output(0): i1,
		},

		[]tf.Output{
			graph.Operation("output/op").Output(0),
		},

		nil)

	if err != nil {
		log.Fatalf("could not run face detection inference: %v", err)
	}

	fmt.Printf("op: %v\n", res[0].Value())
}
