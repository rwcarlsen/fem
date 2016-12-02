package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/gonum/matrix/mat64"
)

var printmats = flag.Bool("print", false, "print stiffness and force matrices")
var nnodes = flag.Int("nodes", 5, "number of nodes/domain divisions-1")

func main() {
	flag.Parse()
	TestHeatKernel()
}

func TestHeatKernel() {
	degree := 3
	xs := []float64{}
	for i := 0; i < *nnodes; i++ {
		xs = append(xs, float64(i)/float64(*nnodes-1)*4)
	}
	hc := &HeatConduction{
		X: xs,
		K: ConstVal(2), // W/(m*C)
		S: ConstVal(5), // W/m
		// Area is the cross section area of the conduction medium
		Area:  0.1,            // m^2
		Left:  DirichletBC(0), // deg C
		Right: NeumannBC(5),   // W/m^2
	}
	mesh, err := NewMeshSimple1D(hc.X, degree)
	if err != nil {
		log.Fatal(err)
	}

	if *printmats {
		stiffness := mesh.StiffnessMatrix(hc, DefaultPenalty)
		fmt.Printf("stiffness:\n%v\n", mat64.Formatted(stiffness))
		force := mesh.ForceMatrix(hc, DefaultPenalty)
		fmt.Printf("force:\n%v\n", mat64.Formatted(force))
	}

	err = mesh.Solve(hc)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Solution:")
	n := 100
	for i := 0; i < n+1; i++ {
		x1 := hc.X[0]
		x2 := hc.X[len(hc.X)-1]
		x := []float64{float64(i)/float64(n)*(x2-x1) + x1}
		y, err := mesh.Interpolate(x)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%v\t%v\n", x[0], y)
	}
}
