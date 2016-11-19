package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gonum/matrix/mat64"
)

func main() {
	//TestHeatKernel()
	xs := []float64{0, 1}
	ys := []float64{1, 2}

	n := 100
	elem := NewElementSimple(xs)
	for i, n := range elem.Nodes {
		n.Val = ys[i]
	}
	//elem.PrintShapeFuncs(os.Stdout, n)
	elem.PrintFunc(os.Stdout, n)

	//xs = []float64{0, 1, 2, 3, 4, 5, 6}
	//mesh, err := NewMeshSimple(xs, 3)
	//if err != nil {
	//	log.Fatal(err)
	//}

	//for i, elem := range mesh.Elems {
	//	fmt.Printf("elem %v\n", i)
	//	for _, n := range elem.Nodes {
	//		fmt.Printf("    node %p at x=%v\n", n, n.X())
	//	}
	//}

	//PrintStiffness(xs, []float64{7, 8, 9, 11, 13, 19}, 3)
	//PrintStiffness([]float64{0, 1, 2}, []float64{7, 8}, 2)
}

func TestHeatKernel() {
	degree := 3
	hc := &HeatConduction{
		X: []float64{0, 1, 2, 3, 4},
		K: ConstVal(2), // W/(m*C)
		S: ConstVal(5), // W/m
		// Area is the cross section area of the conduction medium
		Area:  0.1,            // m^2
		Left:  EssentialBC(0), // deg C
		Right: DirichletBC(5), // W/m^2
	}
	mesh, err := NewMeshSimple(hc.X, degree)
	if err != nil {
		log.Fatal(err)
	}
	stiffness := mesh.StiffnessMatrix(hc)
	fmt.Printf("stiffness:\n%v\n", mat64.Formatted(stiffness))
	force := mesh.ForceMatrix(hc)
	fmt.Printf("force:\n%v\n", mat64.Formatted(force))

	err = mesh.Solve(hc)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Solution:")
	n := 100
	for i := 0; i < n+1; i++ {
		x1 := hc.X[0]
		x2 := hc.X[len(hc.X)-1]
		x := float64(i)/float64(n)*(x2-x1) + x1
		y := mesh.Interpolate(x)
		fmt.Printf("%v\t%v\n", x, y)
	}
}

func PrintStiffness(xs, ks []float64, degree int) {
	k := &SpringKernel{X: xs, K: ks}
	mesh, err := NewMeshSimple(xs, degree)
	if err != nil {
		log.Fatal(err)
	}
	stiffness := mesh.StiffnessMatrix(k)
	fmt.Printf("%v\n", mat64.Formatted(stiffness))
}
