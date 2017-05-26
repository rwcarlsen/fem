package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/fem/sparse"
)

var printmats = flag.Bool("print", false, "print stiffness and force matrices")
var nnodes = flag.Int("nodes", 5, "number of nodes/domain divisions-1")
var order = flag.Int("order", 2, "lagrange shape function order")
var iter = flag.Int("iter", 1000, "number of iterations for solve (default=direct)")
var usertol = flag.Float64("tol", 1e-5, "l2 norm consecutive iterative soln diff threshold")
var nsoln = flag.Int("nsol", 10, "number of uniformly distributed points to sample+print solution over")
var solver = flag.String("solver", "gaussian", "solver type (gaussian, denselu, cg)")
var dim = flag.Int("dim", 1, "dimesionality of sample problem - either 1 or 2")

var cpuprofile = flag.String("cpuprofile", "", "profile file name")

func main() {
	log.SetFlags(0)
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if *dim == 1 {
		TestHeatKernel()
	} else {
		TestHeatKernel2D()
	}
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func TestHeatKernel() {
	xs := []float64{}
	for i := 0; i < *nnodes; i++ {
		xs = append(xs, float64(i)/float64(*nnodes-1)*4)
	}
	hc := &HeatConduction{
		K: ConstVal(2),  // W/(m*C)
		S: ConstVal(50), // W/m^3
		Boundary: &Boundary1D{
			Left:      xs[0],
			LeftVal:   0, // deg C
			LeftType:  Dirichlet,
			Right:     xs[len(xs)-1],
			RightVal:  5, // W/m^2
			RightType: Neumann,
		},
	}
	mesh, err := NewMeshSimple1D(xs, *order)
	check(err)

	if *printmats {
		stiffness := mesh.StiffnessMatrix(hc)
		fmt.Printf("stiffness:\n%.5v\n", mat64.Formatted(stiffness))
		force := mesh.ForceVector(hc)
		fmt.Printf("force:\n%.4v\n", force)
	}

	switch *solver {
	case "gaussian":
		mesh.Solver = sparse.GaussJordan{}
	case "cg":
		mesh.Solver = &sparse.CG{MaxIter: *iter, Tol: *usertol}
	case "denselu":
		mesh.Solver = sparse.DenseLU{}
	default:
		log.Fatalf("unrecognized solver %v", *solver)
	}

	err = mesh.Solve(hc)
	check(err)
	log.Print(mesh.Solver.Status())

	fmt.Println("Solution:")
	for i := 0; i < *nsoln+1; i++ {
		x1 := xs[0]
		x2 := xs[len(xs)-1]
		x := []float64{float64(i)/float64(*nsoln)*(x2-x1) + x1}
		y, err := mesh.Interpolate(x)
		check(err)
		fmt.Printf("%v\t%v\n", x[0], y)
	}
}

func TestHeatKernel2D() {
	// build mesh
	xs := []float64{}
	ys := []float64{}
	for i := 0; i < *nnodes; i++ {
		xs = append(xs, float64(i)/float64(*nnodes-1)*4)
		ys = append(ys, float64(i)/float64(*nnodes-1)*4)
	}
	mesh, err := NewMeshSimple2D(xs, ys)
	check(err)

	// build kernel and boundary conditions
	end := len(xs) - 1
	boundary := &Boundary2D{Tol: 1e-6}
	boundary.Append(xs[0], ys[0], Dirichlet, 0)     // bottom at zero deg
	boundary.Append(xs[end], ys[0], Dirichlet, 0)   // right at zero deg
	boundary.Append(xs[end], ys[end], Dirichlet, 0) // top at zero deg
	boundary.Append(xs[0], ys[end], Dirichlet, 0)   // bottom at zero deg

	hc := &HeatConduction{
		K:        ConstVal(2),  // thermal conductivity W/(m*C)
		S:        ConstVal(50), // volumetric source W/m^3
		Boundary: boundary,
	}

	if *printmats {
		stiffness := mesh.StiffnessMatrix(hc)
		fmt.Printf("stiffness:\n%.5v\n", mat64.Formatted(stiffness))
		force := mesh.ForceVector(hc)
		fmt.Printf("force:\n%.4v\n", force)
	}

	switch *solver {
	case "gaussian":
		mesh.Solver = sparse.GaussJordan{}
	case "cg":
		mesh.Solver = &sparse.CG{MaxIter: *iter, Tol: *usertol}
	case "denselu":
		mesh.Solver = sparse.DenseLU{}
	default:
		log.Fatalf("unrecognized solver %v", *solver)
	}

	err = mesh.Solve(hc)
	check(err)
	log.Print(mesh.Solver.Status())

	fmt.Println("Solution:")
	x1 := xs[0]
	x2 := xs[end]
	y1 := ys[0]
	y2 := ys[end]
	for i := 0; i < *nsoln+1; i++ {
		for j := 0; j < *nsoln+1; j++ {
			x := []float64{
				float64(i)/float64(*nsoln)*(x2-x1) + x1,
				float64(j)/float64(*nsoln)*(y2-y1) + y1,
			}
			u, err := mesh.Interpolate(x)
			check(err)
			fmt.Printf("%v\t%v\t%v\n", x[0], x[1], u)
		}
	}
}
