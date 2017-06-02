package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
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
var plot = flag.String("plot", "", "'svg' to create svg plot with gnuplot")

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

	solveProb(mesh, hc)
	var buf bytes.Buffer
	printSolution(&buf, mesh, []float64{xs[0]}, []float64{xs[len(xs)-1]})

	if *plot == "" {
		log.Print("Solution:")
		fmt.Print(buf.String())
	} else {
		cmd := exec.Command("gnuplot", "-e", `set terminal svg; set output "`+*plot+`"; plot "-" u 1:2 w l`)
		cmd.Stdin = &buf
		err := cmd.Run()
		check(err)
	}
}

func TestHeatKernel2D() {
	// build mesh
	xs := []float64{}
	ys := []float64{}
	nacross := int(math.Sqrt(float64(*nnodes)))
	for i := 0; i < nacross; i++ {
		xs = append(xs, float64(i)/float64(nacross-1)*4)
		ys = append(ys, float64(i)/float64(nacross-1)*4)
	}
	mesh, err := NewMeshSimple2D(xs, ys)
	check(err)

	// build kernel and boundary conditions
	end := len(xs) - 1
	boundary := &Boundary2D{Tol: 1e-6}
	boundary.Append(0, 0, Dirichlet, 0)
	boundary.Append(1.7, 0, Dirichlet, 0)
	boundary.Append(3.7, 0, Dirichlet, 8)
	boundary.Append(4.0, 0, Dirichlet, 40)
	boundary.Append(4.0, 2.9, Dirichlet, 0)
	boundary.Append(xs[end], ys[end], Dirichlet, 0) // top at zero deg
	boundary.Append(xs[0], ys[end], Dirichlet, 0)   // bottom at zero deg

	hc := &HeatConduction{
		K:        ConstVal(2),  // thermal conductivity W/(m*C)
		S:        ConstVal(50), // volumetric source W/m^3
		Boundary: boundary,
	}

	solveProb(mesh, hc)

	var buf bytes.Buffer
	minbounds := []float64{xs[0], ys[0]}
	maxbounds := []float64{xs[end], ys[end]}
	printSolution(&buf, mesh, minbounds, maxbounds)

	if *plot == "" {
		log.Print("Solution:")
		fmt.Print(buf.String())
	} else {
		cmd := exec.Command("gnuplot", "-e", `set terminal svg; set output "`+*plot+`"; plot "-" u 1:2:3 w image`)
		cmd.Stdin = &buf
		err := cmd.Run()
		check(err)
	}
}

func printSolution(w io.Writer, mesh *Mesh, min, max []float64) {
	n := len(min)
	dims := make([]int, n)
	for i := range dims {
		dims[i] = *nsoln
	}

	perms := Permute(nil, dims...)
	for _, p := range perms {
		x := make([]float64, len(p))
		for i, ii := range p {
			x[i] = float64(ii)/float64(*nsoln)*(max[i]-min[i]) + min[i]
		}
		u, err := mesh.Interpolate(x)
		check(err)
		if u < 1e-6 {
			u = 0
		}
		for _, val := range x {
			fmt.Fprintf(w, "%.3v\t", val)
		}
		fmt.Fprintf(w, "%.3v\n", u)
	}
}

func solveProb(mesh *Mesh, k Kernel) {
	if *printmats {
		stiffness := mesh.StiffnessMatrix(k)
		fmt.Printf("stiffness:\n% .3v\n", mat64.Formatted(stiffness))
		force := mesh.ForceVector(k)
		fmt.Printf("force:\n% .3v\n", force)
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

	err := mesh.Solve(k)
	check(err)
	log.Print(mesh.Solver.Status())
}
