package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"

	"github.com/gonum/matrix/mat64"
	"github.com/rwcarlsen/fem/sparse"
)

var simplediffusion = flag.Bool("simplediffusion", false, "run 2d simple diffusion instead of other problem")
var printmats = flag.Bool("print", false, "print stiffness and force matrices")
var ndivs = flag.Int("n", 5, "number of divisions per dimension in the structured mesh")
var order = flag.Int("order", 1, "lagrange shape function order")
var iter = flag.Int("iter", 1000, "number of iterations for solve (default=direct)")
var usertol = flag.Float64("tol", 1e-5, "l2 norm consecutive iterative soln diff threshold")
var nsoln = flag.Int("nsol", 4, "number of uniformly distributed points to sample+print solution over")
var solver = flag.String("solver", "cg", "solver type (gaussian, denselu, cg)")
var pc = flag.String("pc", "ilu", "preconditioner type (ilu, jacobi, none)")
var dim = flag.Int("dim", 1, "dimesionality of sample problem - either 1 or 2")

var plot = flag.String("plot", "", "'svg' to create svg plot with gnuplot")

var cpuprofile = flag.String("cpuprofile", "", "profile file name")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

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

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	if *simplediffusion {
		TestSimpleDiffusion()
	} else if *dim == 1 {
		TestHeatKernel()
	} else if *dim == 2 {
		TestHeatKernel2D()
	} else {
		TestHeatKernelND(*dim)
	}

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal("could not create memory profile: ", err)
		}
		runtime.GC() // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
		f.Close()
	}

}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func TestHeatKernel() {
	xs := []float64{}
	for i := 0; i < *ndivs; i++ {
		xs = append(xs, float64(i)/float64(*ndivs-1)*4)
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
	mesh, err := NewMeshStructured(*order, xs)
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
	for i := 0; i < *ndivs; i++ {
		xs = append(xs, float64(i)/float64(*ndivs-1)*4)
		ys = append(ys, float64(i)/float64(*ndivs-1)*4)
	}
	mesh, err := NewMeshStructured(*order, xs, ys)
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

func TestHeatKernelND(ndim int) {
	// build mesh
	divs := make([][]float64, ndim)

	low := make([]float64, ndim)
	up := make([]float64, ndim)
	lowTypes := make([]BoundaryType, ndim)
	upTypes := make([]BoundaryType, ndim)
	lowVals := make([]float64, ndim)
	upVals := make([]float64, ndim)
	for d := range divs {
		lowTypes[d] = Dirichlet
		upTypes[d] = Dirichlet
		lowVals[d] = 0
		upVals[d] = 0

		low[d] = 0
		up[d] = 4

		divs[d] = make([]float64, *ndivs)
		for i := range divs[d] {
			divs[d][i] = float64(i) / float64(*ndivs-1) * 4
		}
	}
	mesh, err := NewMeshStructured(*order, divs...)
	check(err)

	// build kernel and boundary conditions
	boundary := &StructuredBoundary{
		Tol:      1e-6,
		Low:      low,
		Up:       up,
		LowTypes: lowTypes,
		UpTypes:  upTypes,
		LowVals:  lowVals,
		UpVals:   upVals,
	}

	hc := &HeatConduction{
		K:        ConstVal(2),  // thermal conductivity W/(m*C)
		S:        ConstVal(50), // volumetric source W/m^3
		Boundary: boundary,
	}

	solveProb(mesh, hc)

	var buf bytes.Buffer
	printSolution(&buf, mesh, low, up)

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
		dims[i] = *nsoln + 1
	}

	perms := Permute(nil, nil, dims...)
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

	var precon sparse.Preconditioner
	switch *pc {
	case "ilu":
		precon = sparse.IncompleteLU
	case "jacobi":
		precon = sparse.Jacobi
	case "blocklu":
		blocksize := 50
		precon = sparse.BlockLU(blocksize)
	case "none":
	default:
		log.Fatalf("invalid preconditioner type")
	}

	switch *solver {
	case "gaussian":
		mesh.Solver = sparse.GaussJordan{}
	case "cg":
		cg := &sparse.CG{MaxIter: *iter, Tol: *usertol}
		cg.Preconditioner = precon
		mesh.Solver = cg
	case "denselu":
		mesh.Solver = sparse.DenseLU{}
	default:
		log.Fatalf("unrecognized solver %v", *solver)
	}

	err := mesh.Solve(k)
	check(err)
	log.Print(mesh.Solver.Status())
}

func TestSimpleDiffusion() {
	low := []float64{0, 0}
	up := []float64{1, 1}

	xs := []float64{}
	ys := []float64{}
	for i := 0; i < *ndivs; i++ {
		xs = append(xs, float64(i)/float64(*ndivs-1)*1)
		ys = append(ys, float64(i)/float64(*ndivs-1)*1)
	}

	mesh, err := NewMeshStructured(*order, xs, ys)
	check(err)

	// build kernel and boundary conditions
	boundary := &StructuredBoundary{
		Tol:      1e-6,
		Low:      []float64{0, 0},
		Up:       []float64{1, 1},
		LowTypes: []BoundaryType{Dirichlet, Neumann},
		UpTypes:  []BoundaryType{Dirichlet, Neumann},
		LowVals:  []float64{0, 0},
		UpVals:   []float64{15, 0},
	}

	hc := &HeatConduction{
		K:        ConstVal(2), // thermal conductivity W/(m*C)
		S:        ConstVal(0), // volumetric source W/m^3
		Boundary: boundary,
	}

	solveProb(mesh, hc)

	var buf bytes.Buffer
	printSolution(&buf, mesh, low, up)

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
