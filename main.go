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

var printmats = flag.Bool("print", false, "print stiffness and force matrices")
var ndivs = flag.Int("ndivs", 5, "number of divisions per dimension in the structured mesh")
var order = flag.Int("order", 1, "lagrange shape function order")
var iter = flag.Int("iter", 1000, "number of iterations for solve (default=direct)")
var usertol = flag.Float64("tol", 1e-5, "l2 norm consecutive iterative soln diff threshold")
var nsoln = flag.Int("nsol", 4, "number of uniformly distributed points to sample+print solution over")
var solver = flag.String("solver", "cg", "solver type (gaussian, denselu, cg)")
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

	if *dim == 1 {
		TestHeatKernel()
	} else if *dim == 2 {
		TestHeatKernel2D()
	} else {
		TestHeatKernel3D()
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
	mesh, err := NewMeshSimple1D(*order, xs)
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
	mesh, err := NewMeshSimple2D(*order, xs, ys)
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

func TestHeatKernel3D() {
	// build mesh
	xs := []float64{}
	ys := []float64{}
	zs := []float64{}
	for i := 0; i < *ndivs; i++ {
		xs = append(xs, float64(i)/float64(*ndivs-1)*4)
		ys = append(ys, float64(i)/float64(*ndivs-1)*4)
		zs = append(zs, float64(i)/float64(*ndivs-1)*4)
	}
	mesh, err := NewMeshSimple3D(*order, xs, ys, zs)
	check(err)

	// build kernel and boundary conditions
	end := len(xs) - 1
	boundary := &StructuredBoundary{
		Tol:      1e-6,
		Low:      []float64{0, 0},
		Up:       []float64{4, 4},
		LowTypes: []BoundaryType{Dirichlet, Dirichlet},
		UpTypes:  []BoundaryType{Dirichlet, Dirichlet},
		LowVals:  []float64{0, 0},
		UpVals:   []float64{0, 0},
	}

	hc := &HeatConduction{
		K:        ConstVal(2),  // thermal conductivity W/(m*C)
		S:        ConstVal(50), // volumetric source W/m^3
		Boundary: boundary,
	}

	solveProb(mesh, hc)

	var buf bytes.Buffer
	minbounds := []float64{xs[0], ys[0], zs[0]}
	maxbounds := []float64{xs[end], ys[end], zs[end]}
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
