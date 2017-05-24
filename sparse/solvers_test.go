package sparse

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func makeSparse(size int, data []float64) *Sparse {
	A := NewSparse(size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			A.Set(i, j, data[i*size+j])
		}
	}
	return A
}

func TestNewCholesky(t *testing.T) {
	size := 3
	data := []float64{
		4, 12, -16,
		12, 37, -43,
		-16, -43, 98,
	}
	wantdata := []float64{
		2, 0, 0,
		6, 1, 0,
		-8, 5, 3,
	}
	tol := 1e-6

	A := makeSparse(size, data)
	wantL := makeSparse(size, wantdata)

	chol := NewCholesky(A)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if math.Abs(chol.L.At(i, j)-wantL.At(i, j)) > tol {
				t.Fatalf("solutions don't match:\ngot\n% .3v\nwant\n% .3v", mat64.Formatted(chol.L), mat64.Formatted(wantL))
			}
		}
	}
}

func TestNewCholesky2(t *testing.T) {
	size := 6
	nfill := 4 // number filled entries per row
	tol := 1e-6

	s := randSparse(size, nfill, 0)
	f := make([]float64, size)
	d := mat64.NewSymDense(size, nil)
	for i := range f {
		f[i] = 1
		for j := range f {
			if j >= i {
				d.SetSym(i, j, s.At(i, j))
			}
		}
	}

	var refchol mat64.Cholesky
	refchol.Factorize(d)
	var wantL mat64.TriDense
	wantL.LFromCholesky(&refchol)

	chol := NewCholesky(s)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if math.Abs(chol.L.At(i, j)-wantL.At(i, j)) > tol {
				t.Fatalf("solutions don't match:\ngot\n% .3v\nwant\n% .3v", mat64.Formatted(chol.L), mat64.Formatted(&wantL))
			}
		}
	}
}

func TestCholesky_Solve(t *testing.T) {
	size := 50
	nfill := 4 // number filled entries per row
	tol := 1e-6

	s := randSparse(size, nfill, 0)
	f := make([]float64, size)
	for i := range f {
		f[i] = 1
	}

	d := mat64.DenseCopyOf(s)
	var want mat64.Vector
	want.SolveVec(d, mat64.NewVector(size, f))

	chol := NewCholesky(s)
	got, _ := chol.Solve(f)
	for i := range got {
		if math.Abs(got[i]-want.At(i, 0)) > tol {
			t.Errorf("A:\n% 3g\nb=%v", mat64.Formatted(s), f)
			t.Errorf("condition number of A is %v", mat64.Cond(s, 2))
			t.Fatalf("solutions don't match:\ngot %v\nwant %v", got, want.RawVector().Data)
		}
	}
}

func TestCGSolve(t *testing.T) {
	size := 50
	nfill := 4 // number filled entries per row
	maxiter := 1000
	tol := 1e-6

	s := randSparse(size, nfill, 0)
	f := make([]float64, size)
	for i := range f {
		f[i] = 1
	}

	d := mat64.DenseCopyOf(s)
	var want mat64.Vector
	want.SolveVec(d, mat64.NewVector(size, f))

	cg := &CG{MaxIter: maxiter, Tol: tol}
	got, _ := cg.Solve(s, f)
	t.Logf("converged in %v iterations", cg.Niter)
	for i := range got {
		if math.Abs(got[i]-want.At(i, 0)) > tol {
			t.Fatalf("solutions don't match")
		}
	}
	t.Logf("    solver stats:\n%v", cg.Status())
}

func TestGaussJordan(t *testing.T) {
	var tests = []struct {
		vals []float64
		b    []float64
	}{
		{
			vals: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				0, 2, 0, 0,
				1, 0, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				.5, .5, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				0, 2, 0, 0,
				.5, .5, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		},
	}

	for i, test := range tests {
		size := len(test.b)

		A := NewSparse(size)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				A.Set(i, j, test.vals[i*size+j])
			}
		}
		refA := NewSparse(size)
		refA.Clone(A)
		refb := make([]float64, len(test.b))
		copy(refb, test.b)

		var x mat64.Vector
		x.SolveVec(mat64.NewDense(size, size, test.vals), mat64.NewVector(size, test.b))
		want := x.RawVector().Data

		solver := GaussJordan{}
		gotx, _ := solver.Solve(A, test.b)

		failed := false
		for i := range want {
			if diff := math.Abs(gotx[i] - want[i]); diff > eps {
				failed = true
				break
			}
		}

		if failed {
			t.Errorf("test %v A=\n%v\nb=%v", i+1, mat64.Formatted(refA), refb)
			t.Errorf("    x: got %v, want %v", gotx, want)
		}
	}
}

func TestGaussJordanSym(t *testing.T) {
	var tests = []struct {
		vals []float64
		b    []float64
	}{
		{
			vals: []float64{
				1, 1, 1, 1,
				1, 2, 0, 0,
				1, 0, 3, 0,
				1, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				1, 1, 1, 1,
				0, 1, 3, 0,
				0, 1, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				1, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				.5, .5, 0, 0,
				.5, 2, 1, 0,
				0, 1, 3, 1,
				0, 0, 1, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, .5, 1, 1,
				.5, .5, 0, 1,
				1, 0, 3, 0,
				1, 1, 0, 1,
			},
			b: []float64{1, 2, 3, 4},
		},
	}

	for i, test := range tests {
		t.Logf("test %v:", i+1)
		size := len(test.b)

		A := NewSparse(size)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				A.Set(i, j, test.vals[i*size+j])
			}
		}
		refA := NewSparse(size)
		refA.Clone(A)
		refb := make([]float64, len(test.b))
		copy(refb, test.b)

		var x mat64.Vector
		x.SolveVec(mat64.NewDense(size, size, test.vals), mat64.NewVector(size, test.b))
		want := x.RawVector().Data

		solver := GaussJordanSym{}
		gotx, _ := solver.Solve(A, test.b)

		failed := false
		for i := range want {
			if diff := math.Abs(gotx[i] - want[i]); diff > eps {
				failed = true
				break
			}
		}

		if failed {
			t.Errorf("test %v FAILED:", i+1)
			t.Errorf("        A=\n%v\nb=%v", i+1, mat64.Formatted(refA), refb)
			t.Errorf("    x: got %v, want %v", gotx, want)
		} else {
			t.Logf("    PASSED")
		}
	}
}

func BenchmarkGonumLU(b *testing.B) {
	size := 5000
	nfill := 6 // number filled entries per row

	s := mat64.NewDense(size, size, nil)
	for i := 0; i < size; i++ {
		s.Set(i, i, 10)
	}

	for i := 0; i < size; i++ {
		for n := 0; n < nfill/2; n++ {
			j := rand.Intn(size)
			if i == j {
				n--
				continue
			}
			v := rand.Float64()
			s.Set(i, j, v)
			s.Set(j, i, v)
		}
	}

	f := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		f.SetVec(i, 1)
	}

	b.ResetTimer()
	var x mat64.Vector
	for i := 0; i < b.N; i++ {
		x.SolveVec(s, f)
	}
}

func BenchmarkGaussJordanSym(b *testing.B) {
	size := 5000
	nfill := 6 // number filled entries per row

	s := randSparse(size, nfill, 0)

	f := make([]float64, size)
	for i := range f {
		f[i] = 1
	}
	solver := GaussJordanSym{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		solver.Solve(s, f)
	}
}

func BenchmarkCGSolve(b *testing.B) {
	size := 5000
	nfill := 6 // number filled entries per row

	s := randSparse(size, nfill, 0)

	f := make([]float64, size)
	for i := range f {
		f[i] = 1
	}

	maxiter := 1000
	tol := 1e-6
	b.ResetTimer()
	cg := &CG{MaxIter: maxiter, Tol: tol}
	for i := 0; i < b.N; i++ {
		cg.Solve(s, f)
		b.Logf("converged in %v iterations", cg.Niter)
	}
}
