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
	t.Logf("converged in %v iterations", cg.niter)
	for i := range got {
		if math.Abs(got[i]-want.At(i, 0)) > tol {
			t.Errorf("solutions don't match")
			t.Errorf("    got %v", got)
			t.Errorf("    want %v", want.RawVector().Data)
			return
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
		Copy(refA, A)
		refb := make([]float64, len(test.b))
		copy(refb, test.b)

		var x mat64.Vector
		x.SolveVec(mat64.NewDense(size, size, test.vals), mat64.NewVector(size, test.b))
		want := x.RawVector().Data

		solver := GaussJordan{}
		gotx, _ := solver.Solve(A, test.b)

		failed := false
		for i := range want {
			if diff := math.Abs(gotx[i] - want[i]); diff > refA.Tol {
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
		Copy(refA, A)
		refb := make([]float64, len(test.b))
		copy(refb, test.b)

		var x mat64.Vector
		x.SolveVec(mat64.NewDense(size, size, test.vals), mat64.NewVector(size, test.b))
		want := x.RawVector().Data

		solver := GaussJordanSym{}
		gotx, _ := solver.Solve(A, test.b)

		failed := false
		for i := range want {
			if diff := math.Abs(gotx[i] - want[i]); diff > refA.Tol {
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
		b.Logf("converged in %v iterations", cg.niter)
	}
}
