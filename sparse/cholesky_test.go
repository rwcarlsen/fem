package sparse

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

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
	t.Logf("\n% v", mat64.Formatted(A))

	chol := NewCholesky(nil, A)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if math.Abs(chol.L.At(i, j)-wantL.At(i, j)) > tol {
				t.Fatalf("factorizations don't match:\ngot\n% .3v\nwant\n% .3v", mat64.Formatted(chol.L), mat64.Formatted(wantL))
			}
		}
	}
}

func TestNewCholesky2(t *testing.T) {
	rand.Seed(1)

	for size := 10; size < 500; size = int(float64(size) * 1.5) {
		for nfill := 5; nfill <= size/2; nfill = nfill*2 + 1 {
			t.Run(testNewCholesky(size, nfill))
		}
	}
}

func testNewCholesky(size, nfill int) (string, func(t *testing.T)) {
	return fmt.Sprintf("size=%v,nfill=%v", size, nfill), func(t *testing.T) {
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

		chol := NewCholesky(nil, s)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				if math.Abs(chol.L.At(i, j)-wantL.At(i, j)) > tol {
					if size < 35 {
						t.Fatalf("factorizations don't match at (%v,%v): got %v, want %v:\ngot\n% .3v\nwant\n% .3v", i, j, chol.L.At(i, j), wantL.At(i, j), mat64.Formatted(chol.L), mat64.Formatted(&wantL))
					} else {
						t.Fatalf("factorizations don't match")
					}
				}
			}
		}
	}
}

func testCholesky(size, nfill int) func(t *testing.T) {
	return func(t *testing.T) {
		const tol = 1e-6
		s := randSparse(size, nfill, 0)
		f := make([]float64, size)
		for i := range f {
			f[i] = 1
		}

		d := mat64.DenseCopyOf(s)
		var want mat64.Vector
		want.SolveVec(d, mat64.NewVector(size, f))

		chol := NewCholesky(nil, s)
		got, _ := chol.Solve(f)
		for i := range got {
			if math.Abs(got[i]-want.At(i, 0)) > tol {
				//t.Errorf("A:\n% 3g\nb=%v", mat64.Formatted(s), f)
				//t.Errorf("solutions don't match:\ngot %v\nwant %v", got, want.RawVector().Data)
				t.Errorf("    solutions don't match")
				break
			}
		}
	}
}

func TestCholesky_Solve(t *testing.T) {
	rand.Seed(1)
	t.Run("size=50, nfill=5", testCholesky(50, 5))
	t.Run("size=150, nfill=15", testCholesky(150, 15))

	if !testing.Short() {
		t.Run("size=601, nfill=12", testCholesky(601, 12))
		t.Run("size=601, nfill=13", testCholesky(601, 13))
		t.Run("size=601, nfill=14", testCholesky(601, 14))
		t.Run("size=601, nfill=15", testCholesky(601, 15))
	}
}
