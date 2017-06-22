package sparse

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestLU_Factorize(t *testing.T) {
	size := 3
	data := []float64{
		4, 12, -16,
		12, 37, -43,
		-16, -43, 98,
	}
	wantLdata := []float64{
		1, 0, 0,
		3, 1, 0,
		-4, 5, 1,
	}
	wantUdata := []float64{
		4, 12, -16,
		0, 1, 5,
		0, 0, 9,
	}
	tol := 1e-6

	A := makeSparse(size, data)
	wantL := makeSparse(size, wantLdata)
	wantU := makeSparse(size, wantUdata)
	t.Logf("\n% v", mat64.Formatted(A))

	var lu LU
	lu.Factorize(A)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if math.Abs(lu.L.At(i, j)-wantL.At(i, j)) > tol {
				t.Errorf("factorization L's don't match:\ngot\n% .3v\nwant\n% .3v", mat64.Formatted(lu.L), mat64.Formatted(wantL))
			}
			if math.Abs(lu.U.At(i, j)-wantU.At(i, j)) > tol {
				t.Errorf("factorization U's don't match:\ngot\n% .3v\nwant\n% .3v", mat64.Formatted(lu.U), mat64.Formatted(wantU))
			}
			if t.Failed() {
				return
			}
		}
	}
}

func TestLU_Factorize2(t *testing.T) {
	rand.Seed(1)

	for size := 5; size < 300; size = int(float64(size) * 1.5) {
		for nfill := 5; nfill <= size/2; nfill = nfill*2 + 1 {
			t.Run(testLU_Factorize(size, nfill))
		}
	}
}

func testLU_Factorize(size, nfill int) (string, func(t *testing.T)) {
	return fmt.Sprintf("size=%v,nfill=%v", size, nfill), func(t *testing.T) {
		tol := 1e-6

		A := randSparse(size, nfill, 0)
		f := make([]float64, size)
		d := mat64.NewSymDense(size, nil)
		for i := range f {
			f[i] = 1
			for j := range f {
				if j >= i {
					d.SetSym(i, j, A.At(i, j))
				}
			}
		}

		var reflu mat64.LU
		reflu.Factorize(d)
		var wantL mat64.TriDense
		var wantU mat64.TriDense
		wantL.LFromLU(&reflu)
		wantU.UFromLU(&reflu)

		var lu LU
		lu.Factorize(A)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				if math.Abs(lu.L.At(i, j)-wantL.At(i, j)) > tol {
					if size < 35 {
						t.Errorf("factorization L's don't match at (%v,%v): got %v, want %v:\ngot\n% .2v\nwant\n% .2v", i, j, lu.L.At(i, j), wantL.At(i, j), mat64.Formatted(lu.L), mat64.Formatted(&wantL))
					} else {
						t.Errorf("factorization L's don't match")
					}
				}
				if math.Abs(lu.U.At(i, j)-wantU.At(i, j)) > tol {
					if size < 35 {
						t.Errorf("factorization U's don't match at (%v,%v): got %v, want %v:\ngot\n% .2v\nwant\n% .2v", i, j, lu.U.At(i, j), wantU.At(i, j), mat64.Formatted(lu.U), mat64.Formatted(&wantU))
					} else {
						t.Errorf("factorization U's don't match")
					}
				}
				if t.Failed() {
					return
				}
			}
		}
	}
}

func testLU(size, nfill int) (string, func(t *testing.T)) {
	return fmt.Sprintf("size=%v,nfill=%v", size, nfill), func(t *testing.T) {
		const tol = 1e-6
		s := randSparse(size, nfill, 0)
		f := make([]float64, size)
		for i := range f {
			f[i] = 1
		}

		d := mat64.DenseCopyOf(s)
		var want mat64.Vector
		want.SolveVec(d, mat64.NewVector(size, f))

		var lu LU
		lu.Factorize(s)
		got, _ := lu.Solve(f, nil)
		for i := range got {
			if math.Abs(got[i]-want.At(i, 0)) > tol {
				//t.Errorf("A:\n% 3g\nb=%v", mat64.Formatted(s), f)
				if size < 35 {
					t.Errorf("solutions don't match:\ngot %v\nwant %v", got, want.RawVector().Data)
				} else {
					t.Errorf("solutions don't match")
				}
				break
			}
		}
	}
}

func TestLU_Solve(t *testing.T) {
	rand.Seed(1)
	t.Run(testLU(5, 5))
	t.Run(testLU(10, 3))
	t.Run(testLU(15, 3))
	t.Run(testLU(50, 5))
	t.Run(testLU(150, 15))

	if !testing.Short() {
		t.Run(testLU(251, 12))
		t.Run(testLU(351, 13))
		t.Run(testLU(601, 12))
		t.Run(testLU(601, 13))
		t.Run(testLU(601, 14))
		t.Run(testLU(601, 15))
	}
}
