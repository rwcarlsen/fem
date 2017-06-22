package sparse

import (
	"fmt"
	"math"
)

type Cholesky struct {
	L Matrix
}

// NewCholesky computes the Cholesky decomposition of A and stores it in L.
// The returned Cholesky object's L is the same as the passed in L.  If L is
// nil, a new Sparse matrix will be created.  Incomplete factorizations can be
// computed by passing in an L that ignores nonzero entries in certain
// locations.
func NewCholesky(L, A Matrix) *Cholesky {
	size, _ := A.Dims()
	if L == nil {
		L = NewSparse(size)
	}
	Copy(L, A)

	for k := 0; k < size; k++ {
		//fmt.Printf("------ iter %v ------\n% v\n", k, mat64.Formatted(L))
		//fmt.Printf("------ iter %v ------\n", k)
		// diag
		akk := L.At(k, k)
		for _, nonzero := range L.SweepRow(k) {
			i := nonzero.J
			val := nonzero.Val
			if i < k {
				akk -= val * val
			}
		}
		if akk < 0 {
			panic(fmt.Sprintf("cholesky factorzation caused neqtive sqrt of %v", akk))
		}
		akk = math.Sqrt(akk)
		L.Set(k, k, akk)

		// below diag
		//for _, nonzero := range L.SweepCol(k) {
		//	i := nonzero.I
		//	if i > k && nonzero.Val != 0 {
		//		nonzero.Val /= akk
		//	}
		//}
		for _, nonzero := range L.SweepCol(k) {
			i := nonzero.I
			if i > k && nonzero.Val != 0 {
				nonzero.Val /= akk
			}

			j := nonzero.I
			ajk := nonzero.Val
			if j <= k {
				continue
			}
			for _, nonzero := range L.SweepCol(k) {
				i := nonzero.I
				aik := nonzero.Val
				if i > j {
					aij := L.At(i, j)
					//fmt.Printf("i=%v, j=%v, aij=%v, subtracting=%v\n", i, j, aij, aik*ajk)
					L.Set(i, j, aij-aik*ajk)
				}
			}
		}

	}

	// zero out above the diagonal
	for i := 0; i < size; i++ {
		for _, nonzero := range L.SweepRow(i) {
			if nonzero.J > i {
				nonzero.Val = 0
			}
		}
	}
	//fmt.Printf("------ done ------\n% v\n", mat64.Formatted(L))
	return &Cholesky{L: L}
}

func (c *Cholesky) Solve(b []float64) (x []float64, err error) {
	// Solve Ly = b via forward substitution
	y := make([]float64, len(b))
	for i := 0; i < len(b); i++ {
		tot := 0.0
		div := 0.0
		for _, nonzero := range c.L.SweepRow(i) {
			if nonzero.J == nonzero.I {
				div = nonzero.Val
			} else {
				tot += y[nonzero.J] * nonzero.Val
			}
		}
		y[i] = (b[i] - tot) / div
	}

	// Solve Ux = y via backward substitution
	x = make([]float64, len(b))
	for i := len(b) - 1; i >= 0; i-- {
		// this inversion (SweepCol instead of SweepRow simulates the L->U transpose)
		tot := 0.0
		div := 0.0
		for _, nonzero := range c.L.SweepCol(i) {
			if nonzero.I == nonzero.J {
				div = nonzero.Val
			} else {
				tot += x[nonzero.I] * nonzero.Val
			}
		}
		x[i] = (y[i] - tot) / div
	}
	return x, nil
}
