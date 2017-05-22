package sparse

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

type Solver interface {
	Solve(A *Matrix, b []float64) (soln []float64, err error)
	Status() string
}

// CG implements a linear conjugate gradient solver (see
// http://wikipedia.org/wiki/Conjugate_gradient_method)
type CG struct {
	MaxIter int
	Tol     float64
	Niter   int
}

func (cg *CG) Status() string { return fmt.Sprintf("converged in %v iterations", cg.Niter) }

func (cg *CG) Solve(A *Matrix, b []float64) (x []float64, err error) {
	size := len(b)

	x = make([]float64, size)
	r := make([]float64, size)
	p := make([]float64, size)
	rnext := make([]float64, size)

	vecSub(r, b, A.Mul(x))
	copy(p, r)

	for cg.Niter = 0; cg.Niter < cg.MaxIter; cg.Niter++ {
		alpha := dot(r, r) / dot(p, A.Mul(p))
		vecAdd(x, x, vecMult(p, alpha))            // xnext = x+alpha*p
		vecSub(rnext, r, vecMult(A.Mul(p), alpha)) // rnext = r-alpha*A*p
		if math.Sqrt(dot(rnext, rnext)) < cg.Tol {
			break
		}
		beta := dot(rnext, rnext) / dot(r, r)
		vecAdd(p, rnext, vecMult(p, beta)) // pnext = rnext + beta*p
		r, rnext = rnext, r
	}

	return x, nil
}

// GaussJordanSymm uses gaussian elimination with the Cuthill-McKee algorithm to permute the
// matrix indices/DOF to have a smaller bandwidth.
type GaussJordanSym struct{}

func (_ GaussJordanSym) Status() string { return "" }

func (_ GaussJordanSym) Solve(A *Matrix, b []float64) ([]float64, error) {
	size, _ := A.Dims()

	mapping := RCM(A)
	AA := A.Permute(mapping)
	bb := make([]float64, size)
	for i, inew := range mapping {
		bb[inew] = b[i]
	}

	x, err := GaussJordan{}.Solve(AA, bb)
	if err != nil {
		return nil, err
	}

	// re-sequence solution based on RCM permutation/reordering
	xx := make([]float64, size)
	for i, inew := range mapping {
		xx[i] = x[inew]
	}
	return xx, nil
}

type DenseLU struct{}

func (_ DenseLU) Status() string { return "" }

func (_ DenseLU) Solve(A *Matrix, b []float64) ([]float64, error) {
	var u mat64.Vector
	if err := u.SolveVec(A, mat64.NewVector(len(b), b)); err != nil {
		return nil, err
	}
	return u.RawVector().Data, nil

}

type GaussJordan struct{}

func (_ GaussJordan) Status() string { return "" }

func (_ GaussJordan) Solve(A *Matrix, b []float64) ([]float64, error) {
	size, _ := A.Dims()

	// Using pivot rows (usually along the diagonal), eliminate all entries
	// below the pivot - doing this choosing a pivot row to eliminate nonzeros
	// in each column.  We only eliminate below the diagonal on the first pass
	// to reduce fill-in.  If we do only one pass total, eliminating entries
	// above the diagonal converts many zero entries into nonzero entries
	// which slows the algorithm down immensely.  The second pass walks the
	// pivot rows in reverse eliminating nonzeros above the pivots (i.e. above
	// the diagonal).

	donerows := make(map[int]bool, size)
	pivots := make([]int, size)

	// first pass
	for j := 0; j < size; j++ {
		// find a first row with a nonzero entry in column i on or below diagonal
		// to use as the pivot row.
		piv := -1
		for i := 0; i < size; i++ {
			if A.At(i, j) != 0 && !donerows[i] {
				piv = i
				break
			}
		}
		//fmt.Printf("selected row %v as pivot\n", piv)
		pivots[j] = piv
		donerows[piv] = true

		applyPivot(A, b, j, pivots[j], -1)
	}

	// second pass
	for j := size - 1; j >= 0; j-- {
		applyPivot(A, b, j, pivots[j], 1)
	}

	// renormalize each row so that leading nonzeros are ones (row echelon to
	// reduced row echelon)
	for j, i := range pivots {
		mult := 1 / A.At(i, j)
		RowMult(A, i, mult)
		b[i] *= mult
	}

	// re-sequence solution based on pivot row indices/order
	x := make([]float64, size)
	for i := range pivots {
		x[i] = b[pivots[i]]
	}

	return x, nil
}

// applyPivot uses the given pivot row to multiply and add to all other rows
// in A either above or below the pivot (dir = -1 for below pivot and 1 for
// above pivot) in order to zero out the given column.  The appropriate
// operations are also performed on b to keep it in sync.
func applyPivot(A *Matrix, b []float64, col int, piv int, dir int) {
	pval := A.At(piv, col)
	bval := b[piv]
	for i, aij := range A.NonzeroRows(col) {
		cond := ((dir == -1) && i > piv) || ((dir == 1) && i < piv)
		if i != piv && cond {
			mult := -aij / pval
			//fmt.Printf("   pivot times %v plus row %v\n", mult, i)
			RowCombination(A, piv, i, mult)
			b[i] += bval * mult
		}
	}
	//fmt.Printf("after:\n%.2v\n", mat64.Formatted(A))
}
