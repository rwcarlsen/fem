package sparse

import (
	"bytes"
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

type Solver interface {
	Solve(A Matrix, b []float64) (soln []float64, err error)
	Status() string
}

// Preconditioner is a function that takes a (e.g. resitual) vector r and
// applies a preconditioning matrix to it and stores the result in z.
type Preconditioner func(z, r []float64)

// IncompleteCholesky returns a preconditioner that uses an incomplete
// cholesky factorization (incomplete via maintaining the same sparsity
// pattern as the matrix A).  The factorization is then used to solve for z in
// the system A*z=r for the preconditioner - i.e. for the preconditioning
// M^(-1)*r, M is the incomplete cholesky factorization.
func IncompleteCholesky(A Matrix) Preconditioner {
	size, _ := A.Dims()
	chol := NewCholesky(RestrictByPattern{NewSparse(size), A}, A)

	return func(z, r []float64) {
		zz, err := chol.Solve(r)
		if err != nil {
			panic(err)
		}
		copy(z, zz)
	}
}

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
		// diag
		akk := math.Sqrt(L.At(k, k))
		L.Set(k, k, akk)

		// below diag
		for i, aik := range L.NonzeroRows(k) {
			if i > k && aik != 0 {
				L.Set(i, k, aik/akk)
			}
		}

		for j, ajk := range L.NonzeroRows(k) {
			if j <= k {
				continue
			}
			for i, aik := range L.NonzeroRows(k) {
				if i >= j {
					aij := L.At(i, j)
					L.Set(i, j, aij-aik*ajk)
				}
			}
		}
	}

	for i := 0; i < size; i++ {
		for j := range L.NonzeroCols(i) {
			if j > i {
				L.Set(i, j, 0)
			}
		}
	}
	return &Cholesky{L: L}
}

func (c *Cholesky) Solve(b []float64) (x []float64, err error) {
	// Solve Ly = b via forward substitution
	y := make([]float64, len(b))
	for i := 0; i < len(b); i++ {
		nonzeros := c.L.SweepRow(i)
		tot := 0.0
		div := 0.0
		for _, nonzero := range nonzeros {
			if nonzero.I == i {
				div = nonzero.Val
			}
			tot += y[nonzero.I] * nonzero.Val
		}
		y[i] = (b[i] - tot) / div
	}

	// Solve Ux = y via backward substitution
	x = make([]float64, len(b))
	for i := len(b) - 1; i >= 0; i-- {
		// this inversion (SweepCol instead of SweepRow simulates the L->U transpose)
		nonzeros := c.L.SweepCol(i)
		tot := 0.0
		div := 0.0
		for _, nonzero := range nonzeros {
			if nonzero.I == i {
				div = nonzero.Val
			}
			tot += x[nonzero.I] * nonzero.Val
		}
		x[i] = (y[i] - tot) / div
	}
	return x, nil
}

// CG implements a linear conjugate gradient solver (see
// http://wikipedia.org/wiki/Conjugate_gradient_method)
type CG struct {
	MaxIter int
	Tol     float64
	// Preconditioner is the preconditioning matrix used for each iteration of
	// the CG solver. If it is nil, a default preconditioner will be used.
	Preconditioner Preconditioner
	niter          int
	ndof           int
}

func (cg *CG) Status() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "CG Solver Stats:\n")
	fmt.Fprintf(&buf, "    %v dof\n", cg.ndof)
	fmt.Fprintf(&buf, "    converged in %v iterations", cg.niter)
	return buf.String()
}

func (cg *CG) Solve(A Matrix, b []float64) (x []float64, err error) {
	if cg.Preconditioner == nil {
		cg.Preconditioner = IncompleteCholesky(A)
	}

	size := len(b)
	cg.ndof = size

	x = make([]float64, size)
	r := make([]float64, size)
	z := make([]float64, size)
	p := make([]float64, size)
	rnext := make([]float64, size)
	znext := make([]float64, size)

	vecSub(r, b, Mul(A, x))
	cg.Preconditioner(z, r)
	copy(p, z)

	for cg.niter = 1; cg.niter <= cg.MaxIter; cg.niter++ {
		alpha := dot(r, z) / dot(p, Mul(A, p))
		vecAdd(x, x, vecMult(p, alpha))             // xnext = x+alpha*p
		vecSub(rnext, r, vecMult(Mul(A, p), alpha)) // rnext = r-alpha*A*p
		diff := math.Sqrt(dot(rnext, rnext))
		if diff < cg.Tol {
			break
		}
		cg.Preconditioner(znext, rnext)
		beta := dot(znext, rnext) / dot(z, r)
		vecAdd(p, znext, vecMult(p, beta)) // pnext = rnext + beta*p
		r, rnext = rnext, r
		z, znext = znext, z
	}

	return x, nil
}

type DenseLU struct{}

func (DenseLU) Status() string { return "" }

func (DenseLU) Solve(A Matrix, b []float64) ([]float64, error) {
	var u mat64.Vector
	if err := u.SolveVec(A, mat64.NewVector(len(b), b)); err != nil {
		return nil, err
	}
	return u.RawVector().Data, nil

}

// GaussJordan performs Gaussian-Jordan elimination on an augmented matrix
// [A|b] to solve the system A*x=b.
type GaussJordan struct{}

func (GaussJordan) Status() string { return "" }

func (gj GaussJordan) Solve(A Matrix, b []float64) ([]float64, error) {
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
		// find a first row with a nonzero entry in column i on or below
		// diagonal to use as the pivot row.
		piv := -1
		for i := 0; i < size; i++ {
			if A.At(i, j) != 0 && !donerows[i] {
				piv = i
				break
			}
		}
		pivots[j] = piv
		donerows[piv] = true

		ApplyPivot(A, b, j, pivots[j], -1)
	}

	// second pass
	for j := size - 1; j >= 0; j-- {
		ApplyPivot(A, b, j, pivots[j], 1)
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

// GaussJordanSymm uses gaussian elimination with the Cuthill-McKee algorithm
// to permute the matrix indices/DOF to have a smaller bandwidth.
type GaussJordanSym struct{}

func (GaussJordanSym) Status() string { return "" }

func (GaussJordanSym) Solve(A Matrix, b []float64) ([]float64, error) {
	size, _ := A.Dims()

	mapping := RCM(A)
	AA := NewSparse(size)
	Permute(AA, A, mapping)
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
