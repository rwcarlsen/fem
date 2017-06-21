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
	chol := NewCholesky(RestrictByPattern{Matrix: NewSparse(size), Pattern: A}, A)

	return func(z, r []float64) {
		zz, err := chol.Solve(r)
		if err != nil {
			panic(err)
		}
		copy(z, zz)
	}
}

func Jacobi(A Matrix) Preconditioner {
	size, _ := A.Dims()
	diag := make([]float64, size)
	for i := range diag {
		diag[i] = 1 / A.At(i, i)
	}

	return func(z, r []float64) {
		for i, val := range r {
			z[i] = val * diag[i]
		}
	}
}

func BlockLU(A Matrix, blocksize int) Preconditioner {
	size, _ := A.Dims()
	end := 0
	lus := []*mat64.LU{}

	//fmt.Printf("A=\n% .2v\n", mat64.Formatted(A))
	for start := 0; end < size; start += blocksize {
		end = start + blocksize
		if end > size {
			end = size
		}
		n := end - start

		Asub := mat64.NewDense(n, n, nil)
		ii := 0
		for i := start; i < end; i++ {
			jj := 0
			for j := start; j < end; j++ {
				Asub.Set(ii, jj, A.At(i, j))
				jj++
			}
			ii++
		}
		//fmt.Printf("Asub=\n% .2v\n", mat64.Formatted(Asub))
		var lu mat64.LU
		lu.Factorize(Asub)
		lus = append(lus, &lu)

		var u mat64.Vector
		b := mat64.NewVector(n, nil)
		for i := 0; i < n; i++ {
			b.SetVec(i, 1)
		}
		u.SolveVec(Asub, b)
		//fmt.Printf("Ax=b soln:\n% .2v\n", mat64.Formatted(&u))
	}

	return func(z, r []float64) {
		i := 0
		u := mat64.NewVector(size, z)
		b := mat64.NewVector(size, r)
		end := 0
		for start := 0; end < size; start += blocksize {
			end = start + blocksize
			if end > size {
				end = size
			}
			subu := u.SliceVec(start, end)
			subb := b.SliceVec(start, end)
			subu.SolveLUVec(lus[i], false, subb)
			//fmt.Printf("iterating Ax=b soln:\n% .2v\n", mat64.Formatted(u))
			i++
		}
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
		//fmt.Printf("------ iter %v ------\n% v\n", k, mat64.Formatted(L))
		// diag
		akk := L.At(k, k)
		for _, nonzero := range L.SweepRow(k) {
			i := nonzero.J
			val := nonzero.Val
			if i < k {
				akk -= val * val
			}
		}
		akk = math.Sqrt(akk)
		L.Set(k, k, akk)

		// below diag
		for _, nonzero := range L.SweepCol(k) {
			i := nonzero.I
			if i > k && nonzero.Val != 0 {
				nonzero.Val /= akk
			}
		}
		for _, nonzero := range L.SweepCol(k) {

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
			}
			tot += y[nonzero.J] * nonzero.Val
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
		//cg.Preconditioner = func(z, r []float64) { copy(z, r) }
		//cg.Preconditioner = Jacobi(A)
		//cg.Preconditioner = BlockLU(A, len(A.SweepRow(0))*2)
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

	// save original residual for convergence/termination criterion
	r0 := make([]float64, size)
	copy(r0, r)

	for cg.niter = 1; cg.niter <= cg.MaxIter; cg.niter++ {
		alpha := dot(r, z) / dot(p, Mul(A, p))
		vecAdd(x, x, vecMult(p, alpha))             // xnext = x+alpha*p
		vecSub(rnext, r, vecMult(Mul(A, p), alpha)) // rnext = r-alpha*A*p
		diff := math.Sqrt(dot(rnext, rnext) / dot(r0, r0))
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

// Note that this implementation is untested and probably broken
//type GaussSeidel struct{}
//
//func (g *GaussSeidel) solveRow(i int, A Matrix, b, soln []float64) {
//	acceleration := 1.8 // between 1.0 and 2.0
//	xold := soln[i]
//	soln[i] = 0
//
//	Ainverse := 1 / A.At(i, i)
//	dot := 0.0
//	for _, nonzero := range A.SweepRow(i) {
//		dot += nonzero.Val * soln[nonzero.I]
//	}
//
//	xnew := (1-acceleration)*xold +
//		acceleration*Ainverse*(b[i]-dot)
//	soln[i] = xnew
//}
//
//func (g *GaussSeidel) forwardIter(A Matrix, b, soln []float64) {
//	for i := 0; i < len(b); i++ {
//		g.solveRow(i, A, b, soln)
//	}
//}
//
//func (g *GaussSeidel) backwardIter(A Matrix, b, soln []float64) {
//	for i := len(b) - 1; i >= 0; i-- {
//		g.solveRow(i, A, b, soln)
//	}
//}
//
//func (g *GaussSeidel) Iterate(A Matrix, x, b []float64) {
//	g.forwardIter(A, b, x)
//	g.backwardIter(A, b, x)
//}
