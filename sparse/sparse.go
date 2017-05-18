package sparse

import (
	"fmt"
	"math"
	"sort"

	"github.com/gonum/matrix/mat64"
)

const eps = 1e-6

type Matrix struct {
	// map[col]map[row]val
	nonzeroRow []map[int]float64
	// map[row]map[col]val
	nonzeroCol []map[int]float64
	size       int
}

func New(size int) *Matrix {
	return &Matrix{
		nonzeroRow: make([]map[int]float64, size),
		nonzeroCol: make([]map[int]float64, size),
		size:       size,
	}
}

func (m *Matrix) Clone() *Matrix {
	clone := New(m.size)
	for i, m := range m.nonzeroCol {
		for j, v := range m {
			clone.Set(i, j, v)
		}
	}
	return clone
}

func (m *Matrix) NonzeroRows(col int) (rows map[int]float64) { return m.nonzeroRow[col] }
func (m *Matrix) NonzeroCols(row int) (cols map[int]float64) { return m.nonzeroCol[row] }

func (m *Matrix) T() mat64.Matrix     { return mat64.Transpose{m} }
func (m *Matrix) Dims() (int, int)    { return m.size, m.size }
func (m *Matrix) At(i, j int) float64 { return m.nonzeroCol[i][j] }
func (m *Matrix) Set(i, j int, v float64) {
	if math.Abs(v) < eps {
		delete(m.nonzeroCol[i], j)
		delete(m.nonzeroRow[j], i)
		return
	}
	if m.nonzeroCol[i] == nil {
		m.nonzeroCol[i] = make(map[int]float64)
	}
	if m.nonzeroRow[j] == nil {
		m.nonzeroRow[j] = make(map[int]float64)
	}

	m.nonzeroCol[i][j] = v
	m.nonzeroRow[j][i] = v
}

// Permute maps i and j indices to new i and j values idendified by the given
// mapping.  Values stored in m.At(i,j) are stored into a newly created sparse
// matrix at new.At(mapping[i], mapping[j]).  The permuted matrix is returned
// and the original remains unmodified.
func (m *Matrix) Permute(mapping []int) *Matrix {
	clone := New(m.size)
	for i := 0; i < m.size; i++ {
		for j, val := range m.NonzeroCols(i) {
			clone.Set(mapping[i], mapping[j], val)
		}
	}
	return clone
}

func (m *Matrix) Mul(b []float64) []float64 {
	result := make([]float64, len(b))
	for i := 0; i < m.size; i++ {
		tot := 0.0
		for j, val := range m.NonzeroCols(i) {
			tot += b[j] * val
		}
		result[i] = tot
	}
	return result
}

func RowCombination(m *Matrix, pivrow, dstrow int, mult float64) {
	for col, aij := range m.NonzeroCols(pivrow) {
		m.Set(dstrow, col, m.At(dstrow, col)+aij*mult)
	}
}

func RowMult(m *Matrix, row int, mult float64) {
	cols := m.NonzeroCols(row)
	for col, val := range cols {
		m.Set(row, col, val*mult)
	}
}

// RCM provides an alternate degree-of-freedom reordering in assembled matrix
// that provides better bandwidth properties for solvers.
func RCM(A *Matrix) []int {
	size, _ := A.Dims()
	mapping := make(map[int]int, size)

	degreemap := make([]int, size)
	for i := range degreemap {
		degreemap[i] = i
	}

	sort.SliceStable(degreemap, func(i, j int) bool {
		return len(A.NonzeroCols(degreemap[i])) < len(A.NonzeroCols(degreemap[j]))
	})
	startrow := degreemap[0]

	// breadth-first search across adjacency/connections between nodes/dofs
	nextlevel := []int{startrow}
	for n := 0; n < size; n++ {
		if len(nextlevel) == 0 {
			// Matrix must not represent a fully connected graph. We need to choose a random dof/index
			// that we haven't remapped yet to start from
			for _, k := range degreemap {
				if _, ok := mapping[k]; !ok {
					nextlevel = []int{k}
					break
				}
			}
		}

		for _, i := range nextlevel {
			if _, ok := mapping[i]; !ok {
				mapping[i] = len(mapping)
			}
		}
		if len(mapping) >= size {
			break
		}
		nextlevel = nextRCMLevel(A, mapping, nextlevel)
	}

	slice := make([]int, size)

	reverse := make([]int, size)
	count := size - 1
	for i := range reverse {
		reverse[i] = count
		count--
	}

	for from, to := range mapping {
		slice[from] = reverse[to]
	}
	return slice
}

func nextRCMLevel(A *Matrix, mapping map[int]int, ii []int) []int {
	var nextlevel []int
	for _, i := range ii {
		for j := range A.NonzeroCols(i) {
			if _, ok := mapping[j]; !ok {
				nextlevel = append(nextlevel, j)
			}
		}
	}
	return nextlevel
}

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

func vecAdd(result, a, b []float64) {
	if len(a) != len(b) {
		panic("inconsistent lengths for vector subtraction")
	}
	for i := range a {
		result[i] = a[i] + b[i]
	}
}

func vecSub(result, a, b []float64) {
	if len(a) != len(b) {
		panic("inconsistent lengths for vector subtraction")
	}
	for i := range a {
		result[i] = a[i] - b[i]
	}
}

// dot performs a vector*vector dot product.
func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("inconsistent lengths for dot product")
	}
	v := 0.0
	for i := range a {
		v += a[i] * b[i]
	}
	return v
}

func vecMult(v []float64, mult float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = mult * v[i]
	}
	return result
}
