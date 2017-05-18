package lu

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

const eps = 1e-6

type Sparse struct {
	// map[col]map[row]val
	nonzeroRow []map[int]float64
	// map[row]map[col]val
	nonzeroCol []map[int]float64
	size       int
}

func NewSparse(size int) *Sparse {
	return &Sparse{
		nonzeroRow: make([]map[int]float64, size),
		nonzeroCol: make([]map[int]float64, size),
		size:       size,
	}
}

func (s *Sparse) Clone() *Sparse {
	clone := NewSparse(s.size)
	for i, m := range s.nonzeroCol {
		for j, v := range m {
			clone.Set(i, j, v)
		}
	}
	return clone
}

func (s *Sparse) NonzeroRows(col int) (rows map[int]float64) { return s.nonzeroRow[col] }
func (s *Sparse) NonzeroCols(row int) (cols map[int]float64) { return s.nonzeroCol[row] }

func (s *Sparse) T() mat64.Matrix     { return mat64.Transpose{s} }
func (s *Sparse) Dims() (int, int)    { return s.size, s.size }
func (s *Sparse) At(i, j int) float64 { return s.nonzeroCol[i][j] }
func (s *Sparse) Set(i, j int, v float64) {
	if math.Abs(v) < eps {
		delete(s.nonzeroCol[i], j)
		delete(s.nonzeroRow[j], i)
		return
	}
	if s.nonzeroCol[i] == nil {
		s.nonzeroCol[i] = make(map[int]float64)
	}
	if s.nonzeroRow[j] == nil {
		s.nonzeroRow[j] = make(map[int]float64)
	}

	s.nonzeroCol[i][j] = v
	s.nonzeroRow[j][i] = v
}

// Permute maps i and j indices to new i and j values idendified by the given
// mapping.  Values stored in s.At(i,j) are stored into a newly created sparse
// matrix at new.At(mapping[i], mapping[j]).  The permuted matrix is returned
// and the original remains unmodified.
func (s *Sparse) Permute(mapping []int) *Sparse {
	clone := NewSparse(s.size)
	for i := 0; i < s.size; i++ {
		for j, val := range s.NonzeroCols(i) {
			clone.Set(mapping[i], mapping[j], val)
		}
	}
	return clone
}

func RowCombination(s *Sparse, pivrow, dstrow int, mult float64) {
	for col, aij := range s.NonzeroCols(pivrow) {
		s.Set(dstrow, col, s.At(dstrow, col)+aij*mult)
	}
}

func RowMult(s *Sparse, row int, mult float64) {
	cols := s.NonzeroCols(row)
	for col, val := range cols {
		s.Set(row, col, val*mult)
	}
}

// RCM provides an alternate degree-of-freedom reordering in assembled matrix
// that provides better bandwidth properties for solvers.
func RCM(A *Sparse) []int {
	size, _ := A.Dims()
	mapping := make(map[int]int, size)

	// find row with farthest left centroid
	minnonzeros := 1000000000
	startrow := -1
	for i := 0; i < size; i++ {
		cols := A.NonzeroCols(i)
		if len(cols) < minnonzeros {
			minnonzeros = len(cols)
			startrow = i
		}
	}

	// breadth-first search across adjacency/connections between nodes/dofs
	nextlevel := []int{startrow}
	for n := 0; n < size; n++ {
		if len(nextlevel) == 0 {
			// Matrix must not represent a fully connected graph. We need to choose a random dof/index
			// that we haven't remapped yet to start from
			newstart := -1
			for k := 0; k < size; k++ {
				if _, ok := mapping[k]; !ok {
					newstart = k
					break
				}
			}
			nextlevel = []int{newstart}
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

func nextRCMLevel(A *Sparse, mapping map[int]int, ii []int) []int {
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

// GaussJordanSymm uses the Cuthill-McKee algorithm to permute the matrix
// indices/DOF to have a smaller bandwidth.
func GaussJordanSym(A *Sparse, b []float64) []float64 {
	size, _ := A.Dims()

	mapping := RCM(A)
	AA := A.Permute(mapping)
	bb := make([]float64, size)
	for i, inew := range mapping {
		bb[inew] = b[i]
	}
	x := GaussJordan(AA, bb)

	// re-sequence solution based on RCM permutation/reordering
	xx := make([]float64, size)
	for i, inew := range mapping {
		xx[i] = x[inew]
	}
	return xx
}

func GaussJordan(A *Sparse, b []float64) []float64 {
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

	return x
}

// applyPivot uses the given pivot row to multiply and add to all other rows
// in A either above or below the pivot (dir = -1 for below pivot and 1 for
// above pivot) in order to zero out the given column.  The appropriate
// operations are also performed on b to keep it in sync.
func applyPivot(A *Sparse, b []float64, col int, piv int, dir int) {
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
