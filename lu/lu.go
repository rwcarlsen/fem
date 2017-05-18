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
		//fmt.Printf("        Set(%v,%v) to zero\n", i, j)
		delete(s.nonzeroCol[i], j)
		delete(s.nonzeroRow[j], i)
		return
	}

	//fmt.Printf("        Set(%v,%v)=%v and added nonzero col+row\n", i, j, v)
	//fmt.Printf("            nonzeroRows(%v)=%v\n", j, s.nonzeroRow[j])
	//fmt.Printf("            nonzeroCols(%v)=%v\n", i, s.nonzeroCol[i])
	if s.nonzeroCol[i] == nil {
		s.nonzeroCol[i] = make(map[int]float64)
	}
	if s.nonzeroRow[j] == nil {
		s.nonzeroRow[j] = make(map[int]float64)
	}

	s.nonzeroCol[i][j] = v
	s.nonzeroRow[j][i] = v
}

func RowCombination(s *Sparse, rowsrc, rowdst int, mult float64) {
	for col, srcval := range s.NonzeroCols(rowsrc) {
		//fmt.Printf("        A(%v,%v) += A(%v,%v)*%v\n", rowdst, col, rowsrc, col, mult)
		s.Set(rowdst, col, s.At(rowdst, col)+srcval*mult)
	}
}

func RowMult(s *Sparse, row int, mult float64) {
	cols := s.NonzeroCols(row)
	for col := range cols {
		s.Set(row, col, s.At(row, col)*mult)
	}
}

func (s *Sparse) PermuteCols(indices []int) *Sparse {
	clone := NewSparse(s.size)
	for j, jnew := range indices {
		for i, val := range s.NonzeroRows(j) {
			clone.Set(i, jnew, val)
		}
	}
	return clone
}

func (s *Sparse) PermuteRows(indices []int) *Sparse {
	clone := NewSparse(s.size)
	for i, inew := range indices {
		for j, val := range s.NonzeroCols(i) {
			clone.Set(inew, j, val)
		}
	}
	return clone
}

func (s *Sparse) Permute(mapping []int) *Sparse {
	clone := NewSparse(s.size)
	for i := 0; i < s.size; i++ {
		for j, val := range s.NonzeroCols(i) {
			clone.Set(mapping[i], mapping[j], val)
		}
	}
	return clone
}

// CM provides an alternate degree-of-freedom reordering in assembled matrix that provides better
// bandwidth properties for solvers.
func CM(A *Sparse) []int {
	size, _ := A.Dims()
	mapping := make(map[int]int, size)

	// find row with farthest left centroid
	mincentroid := math.Inf(1)
	startrow := -1
	for i := 0; i < size; i++ {
		cols := A.NonzeroCols(i)
		centroid := 0.0
		for j := range cols {
			centroid += float64(j)
		}
		centroid /= float64(len(cols))
		if centroid < mincentroid {
			mincentroid = centroid
			startrow = i
		}
	}

	// breadth-first search across adjacency/connections between nodes/dofs
	nextlevel := []int{startrow}
	for n := 0; n < size; n++ {
		for _, i := range nextlevel {
			if _, ok := mapping[i]; !ok {
				mapping[i] = len(mapping)
			}
		}
		if len(mapping) >= size {
			break
		}
		nextlevel = nextCMLevel(A, mapping, nextlevel)
	}

	slice := make([]int, size)
	for i := range slice {
		slice[i] = i
	}
	for from, to := range mapping {
		slice[from] = to
	}
	return slice
}

func nextCMLevel(A *Sparse, mapping map[int]int, ii []int) []int {
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

	mapping := CM(A)
	AA := A.Permute(mapping)
	bb := make([]float64, size)
	for i, inew := range mapping {
		bb[inew] = b[i]
	}
	x := GaussJordan(AA, bb)

	// re-sequence solution based on CM permutation/reordering
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
		//fmt.Printf("Num nonzeros for col %v is %v\n", j, len(A.nonzeroRow[j]))
		pivots[j] = piv
		donerows[piv] = true

		applyPivot(A, b, j, piv, -1)
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

// dir = -1 for below diagonal and 1 for above diagonal
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
