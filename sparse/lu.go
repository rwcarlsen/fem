package sparse

// LU performs Gaussian-Jordan elimination on an augmented matrix
// [A|b] to solve the system A*x=b.
type LU struct {
	L, U   Matrix
	Pivots []int
}

func (LU) Status() string { return "" }

func (lu LU) Solve(A Matrix, b []float64) ([]float64, error) {
	// re-sequence solution based on pivot row indices/order
	x := make([]float64, size)
	for i := range lu.Pivots {
		x[i] = b[lu.Pivots[i]]
	}
}

func NewLU(A Matrix, b []float64, L, U Matrix) *LU {
	size, _ := A.Dims()
	lu := &LU{L: L, U: U, Pivots: make([]int, size)}
	if L == nil {
		lu.L = NewSparse(size)
	}
	if U == nil {
		lu.U = NewSparse(size)
	}

	Copy(lu.U, A)

	// Using pivot rows (usually along the diagonal), eliminate all entries
	// below the pivot - doing this choosing a pivot row to eliminate nonzeros
	// in each column.  We only eliminate below the diagonal on the first pass
	// to reduce fill-in.  If we do only one pass total, eliminating entries
	// above the diagonal converts many zero entries into nonzero entries
	// which slows the algorithm down immensely.  The second pass walks the
	// pivot rows in reverse eliminating nonzeros above the pivots (i.e. above
	// the diagonal).

	donerows := make(map[int]bool, size)

	// first pass
	for j := 0; j < size; j++ {
		// find a first row with a nonzero entry in column i on or below
		// diagonal to use as the pivot row.
		piv := -1
		for i := 0; i < size; i++ {
			if U.At(i, j) != 0 && !donerows[i] {
				piv = i
				break
			}
		}
		lu.Pivots[j] = piv
		donerows[piv] = true

		ApplyPivot(U, b, j, lu.Pivots[j], -1, lu.L)
	}

	return lu
}
