package lu

import (
	"math"
	"sort"

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

func (s *Sparse) Permute(indices []int) *Sparse {
	clone := NewSparse(s.size)
	for i, inew := range indices {
		for j, val := range s.NonzeroCols(i) {
			clone.Set(inew, j, val)
		}
	}
	return clone
}

func GaussJordan(A *Sparse, b []float64) []float64 {
	size, _ := A.Dims()
	rowmap := make([]int, size)
	for i := range rowmap {
		rowmap[i] = i
	}

	//fmt.Println("permuting rows...")
	// permute rows so that nonzero entries are as far left as possible in
	// the higher rows. There was a bug in sort.Slice in go master - so I
	// switched to SliceStable.
	sort.SliceStable(rowmap, func(a, b int) bool {
		centroida := 0.0
		cols := A.NonzeroCols(a)
		for j := range cols {
			centroida += float64(j)
		}
		centroida /= float64(len(cols))

		centroidb := 0.0
		cols = A.NonzeroCols(b)
		for j := range cols {
			centroidb += float64(j)
		}
		centroidb /= float64(len(cols))
		//fmt.Printf("    compare cm(%v)=%v < cm(%v)=%v is %v\n", a, centroida, b, centroidb, centroida < centroidb)
		return centroida < centroidb
	})
	AA := A.Permute(rowmap)
	bb := make([]float64, size)
	for i, inew := range rowmap {
		bb[inew] = b[i]
	}
	//fmt.Println("row permutation:", rowmap)

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
			if AA.At(i, j) != 0 && !donerows[i] {
				piv = i
				break
			}
		}
		//fmt.Printf("selected row %v as pivot\n", piv)
		//fmt.Printf("Num nonzeros for col %v is %v\n", j, len(AA.nonzeroRow[j]))
		pivots[j] = piv
		donerows[piv] = true

		pval := AA.At(piv, j)
		bval := bb[piv]
		for i, aij := range AA.NonzeroRows(j) {
			if i != piv && i > piv {
				mult := -aij / pval
				//fmt.Printf("   pivot times %v plus row %v\n", mult, i)
				RowCombination(AA, piv, i, mult)
				bb[i] += bval * mult
			} else {
				//fmt.Printf("    skipping row %v which is (above) the pivot\n", i)
			}
		}
		//fmt.Printf("after:\n%.2v\n", mat64.Formatted(AA))
	}

	// second pass
	for j := size - 1; j >= 0; j-- {
		piv := pivots[j]
		//fmt.Printf("selected row %v as pivot\n", piv)
		//fmt.Printf("Num nonzeros for col %v is %v\n", j, len(AA.nonzeroRow[j]))

		pval := AA.At(piv, j)
		bval := bb[piv]
		for i, aij := range AA.NonzeroRows(j) {
			if i != piv && i < piv {
				mult := -aij / pval
				//fmt.Printf("   pivot times %v plus row %v\n", mult, i)
				RowCombination(AA, piv, i, mult)
				bb[i] += bval * mult
			} else {
				//fmt.Printf("    skipping row %v which is (below) the pivot\n", i)
			}
		}
		//fmt.Printf("after:\n%.2v\n", mat64.Formatted(AA))
	}

	// renormalize each row so that leading nonzeros are ones (row echelon to
	// reduced row echelon)
	for j, i := range pivots {
		mult := 1 / AA.At(i, j)
		RowMult(AA, i, mult)
		bb[i] *= mult
	}

	// re-sequence solution based on pivot row indices/order
	x := make([]float64, size)
	for i := range pivots {
		x[i] = bb[pivots[i]]
	}

	return x
}
