package lu

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

type index struct {
	row, col int32
}

type Sparse struct {
	// map[row][]col
	data           map[index]float64
	nonzeroRow     []map[int]int
	nonzeroCol     []map[int]int
	nonzeroRowList [][]int
	nonzeroColList [][]int
	size           int
}

func NewSparse(size int) *Sparse {
	return &Sparse{
		data:           make(map[index]float64),
		nonzeroRow:     make([]map[int]int, size),
		nonzeroCol:     make([]map[int]int, size),
		nonzeroRowList: make([][]int, size),
		nonzeroColList: make([][]int, size),
		size:           size,
	}
}

const eps = 1e-6

func (s *Sparse) NonzeroRows(col int) (rows []int) { return s.nonzeroRowList[col] }
func (s *Sparse) NonzeroCols(row int) (cols []int) { return s.nonzeroColList[row] }

func (s *Sparse) Clone() *Sparse {
	clone := NewSparse(s.size)
	for k, v := range s.data {
		clone.Set(int(k.row), int(k.col), v)
	}
	return clone
}

func (s *Sparse) T() mat64.Matrix     { return mat64.Transpose{s} }
func (s *Sparse) Dims() (int, int)    { return s.size, s.size }
func (s *Sparse) At(i, j int) float64 { return s.data[index{int32(i), int32(j)}] }
func (s *Sparse) Set(i, j int, v float64) {
	defer func() {
		fmt.Printf("    len(s.nonzeroCol[%v])=%v\n", i, len(s.nonzeroCol[i]))
		fmt.Printf("    len(s.nonzeroColList[%v])=%v\n", i, len(s.nonzeroColList[i]))
	}()
	fmt.Println("Sparse.Set()")
	if math.Abs(v) < eps {
		fmt.Println("setting to zero")
		//fmt.Printf("        Set(%v,%v) to zero\n", i, j)
		fmt.Printf("len(s.nonzeroCol[%v])=%v\n", i, len(s.nonzeroCol[i]))
		fmt.Printf("len(s.nonzeroColList[%v])=%v\n", i, len(s.nonzeroColList[i]))
		if ind, ok := s.nonzeroCol[i][j]; ok {
			list := s.nonzeroColList[i]
			for n := ind; n < len(list)-1; n++ {
				list[n] = list[n+1]
			}
			for n := 0; n < ind; n++ {
				s.nonzeroCol[i][n]--
			}
			s.nonzeroColList[i] = list[:len(list)-1]
		}
		if ind, ok := s.nonzeroRow[j][i]; ok {
			list := s.nonzeroRowList[j]
			for n := ind; n < len(list)-1; n++ {
				list[n] = list[n+1]
			}
			for n := 0; n < ind; n++ {
				s.nonzeroRow[j][n]--
			}
			s.nonzeroRowList[j] = list[:len(list)-1]
		}

		delete(s.data, index{int32(i), int32(j)})
		delete(s.nonzeroCol[i], j)
		delete(s.nonzeroRow[j], i)
		return
	}

	//fmt.Printf("        Set(%v,%v)=%v and added nonzero col+row\n", i, j, v)
	//fmt.Printf("            nonzeroRows(%v)=%v\n", j, s.nonzeroRow[j])
	//fmt.Printf("            nonzeroCols(%v)=%v\n", i, s.nonzeroCol[i])
	if s.nonzeroCol[i] == nil {
		s.nonzeroCol[i] = make(map[int]int)
	} else if ind, ok := s.nonzeroCol[i][j]; ok {
		fmt.Printf("ind=%v, len(nonzeroColList[%v])=%v\n", ind, i, len(s.nonzeroColList))
		s.nonzeroColList[i][ind] = j
	} else {
		s.nonzeroCol[i][j] = len(s.nonzeroColList)
		s.nonzeroColList[i] = append(s.nonzeroColList[i], j)
	}

	if s.nonzeroRow[j] == nil {
		s.nonzeroRow[j] = make(map[int]int)
	} else if ind, ok := s.nonzeroRow[j][i]; ok {
		s.nonzeroRowList[j][ind] = i
	} else {
		s.nonzeroRow[j][i] = len(s.nonzeroRowList)
		s.nonzeroRowList[j] = append(s.nonzeroRowList[j], i)
	}

	s.data[index{int32(i), int32(j)}] = v
}

func RowCombination(s *Sparse, rowsrc, rowdst int, mult float64) {
	cols := s.NonzeroCols(rowsrc)
	for _, col := range cols {
		//fmt.Printf("        A(%v,%v) += A(%v,%v)*%v\n", rowdst, col, rowsrc, col, mult)
		s.Set(rowdst, col, s.At(rowdst, col)+s.At(rowsrc, col)*mult)
	}
}

func RowMult(s *Sparse, row int, mult float64) {
	cols := s.NonzeroCols(row)
	for col := range cols {
		s.Set(row, col, s.At(row, col)*mult)
	}
}

func GaussJordan(A *Sparse, b []float64) []float64 {
	size, _ := A.Dims()
	donerows := make(map[int]bool, size)
	pivots := make([]int, size)

	// Using pivot rows (usually along the diagonal), eliminate all entries
	// below the pivot - doing this choosing a pivot row to eliminate nonzeros
	// in each column.  We only eliminate below the diagonal on the first pass
	// to reduce fill-in.  If we do only one pass total, eliminating entries
	// above the diagonal converts many zero entries into nonzero entries
	// which slows the algorithm down immensely.  The second pass walks the
	// pivot rows in reverse eliminating nonzeros above the pivots (i.e. above
	// the diagonal).

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

		pval := A.At(piv, j)
		bval := b[piv]
		for _, i := range A.NonzeroRows(j) {
			if i != piv && i > j {
				mult := -A.At(i, j) / pval
				//fmt.Printf("   pivot times %v plus row %v\n", mult, i)
				RowCombination(A, piv, i, mult)
				b[i] += bval * mult
			} else {
				//fmt.Printf("    skipping row %v which is the pivot\n", i)
			}
		}
		//fmt.Printf("after:\n%.2v\n", mat64.Formatted(A))
	}

	// second pass
	for j := size - 1; j >= 0; j-- {
		piv := pivots[j]
		//fmt.Printf("selected row %v as pivot\n", piv)
		//fmt.Printf("Num nonzeros for col %v is %v\n", j, len(A.nonzeroRow[j]))

		pval := A.At(piv, j)
		bval := b[piv]
		for _, i := range A.NonzeroRows(j) {
			if i != piv && i < j {
				mult := -A.At(i, j) / pval
				//fmt.Printf("   pivot times %v plus row %v\n", mult, i)
				RowCombination(A, piv, i, mult)
				b[i] += bval * mult
			} else {
				//fmt.Printf("    skipping row %v which is the pivot\n", i)
			}
		}
		//fmt.Printf("after:\n%.2v\n", mat64.Formatted(A))
	}

	for j, i := range pivots {
		mult := 1 / A.At(i, j)
		RowMult(A, i, mult)
		b[i] *= mult
	}

	x := make([]float64, size)
	for i := range pivots {
		x[i] = b[pivots[i]]
	}

	return x
}
