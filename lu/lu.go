package lu

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

type index struct {
	row, col int
}

type Sparse struct {
	// map[row][]col
	data       map[index]float64
	nonzeroRow [][]int
	nonzeroCol [][]int
	size       int
}

func NewSparse(size int) *Sparse {
	return &Sparse{
		data:       make(map[index]float64),
		nonzeroRow: make([][]int, size),
		nonzeroCol: make([][]int, size),
		size:       size,
	}
}

const eps = 1e-6

func (s *Sparse) NonzeroRows(col int) (row []int) { return s.nonzeroRow[col] }
func (s *Sparse) NonzeroCols(row int) (col []int) { return s.nonzeroCol[row] }

func (s *Sparse) Clone() *Sparse {
	clone := NewSparse(s.size)
	for k, v := range s.data {
		clone.Set(k.row, k.col, v)
	}
	return clone
}

func (s *Sparse) T() mat64.Matrix     { return mat64.Transpose{s} }
func (s *Sparse) Dims() (int, int)    { return s.size, s.size }
func (s *Sparse) At(i, j int) float64 { return s.data[index{i, j}] }
func (s *Sparse) Set(i, j int, v float64) {
	if math.Abs(v) < eps {
		//fmt.Printf("        Set(%v,%v) to zero\n", i, j)
		delete(s.data, index{i, j})
		fmt.Printf("len nonzerocols before = %v\n", len(s.nonzeroCol[i]))
		fmt.Printf("len nonzerorows before = %v\n", len(s.nonzeroRow[i]))
		for ii, col := range s.nonzeroCol[i] {
			if col == j {
				s.nonzeroCol[i] = append([]int{}, append(s.nonzeroCol[i][:ii], s.nonzeroCol[i][ii+1:]...)...)
				break
			}
		}
		for ii, row := range s.nonzeroRow[j] {
			if row == i {
				s.nonzeroRow[j] = append([]int{}, append(s.nonzeroRow[j][:ii], s.nonzeroRow[j][ii+1:]...)...)
				break
			}
		}
		fmt.Printf("len nonzerocols after = %v\n", len(s.nonzeroCol[i]))
		fmt.Printf("len nonzerorows after = %v\n", len(s.nonzeroRow[i]))
		return
	}

	if _, ok := s.data[index{i, j}]; !ok {
		//fmt.Printf("        Set(%v,%v)=%v and added nonzero col+row\n", i, j, v)
		s.nonzeroCol[i] = append(s.nonzeroCol[i], j)
		s.nonzeroRow[j] = append(s.nonzeroRow[j], i)
		//fmt.Printf("            nonzeroRows(%v)=%v\n", j, s.nonzeroRow[j])
		//fmt.Printf("            nonzeroCols(%v)=%v\n", i, s.nonzeroCol[i])
	}
	s.data[index{i, j}] = v
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
	for _, col := range cols {
		s.Set(row, col, s.At(row, col)*mult)
	}
}

func GaussJordan(A *Sparse, b []float64) []float64 {
	size, _ := A.Dims()
	donerows := make(map[int]bool, size)
	pivots := make([]int, size)
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
		fmt.Printf("selected row %v as pivot\n", piv)
		fmt.Printf("Num nonzeros for col %v is %v\n", j, len(A.nonzeroRow[j]))
		pivots[j] = piv
		donerows[piv] = true

		pval := A.At(piv, j)
		bval := b[piv]
		for _, i := range A.NonzeroRows(j) {
			if i != piv {
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
