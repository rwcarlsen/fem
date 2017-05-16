package lu

import (
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

const eps = 1e-10

func (s *Sparse) NonzeroRows(col int) (row []int) { return s.nonzeroRow[col] }
func (s *Sparse) NonzeroCols(row int) (col []int) { return s.nonzeroCol[row] }

func (s *Sparse) T() mat64.Matrix     { return mat64.Transpose{s} }
func (s *Sparse) Dims() (int, int)    { return s.size, s.size }
func (s *Sparse) At(i, j int) float64 { return s.data[index{i, j}] }
func (s *Sparse) Set(i, j int, v float64) {
	if math.Abs(v) < eps {
		delete(s.data, index{i, j})
		for ii, col := range s.nonzeroCol[i] {
			if col == j {
				s.nonzeroCol = append(s.nonzeroCol[:ii], s.nonzeroCol[ii+1:]...)
				break
			}
		}
		for ii, row := range s.nonzeroRow[j] {
			if row == i {
				s.nonzeroRow = append(s.nonzeroRow[:ii], s.nonzeroRow[ii+1:]...)
				break
			}
		}
		return
	}

	s.data[index{i, j}] = math.Abs(v)
	s.nonzeroCol[i] = append(s.nonzeroCol[i], j)
	s.nonzeroRow[j] = append(s.nonzeroRow[j], i)
}

func RowCombination(s *Sparse, rowsrc, rowdst int, mult float64) {
	cols := s.NonzeroCols(rowsrc)
	for _, col := range cols {
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
	for j := 0; j < size; j++ {
		// find a first row with a nonzero entry in column i on or below diagonal
		// to use as the pivot row.
		piv := -1
		for i := 0; i < size; i++ {
			if A.At(i, j) != 0 {
				piv = i
				break
			}
		}

		pval := A.At(piv, j)
		bval := b[piv]
		for _, i := range A.NonzeroRows(j) {
			if i != piv {
				mult := -pval / A.At(i, j)
				RowCombination(A, piv, i, mult)
				b[i] += bval * mult
			}
		}
	}
	for i := 0; i < size; i++ {
		col := A.NonzeroCols(i)[0]
		mult := 1 / A.At(i, col)
		RowMult(A, i, mult)
		b[col] *= mult
	}

	return b
}
