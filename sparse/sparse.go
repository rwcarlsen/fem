package sparse

import (
	"math"

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
