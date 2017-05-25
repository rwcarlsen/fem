package sparse

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

const eps = 1e-9

type RestrictByWidth struct {
	Matrix
	MaxWidth int
}

func (r RestrictByWidth) Set(i, j int, val float64) {
	diff := i - j
	if diff < 0 {
		diff *= -1
	}

	if diff <= r.MaxWidth {
		r.Matrix.Set(i, j, val)
	}
}

type RestrictByPattern struct {
	Matrix
	Pattern Matrix
}

func (r RestrictByPattern) Set(i, j int, val float64) {
	if r.Pattern.At(i, j) != 0 {
		r.Matrix.Set(i, j, val)
	}
}

type Matrix interface {
	Dims() (int, int)
	Set(i, j int, val float64)
	At(i, j int) float64
	NonzeroRows(col int) (rows map[int]float64)
	NonzeroCols(row int) (cols map[int]float64)
	T() mat64.Matrix
}

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

func Copy(dst, src Matrix) {
	size, _ := src.Dims()
	for i := 0; i < size; i++ {
		for j, val := range src.NonzeroCols(i) {
			dst.Set(i, j, val)
		}
	}
}

func (m *Sparse) Clone(b Matrix) {
	size, _ := b.Dims()
	for i := 0; i < size; i++ {
		for j, v := range b.NonzeroCols(i) {
			m.Set(i, j, v)
		}
	}
}

func (m *Sparse) NonzeroRows(col int) (rows map[int]float64) { return m.nonzeroRow[col] }
func (m *Sparse) NonzeroCols(row int) (cols map[int]float64) { return m.nonzeroCol[row] }

func (m *Sparse) T() mat64.Matrix     { return mat64.Transpose{m} }
func (m *Sparse) Dims() (int, int)    { return m.size, m.size }
func (m *Sparse) At(i, j int) float64 { return m.nonzeroCol[i][j] }
func (m *Sparse) Set(i, j int, v float64) {
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

func Mul(m Matrix, b []float64) []float64 {
	size := len(b)
	result := make([]float64, len(b))
	for i := 0; i < size; i++ {
		tot := 0.0
		for j, val := range m.NonzeroCols(i) {
			tot += b[j] * val
		}
		result[i] = tot
	}
	return result
}

func RowCombination(m Matrix, pivrow, dstrow int, mult float64) {
	for col, aij := range m.NonzeroCols(pivrow) {
		m.Set(dstrow, col, m.At(dstrow, col)+aij*mult)
	}
}

func RowMult(m Matrix, row int, mult float64) {
	for col, val := range m.NonzeroCols(row) {
		m.Set(row, col, val*mult)
	}
}

// Permute maps i and j indices to new i and j values idendified by the given
// mapping.  Values stored in src.At(i,j) are stored into dst.At(mapping[i],
// mapping[j]) The permuted matrix is stored in dst overwriting values stored
// there and the original remains unmodified.
func Permute(dst, src Matrix, mapping []int) {
	size, _ := src.Dims()
	for i := 0; i < size; i++ {
		for j, val := range src.NonzeroCols(i) {
			dst.Set(mapping[i], mapping[j], val)
		}
	}
}
