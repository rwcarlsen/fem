package sparse

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

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
	Pattern  Matrix
	MaxWidth int
}

func (r RestrictByPattern) Set(i, j int, val float64) {
	diff := i - j
	if diff < 0 {
		diff *= -1
	}
	if r.Pattern.At(i, j) != 0 || diff <= r.MaxWidth {
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
	// Index should be called after any mutations to the nonzeros before using the Sweep[Row/Col]
	// methods.
	Index()
	SweepRow(r int) []Nonzero
	SweepCol(c int) []Nonzero
}

type Nonzero struct {
	I   int
	Val float64
}

type Sparse struct {
	// Tol specifies an absolute tolerance within which values are treated
	// as and set to zero.
	Tol float64
	// map[col]map[row]val
	nonzeroRow []map[int]float64
	// map[row]map[col]val
	nonzeroCol     []map[int]float64
	size           int
	nonzeroRowList [][]Nonzero
	nonzeroColList [][]Nonzero
}

// NewSparse creates a new square [size]x[size] sparse matrix representation with a default
// tolerance of 1e-6 for zero values.
func NewSparse(size int) *Sparse {
	return &Sparse{
		Tol:        1e-6,
		nonzeroRow: make([]map[int]float64, size),
		nonzeroCol: make([]map[int]float64, size),
		size:       size,
	}
}

func (m *Sparse) Index() {
	if m.nonzeroRowList == nil {
		m.nonzeroRowList = make([][]Nonzero, m.size)
		m.nonzeroColList = make([][]Nonzero, m.size)
	}
	for i := range m.nonzeroRowList {
		m.nonzeroRowList[i] = m.nonzeroRowList[i][:0]
		m.nonzeroColList[i] = m.nonzeroColList[i][:0]
	}
	for c, rows := range m.nonzeroRow {
		for r, val := range rows {
			m.nonzeroRowList[c] = append(m.nonzeroRowList[c], Nonzero{I: r, Val: val})
			m.nonzeroColList[r] = append(m.nonzeroColList[r], Nonzero{I: c, Val: val})
		}
	}
}
func (m *Sparse) SweepRow(r int) []Nonzero {
	if m.nonzeroRowList == nil {
		m.Index()
	}
	return m.nonzeroColList[r]
}
func (m *Sparse) SweepCol(c int) []Nonzero {
	if m.nonzeroRowList == nil {
		m.Index()
	}
	return m.nonzeroRowList[c]
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
	if math.Abs(v) < m.Tol {
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
		for _, nonzero := range m.SweepRow(i) {
			tot += b[nonzero.I] * nonzero.Val
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
