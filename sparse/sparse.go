package sparse

import (
	"fmt"
	"runtime"
	"sort"
	"sync"

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
	T() mat64.Matrix
	SweepRow(r int) []*Nonzero
	SweepCol(c int) []*Nonzero
}

type Nonzero struct {
	I, J int
	Val  float64
}

type Sparse struct {
	// Tol specifies an absolute tolerance within which values are treated
	// as and set to zero.
	Tol          float64
	size         int
	nonzeroByCol [][]*Nonzero
	nonzeroByRow [][]*Nonzero
	searchslice  []*Nonzero
	searchi      int
	searchval    float64
}

// NewSparse creates a new square [size]x[size] sparse matrix representation with a default
// tolerance of 1e-6 for zero values.
func NewSparse(size int) *Sparse {
	return &Sparse{
		Tol:          1e-6,
		nonzeroByRow: make([][]*Nonzero, size),
		nonzeroByCol: make([][]*Nonzero, size),
		size:         size,
	}
}

func (m *Sparse) SweepRow(r int) []*Nonzero { return m.nonzeroByRow[r] }
func (m *Sparse) SweepCol(c int) []*Nonzero { return m.nonzeroByCol[c] }

func Copy(dst, src Matrix) {
	size, _ := src.Dims()
	for i := 0; i < size; i++ {
		for _, nonzero := range src.SweepRow(i) {
			dst.Set(i, nonzero.J, nonzero.Val)
		}
	}
}

func (m *Sparse) T() mat64.Matrix  { return mat64.Transpose{m} }
func (m *Sparse) Dims() (int, int) { return m.size, m.size }
func (m *Sparse) At(i, j int) float64 {
	m.searchslice = m.nonzeroByRow[i]
	m.searchi = j
	jindex := sort.Search(len(m.searchslice), m.atsearch)
	if jindex < len(m.searchslice) && m.searchslice[jindex].J == j {
		return m.searchslice[jindex].Val
	}
	return 0
}

// TODO: add method for cleaning out accumulated nonzero entries that store a 0.0 (or small) value

func (m *Sparse) setsearchi(i int) bool {
	nonzero := m.searchslice[i]
	//fmt.Println("len(searchslize)=", len(m.searchslice), ", index=", i)
	if nonzero.I == m.searchi {
		nonzero.Val = m.searchval
		return true
	}
	return nonzero.I >= m.searchi
}

func (m *Sparse) setsearchj(i int) bool {
	nonzero := m.searchslice[i]
	if nonzero.J == m.searchi {
		nonzero.Val = m.searchval
		return true
	}
	return nonzero.J >= m.searchi
}

func (m *Sparse) atsearch(j int) bool { return m.searchslice[j].J >= m.searchi }

func (m *Sparse) Set(i, j int, v float64) {
	defer func() {
		fmt.Printf("Set(%v,%v)=%v\n", i, j, v)
		//for _, nonzeros := range m.nonzeroByRow {
		//	for _, nonzero := range nonzeros {
		//		fmt.Printf("    nonzero %v\n", nonzero)
		//	}
		//}
	}()
	m.searchval = v

	m.searchslice = m.nonzeroByRow[i]
	m.searchi = j
	byrowindex := sort.Search(len(m.searchslice), m.setsearchj)
	if byrowindex < len(m.searchslice) && m.searchslice[byrowindex].J == j {
		// we already found and updated existing entry - so just return early
		return
	}
	m.searchslice = m.nonzeroByCol[j]
	m.searchi = i
	bycolindex := sort.Search(len(m.searchslice), m.setsearchi)

	// insert new nonzero
	nonzero := &Nonzero{I: i, J: j, Val: v}

	lena := len(m.nonzeroByRow[i])
	fmt.Printf("lena=%v, byrowindex=%v\n", lena, byrowindex)
	m.nonzeroByRow[i] = append(m.nonzeroByRow[i], nil)
	byrow := m.nonzeroByRow[i]
	for a := lena - 1; a >= byrowindex; a-- {
		fmt.Printf("  sliding col %v to col %v\n", a, a+1)
		byrow[a+1] = byrow[a] // slide entries back one
	}
	byrow[byrowindex] = nonzero // insert new entry in now empty slot

	lenb := len(m.nonzeroByCol[j])
	m.nonzeroByCol[j] = append(m.nonzeroByCol[j], nil)
	bycol := m.nonzeroByCol[j]
	for b := lenb - 1; b >= bycolindex; b-- {
		bycol[b+1] = bycol[b] // slide entries back one
	}
	bycol[bycolindex] = nonzero // insert new entry in now empty slot
}

func mul(m Matrix, b, result []float64, start, end int) {
	for i := start; i < end; i++ {
		tot := 0.0
		for _, nonzero := range m.SweepRow(i) {
			tot += b[nonzero.I] * nonzero.Val
		}
		result[i] = tot
	}
}

func Mul(m Matrix, b []float64) []float64 {
	size := len(b)
	result := make([]float64, size)

	nworkers := runtime.NumCPU()
	blocksize := size / (nworkers - 1)

	if size < 5000 { // run serial for small cases
		mul(m, b, result, 0, size)
		return result
	}

	var wg sync.WaitGroup
	wg.Add(nworkers)
	for i := 0; i < nworkers; i++ {
		start := i * blocksize
		end := start + blocksize
		if end > size {
			end = size
		}

		go func(start, end int) {
			mul(m, b, result, start, end)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
	return result
}

func RowCombination(m Matrix, pivrow, dstrow int, mult float64) {
	for _, nonzero := range m.SweepRow(pivrow) {
		m.Set(dstrow, nonzero.J, m.At(dstrow, nonzero.J)+nonzero.Val*mult)
	}
}

func RowMult(m Matrix, row int, mult float64) {
	for _, nonzero := range m.SweepRow(row) {
		m.Set(row, nonzero.J, nonzero.Val*mult)
	}
}

// Permute maps i and j indices to new i and j values idendified by the given
// mapping.  Values stored in src.At(i,j) are stored into dst.At(mapping[i],
// mapping[j]) The permuted matrix is stored in dst overwriting values stored
// there and the original remains unmodified.
func Permute(dst, src Matrix, mapping []int) {
	size, _ := src.Dims()
	for i := 0; i < size; i++ {
		for _, nonzero := range src.SweepRow(i) {
			dst.Set(mapping[i], mapping[nonzero.J], nonzero.Val)
		}
	}
}
