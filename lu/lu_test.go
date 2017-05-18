package lu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestRCM_big(t *testing.T) {
	size := 35
	nfill := 6
	big := randSparse(size, nfill)
	mapping := RCM(big)
	permuted := big.Permute(mapping)
	t.Logf("original=\n% v\n", mat64.Formatted(big), mat64.DotByte(' '))
	t.Logf("permuted=\n% v\n", mat64.Formatted(permuted), mat64.DotByte(' '))
}

func TestRCM(t *testing.T) {
	var tests = []struct {
		size    int
		vals    []float64
		wantmap []int
	}{
		{
			size: 4,
			vals: []float64{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1,
			},
			wantmap: []int{3, 2, 1, 0},
		}, {
			size: 4,
			vals: []float64{
				0, 1, 0, 0,
				1, 0, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1,
			},
			wantmap: []int{3, 2, 1, 0},
		}, {
			size: 4,
			vals: []float64{
				1, 1, 0, 0,
				1, 1, 1, 0,
				0, 1, 1, 1,
				0, 0, 1, 1,
			},
			wantmap: []int{3, 2, 1, 0},
		}, {
			size: 4,
			vals: []float64{
				1, 1, 0, 1,
				1, 1, 0, 1,
				0, 0, 1, 1,
				1, 1, 1, 1,
			},
			wantmap: []int{1, 0, 3, 2},
		},
	}

	for i, test := range tests {
		A := NewSparse(test.size)
		for i := 0; i < test.size; i++ {
			for j := 0; j < test.size; j++ {
				A.Set(i, j, test.vals[i*test.size+j])
			}
		}

		got := RCM(A)

		failed := false
		for i := range got {
			if test.wantmap[i] != got[i] {
				failed = true
				break
			}
		}

		if failed {
			t.Errorf("test %v:  A=%v", i+1, mat64.Formatted(A, mat64.Prefix("                   ")))
			t.Errorf("    A_perm=%v", mat64.Formatted(A.Permute(got), mat64.Prefix("                   ")))
			t.Errorf("    mapping: got %v, want %v", got, test.wantmap)
		}
	}
}

func TestGaussJordan(t *testing.T) {
	var tests = []struct {
		vals []float64
		b    []float64
	}{
		{
			vals: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				0, 2, 0, 0,
				1, 0, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				.5, .5, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				0, 2, 0, 0,
				.5, .5, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		},
	}

	for i, test := range tests {
		size := len(test.b)

		A := NewSparse(size)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				A.Set(i, j, test.vals[i*size+j])
			}
		}
		refA := A.Clone()
		refb := make([]float64, len(test.b))
		copy(refb, test.b)

		var x mat64.Vector
		x.SolveVec(mat64.NewDense(size, size, test.vals), mat64.NewVector(size, test.b))
		want := x.RawVector().Data

		gotx := GaussJordan(A, test.b)

		failed := false
		for i := range want {
			if diff := math.Abs(gotx[i] - want[i]); diff > eps {
				failed = true
				break
			}
		}

		if failed {
			t.Errorf("test %v A=\n%v\nb=%v", i+1, mat64.Formatted(refA), refb)
			t.Errorf("    x: got %v, want %v", gotx, want)
		}
	}
}

func TestGaussJordanSym(t *testing.T) {
	var tests = []struct {
		vals []float64
		b    []float64
	}{
		{
			vals: []float64{
				1, 1, 1, 1,
				1, 2, 0, 0,
				1, 0, 3, 0,
				1, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				1, 1, 1, 1,
				0, 1, 3, 0,
				0, 1, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				1, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				.5, .5, 0, 0,
				.5, 2, 1, 0,
				0, 1, 3, 1,
				0, 0, 1, 4,
			},
			b: []float64{1, 2, 3, 4},
		}, {
			vals: []float64{
				1, .5, 1, 1,
				.5, .5, 0, 1,
				1, 0, 3, 0,
				1, 1, 0, 1,
			},
			b: []float64{1, 2, 3, 4},
		},
	}

	for i, test := range tests {
		t.Logf("test %v:", i+1)
		size := len(test.b)

		A := NewSparse(size)
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				A.Set(i, j, test.vals[i*size+j])
			}
		}
		refA := A.Clone()
		refb := make([]float64, len(test.b))
		copy(refb, test.b)

		var x mat64.Vector
		x.SolveVec(mat64.NewDense(size, size, test.vals), mat64.NewVector(size, test.b))
		want := x.RawVector().Data

		gotx := GaussJordanSym(A, test.b)

		failed := false
		for i := range want {
			if diff := math.Abs(gotx[i] - want[i]); diff > eps {
				failed = true
				break
			}
		}

		if failed {
			t.Errorf("test %v FAILED:", i+1)
			t.Errorf("        A=\n%v\nb=%v", i+1, mat64.Formatted(refA), refb)
			t.Errorf("    x: got %v, want %v", gotx, want)
		} else {
			t.Logf("    PASSED")
		}
	}
}

func BenchmarkGonumLU(b *testing.B) {
	size := 5000
	nfill := 4 // number filled entries per row

	s := mat64.NewDense(size, size, nil)
	for i := 0; i < size; i++ {
		s.Set(i, i, 10)
	}

	for i := 0; i < size; i++ {
		for n := 0; n < nfill/2; n++ {
			j := rand.Intn(size)
			if i == j {
				n--
				continue
			}
			v := rand.Float64()
			s.Set(i, j, v)
			s.Set(j, i, v)
		}
	}

	f := mat64.NewVector(size, nil)
	for i := 0; i < size; i++ {
		f.SetVec(i, 1)
	}

	b.ResetTimer()
	var x mat64.Vector
	for i := 0; i < b.N; i++ {
		x.SolveVec(s, f)
	}
}

func BenchmarkGaussJordanSym(b *testing.B) {
	size := 5000
	nfill := 4 // number filled entries per row

	s := randSparse(size, nfill)

	f := make([]float64, size)
	for i := range f {
		f[i] = 1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GaussJordanSym(s, f)
	}
}

func randSparse(size, fillPerRow int) *Sparse {
	s := NewSparse(size)
	for i := 0; i < size; i++ {
		s.Set(i, i, 9)
	}

	for i := 0; i < size; i++ {
		nfill := fillPerRow / 2
		if i%7 == 0 {
			nfill = fillPerRow / 3
		}
		for n := 0; n < nfill; n++ {
			j := rand.Intn(size)
			if i == j {
				continue
			}
			v := rand.Float64()
			s.Set(i, j, v)
			s.Set(j, i, v)
		}
	}
	return s
}
