package lu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func BenchmarkGaussJordan(b *testing.B) {
	size := 1000
	nfill := 3 // number filled entries per row

	s := NewSparse(size)
	for i := 0; i < size; i++ {
		s.Set(i, i, 10)
	}

	for i := 0; i < size; i++ {
		for n := 0; n < nfill; n++ {
			j := rand.Intn(size)
			if i == j {
				n--
				continue
			}
			s.Set(i, j, rand.Float64())
		}
	}

	f := make([]float64, size)
	for i := range f {
		f[i] = 1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GaussJordan(s, f)
	}
}

func TestGaussJordan(t *testing.T) {
	var tests = []struct {
		vals  []float64
		b     []float64
		wantx []float64
	}{
		{
			vals: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b:     []float64{1, 2, 3, 4},
			wantx: []float64{1, 1, 1, 1},
		}, {
			vals: []float64{
				0, 2, 0, 0,
				1, 0, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b:     []float64{1, 2, 3, 4},
			wantx: []float64{0.5, 2, 1, 1},
		}, {
			vals: []float64{
				1, 1, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b:     []float64{1, 2, 3, 4},
			wantx: []float64{0, 1, 1, 1},
		}, {
			vals: []float64{
				.5, .5, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b:     []float64{1, 2, 3, 4},
			wantx: []float64{1, 1, 1, 1},
		}, {
			vals: []float64{
				0, 2, 0, 0,
				.5, .5, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b:     []float64{1, 2, 3, 4},
			wantx: []float64{3.5, .5, 1, 1},
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
