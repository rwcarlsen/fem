package lu

import "testing"
import "github.com/gonum/matrix/mat64"

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
				0, 1, 0, 0,
				2, 0, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			b:     []float64{1, 2, 3, 4},
			wantx: []float64{0.5, 2, 1, 1},
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

		t.Logf("test %v A=\n%v\nb=\n%v", i+1, mat64.Formatted(A), test.b)

		gotx := GaussJordan(A, test.b)

		for i := range test.wantx {
			if gotx[i] != test.wantx[i] {
				t.Errorf("    x: got %v, want %v", gotx, test.wantx)
			}
		}
		if !t.Failed() {
			t.Logf("PASSED")
		}
	}
}
