package sparse

import (
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func randSparse(size, fillPerRow int, off float64) Matrix {
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

			v := off
			if v == 0 {
				v = rand.Float64()
			}
			s.Set(i, j, v)
			s.Set(j, i, v)
		}
	}
	return s
}

func TestRCM_big(t *testing.T) {
	size := 35
	nfill := 6
	big := randSparse(size, nfill, 8)
	mapping := RCM(big)
	permuted := NewSparse(size)
	Permute(permuted, big, mapping)
	t.Logf("original=\n% v\n", mat64.Formatted(big))
	t.Logf("permuted=\n% v\n", mat64.Formatted(permuted))
}

func TestRCM(t *testing.T) {
	return
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
			wantmap: []int{1, 3, 2, 0},
		}, {
			size: 4,
			vals: []float64{
				1, 1, 0, 1,
				1, 1, 0, 1,
				0, 0, 1, 1,
				1, 1, 1, 1,
			},
			wantmap: []int{2, 1, 0, 3},
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
			permuted := NewSparse(test.size)
			Permute(permuted, A, got)
			t.Errorf("test %v:  A=%v", i+1, mat64.Formatted(A, mat64.Prefix("                   ")))
			t.Errorf("    A_perm=%v", mat64.Formatted(permuted, mat64.Prefix("                   ")))
			t.Errorf("    mapping: got %v, want %v", got, test.wantmap)
		}
	}
}
