package main

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestAffineTransform(t *testing.T) {
	var tests = []struct {
		srcNodes [][]float64
		dstNodes [][]float64
		points   [][]float64
		want     [][]float64
	}{
		{
			// identity transformation
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			want:     [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
		}, {
			// 90 degree rotation
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{0, 1}, {0, 0}, {1, 0}, {1, 1}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			want:     [][]float64{{0, 1}, {0, 0}, {1, 0}, {1, 1}, {.25, 1}, {.25, .75}},
		}, {
			// arbitrary projective transform
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{0, 0}, {2, 0}, {1, 3}, {0, 2}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			want:     [][]float64{{0, 0}, {2, 0}, {1, 3}, {0, 2}, {.0, 2. / 3}, {.25, .75}},
		},
	}

	for i, test := range tests {
		t.Logf("test %v (src=%v, dst=%v):", i+1, test.srcNodes, test.dstNodes)
		trans, err := NewAffineTransform(test.srcNodes, test.dstNodes)
		if err != nil {
			t.Error(err)
			continue
		}
		for j, point := range test.points {
			want := test.want[j]
			x, y := point[0], point[1]
			wantx, wanty := want[0], want[1]
			gotx, goty := trans.Apply(x, y)

			if wantx != gotx || wanty != goty {
				t.Errorf("    ERR transform[%v,%v] = got %v,%v want %v,%v", x, y, gotx, goty, wantx, wanty)
			} else {
				t.Logf("        transform[%v,%v] = %v,%v", x, y, gotx, goty)
			}
		}
		if t.Failed() {
			t.Logf("    transform matrix:\n              %v", mat64.Formatted(trans.srcToDst, mat64.Prefix("              ")))
		}
	}
}
