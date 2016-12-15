package main

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestProjectiveTransform(t *testing.T) {
	var tests = []struct {
		srcNodes [][]float64
		dstNodes [][]float64
		points   [][]float64
		want     [][]float64
		wantJ    [][]float64 // expected jacboians for each point
	}{
		{
			// identity transformation
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			want:     [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			wantJ: [][]float64{
				{1, 0, 0, 1},
				{1, 0, 0, 1},
				{1, 0, 0, 1},
				{1, 0, 0, 1},
				{1, 0, 0, 1},
				{1, 0, 0, 1},
			},
		}, {
			// 90 degree rotation
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{0, 1}, {0, 0}, {1, 0}, {1, 1}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			want:     [][]float64{{0, 1}, {0, 0}, {1, 0}, {1, 1}, {.25, 1}, {.25, .75}},
			wantJ: [][]float64{
				{0, 1, -1, 0},
				{0, 1, -1, 0},
				{0, 1, -1, 0},
				{0, 1, -1, 0},
				{0, 1, -1, 0},
				{0, 1, -1, 0},
			},
		}, {
			// double dst lengths and translate
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{1, 1}, {3, 1}, {3, 3}, {1, 3}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {.5, .5}},
			want:     [][]float64{{1, 1}, {3, 1}, {3, 3}, {1, 3}, {2, 2}},
			wantJ: [][]float64{
				{.5, 0, 0, .5},
				{.5, 0, 0, .5},
				{.5, 0, 0, .5},
				{.5, 0, 0, .5},
				{.5, 0, 0, .5},
				{.5, 0, 0, .5},
			},
		}, {
			// arbitrary projective transform
			srcNodes: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}},
			dstNodes: [][]float64{{0, 0}, {2, 0}, {1, 3}, {0, 2}},
			points:   [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, .25}, {.25, .25}},
			want:     [][]float64{{0, 0}, {2, 0}, {1, 3}, {0, 2}, {.0, 2. / 3}, {.25, .75}},
			wantJ: [][]float64{
				{1, 0, 0, 1. / 3},
				{.25, 0, 1. / 12, 1. / 6},
				{2. / 3, -2. / 3, 0.09523809523809523, 0.2857142857142857},
				{1.5, -.75, 0, .5},
				{1.125, -.140625, 0, .375},
				{8. / 9, -0.12698412698412698, 0.0365296803652968, 0.3287671232876712},
			},
		},
	}

	for i, test := range tests {
		t.Logf("test %v (src=%v, dst=%v):", i+1, test.srcNodes, test.dstNodes)
		trans, err := NewProjectiveTransform(test.srcNodes, test.dstNodes)
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

			gotJ := trans.Jacobian(x, y)
			gotJraw := gotJ.RawMatrix().Data
			J := test.wantJ[j]
			for k, jacval := range J {
				if math.Abs(gotJraw[k]-jacval) > tol {
					t.Errorf("           ERR wrong jacobian: got")
					t.Errorf("            %v", mat64.Formatted(gotJ, mat64.Prefix("                      ")))
					t.Errorf("             want:\n                      %v", mat64.Formatted(mat64.NewDense(2, 2, J), mat64.Prefix("                      ")))
					break
				}
			}

		}
		if t.Failed() {
			t.Logf("    transform matrix:\n              %v", mat64.Formatted(trans.srcToDst, mat64.Prefix("              ")))
		}
	}
}
