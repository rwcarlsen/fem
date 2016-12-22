package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestParallelSolve(t *testing.T) {
	A := mat64.NewDense(3, 3, []float64{1, 2, 3, 5, 3, 7, 8, 4, 1})
	b := mat64.NewDense(3, 1, []float64{1, 1, 1})

	x := ParallelSolve(A, b)
	var x2 mat64.Dense
	x2.Solve(A, b)
	fmt.Printf("  x=%.5v\n", mat64.Formatted(x, mat64.Prefix("    ")))
	fmt.Printf(" x2=%.5v\n", mat64.Formatted(&x2, mat64.Prefix("    ")))
}

func BenchmarkParallelSolve(b *testing.B) {
	b.Run("n=10,util", benchParallelSolve(ParallelSolve, 10))
	b.Run("n=100,util", benchParallelSolve(ParallelSolve, 100))
	b.Run("n=1000,util", benchParallelSolve(ParallelSolve, 1000))
	b.Run("n=2000,util", benchParallelSolve(ParallelSolve, 2000))
	//b.Run("n=10000,util", benchParallelSolve(ParallelSolve, 10000))

	//mat64Solve := func(A, b *mat64.Dense) *mat64.Dense {
	//	var x2 mat64.Dense
	//	x2.Solve(A, b)
	//	return &x2
	//}
	//b.Run("n=10,mat64", benchParallelSolve(mat64Solve, 10))
	//b.Run("n=100,mat64", benchParallelSolve(mat64Solve, 100))
	//b.Run("n=1000,mat64", benchParallelSolve(mat64Solve, 1000))
	//b.Run("n=10000,mat64", benchParallelSolve(mat64Solve, 10000))
}

func benchParallelSolve(solveFunc func(a, b *mat64.Dense) *mat64.Dense, n int) func(b *testing.B) {
	return func(b *testing.B) {
		xs := make([]float64, n*n)
		for i := range xs {
			xs[i] = rand.Float64()
		}

		bs := make([]float64, n)
		for i := range bs {
			bs[i] = rand.Float64()
		}

		A := mat64.NewDense(n, n, xs)
		bb := mat64.NewDense(n, 1, bs)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = solveFunc(A, bb)
		}
	}
}

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
			gotx, goty := trans.Transform(x, y)

			if wantx != gotx || wanty != goty {
				t.Errorf("    ERR transform[%v,%v] = got %v,%v want %v,%v", x, y, gotx, goty, wantx, wanty)
				continue
			} else {
				t.Logf("        transform[%v,%v] = %v,%v", x, y, gotx, goty)
			}

			revx, revy := trans.Reverse(gotx, goty)
			if math.Abs(revx-x) > tol {
				t.Errorf("            ERR reverse transform failed: got x,y = %v,%v", revx, revy)
				continue
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
