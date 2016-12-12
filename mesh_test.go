package main

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

const tol = 1e-6

func TestMeshSolve(t *testing.T) {
	tests := []struct {
		Degree   int
		Xs       []float64
		K, S     float64
		Left     BoundaryType
		LeftVal  float64
		Right    BoundaryType
		RightVal float64
		Want     []float64
	}{
		{
			Degree:   2,
			Xs:       []float64{0, 2, 4},
			K:        2,
			S:        50,
			Left:     Dirichlet,
			LeftVal:  0,
			Right:    Neumann,
			RightVal: 5,
			Want:     []float64{0, 145, 190},
		}, {
			Degree:   3,
			Xs:       []float64{0, 1, 2, 3, 4},
			K:        2,
			S:        50,
			Left:     Dirichlet,
			LeftVal:  0,
			Right:    Neumann,
			RightVal: 5,
			Want:     []float64{0, 85, 145, 180, 190},
		},
	}

	for i, test := range tests {
		t.Logf("test %v:", i+1)
		mesh, err := NewMeshSimple1D(test.Xs, test.Degree)
		if err != nil {
			t.Errorf("    FAIL: %v", err)
		}

		hc := &HeatConduction{
			X:        test.Xs,
			K:        ConstVal(test.K),
			S:        ConstVal(test.S),
			Boundary: NewBoundary1D(test.Xs, test.LeftVal, test.RightVal, test.Left, test.Right),
		}
		t.Logf("\n            k=%v", mat64.Formatted(mesh.StiffnessMatrix(hc), mat64.Prefix("              ")))
		t.Logf("\n            f=%v", mat64.Formatted(mesh.ForceMatrix(hc), mat64.Prefix("              ")))

		err = mesh.Solve(hc)
		if err != nil {
			t.Errorf("    FAIL: %v", err)
			continue
		}

		for i, x := range test.Xs {
			y, err := mesh.Interpolate([]float64{x})
			if err != nil {
				t.Errorf("    FAIL f(%v)=??: %v", x, err)
			} else if math.Abs(y-test.Want[i]) > tol {
				t.Errorf("    FAIL f(%v)=%v, want %v", x, y, test.Want[i])
			} else {
				t.Logf("         f(%v)=%v", x, y)
			}
		}
	}
}
