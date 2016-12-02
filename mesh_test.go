package main

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

const tol = 1e-6

func TestNode(t *testing.T) {

	tests := []struct {
		Xs      []float64
		Index   int
		SampleX float64
		Want    float64
	}{
		{Xs: []float64{0, 1}, Index: 0, SampleX: 0.0, Want: 1.0},
		{Xs: []float64{0, 1}, Index: 0, SampleX: 0.5, Want: 0.5},
		{Xs: []float64{0, 1}, Index: 0, SampleX: 1.0, Want: 0.0},
		{Xs: []float64{0, 1}, Index: 1, SampleX: 0.0, Want: 0.0},
		{Xs: []float64{0, 1}, Index: 1, SampleX: 0.5, Want: 0.5},
		{Xs: []float64{0, 1}, Index: 1, SampleX: 1.0, Want: 1.0},
		{Xs: []float64{0, 1, 2}, Index: 0, SampleX: 0.0, Want: 1.0},
		{Xs: []float64{0, 1, 2}, Index: 0, SampleX: 1.0, Want: 0.0},
		{Xs: []float64{0, 1, 2}, Index: 0, SampleX: 2.0, Want: 0.0},
		{Xs: []float64{0, 1, 2}, Index: 1, SampleX: 0.0, Want: 0.0},
		{Xs: []float64{0, 1, 2}, Index: 1, SampleX: 1.0, Want: 1.0},
		{Xs: []float64{0, 1, 2}, Index: 1, SampleX: 2.0, Want: 0.0},
		{Xs: []float64{0, 1, 2}, Index: 2, SampleX: 0.0, Want: 0.0},
		{Xs: []float64{0, 1, 2}, Index: 2, SampleX: 1.0, Want: 0.0},
		{Xs: []float64{0, 1, 2}, Index: 2, SampleX: 2.0, Want: 1.0},
	}

	for i, test := range tests {
		n := NewLagrangeNode(test.Index, test.Xs)
		y := n.Sample([]float64{test.SampleX})
		if y != test.Want {
			t.Errorf("FAIL test %v (xs=%v, i=%v): f(%v)=%v, want %v", i+1, test.Xs, test.Index, test.SampleX, y, test.Want)
		} else {
			t.Logf("     test %v (xs=%v, i=%v): f(%v)=%v", i+1, test.Xs, test.Index, test.SampleX, y)
		}
	}
}

func TestElement(t *testing.T) {

	tests := []struct {
		Xs      []float64
		Ys      []float64
		SampleX float64
		Want    float64
	}{
		{Xs: []float64{0, 1}, Ys: []float64{1, 2}, SampleX: 0.0, Want: 1.0},
		{Xs: []float64{0, 1}, Ys: []float64{1, 2}, SampleX: 0.5, Want: 1.5},
		{Xs: []float64{0, 1}, Ys: []float64{1, 2}, SampleX: 1.0, Want: 2.0},
		{Xs: []float64{0, 1, 2}, Ys: []float64{1, 2, 9}, SampleX: 0, Want: 1},
		{Xs: []float64{0, 1, 2}, Ys: []float64{1, 2, 9}, SampleX: 1, Want: 2},
		{Xs: []float64{0, 1, 2}, Ys: []float64{1, 2, 9}, SampleX: 2, Want: 9},
	}

	for i, test := range tests {
		elem := NewElementSimple1D(test.Xs)
		for i, n := range elem.Nodes() {
			n.Set(test.Ys[i], 1)
		}
		y, err := Interpolate(elem, []float64{test.SampleX})
		if err != nil {
			t.Errorf("FAIL test %v (xs=%v, ys=%v): %v", i+1, test.Xs, test.Ys, err)
		} else if y != test.Want {
			t.Errorf("FAIL test %v (xs=%v, ys=%v): f(%v)=%v, want %v", i+1, test.Xs, test.Ys, test.SampleX, y, test.Want)
		} else {
			t.Logf("     test %v (xs=%v, ys=%v): f(%v)=%v", i+1, test.Xs, test.Ys, test.SampleX, y)
		}
	}
}

func TestTransient(t *testing.T) {
	tests := []struct {
		Degree              int
		Xs                  []float64
		T0                  []float64
		K, S, Density, Area float64
		C                   []float64 // heat capacity
		dt                  float64
		nt                  int // number time steps
		Left                *Boundary
		Right               *Boundary
		Want                [][]float64
	}{
		{
			Degree:  2,
			Xs:      []float64{0, 2, 4},
			K:       2,
			S:       5,
			C:       []float64{3, 2, 2},
			dt:      1,
			nt:      10,
			Density: 1,
			Area:    0.1,
			Left:    DirichletBC(0),
			Right:   DirichletBC(190),
			Want: [][]float64{
				{0, 145, 190},
			},
		},
	}

	for i, test := range tests {
		t.Logf("test %v:", i+1)
		mesh, err := NewMeshSimple1D(test.Xs, test.Degree)
		if err != nil {
			t.Errorf("    FAIL: %v", err)
		}

		hc := &HeatConduction{
			X:       test.Xs,
			K:       ConstVal(test.K),
			S:       ConstVal(test.S),
			C:       &SecVals{test.Xs, test.C},
			Density: ConstVal(test.Density),
			Area:    test.Area,
			Left:    test.Left,
			Right:   test.Right,
		}
		t.Logf("\n            K=%v", mat64.Formatted(mesh.StiffnessMatrix(hc, 1.0), mat64.Prefix("              ")))
		t.Logf("\n            C=%v", mat64.Formatted(mesh.TimeDerivMatrix(hc), mat64.Prefix("              ")))
		t.Logf("\n            f=%v", mat64.Formatted(mesh.ForceMatrix(hc, 1.0), mat64.Prefix("              ")))

		time := 0.0
		mesh.InitU(test.T0)
		for j := 0; j < test.nt; j++ {
			time += test.dt
			err = mesh.SolveStep(hc, test.dt)
			if err != nil {
				t.Errorf("    FAIL: %v", err)
				continue
			}

			want := test.Want[j]
			for k, x := range test.Xs {
				y, err := mesh.Interpolate([]float64{x})
				if err != nil {
					t.Errorf("    FAIL f(t=%v,x=%v)=??: %v", time, x, err)
				} else if math.Abs(y-want[k]) > tol {
					t.Errorf("    FAIL f(t=%v,x=%v)=%v, want %v", time, x, y, want[k])
				} else {
					t.Logf("         f(t=%v,x=%v)=%v", time, x, y)
				}
			}
		}
	}
}

func TestMeshSolve(t *testing.T) {
	tests := []struct {
		Degree     int
		Xs         []float64
		K, S, Area float64
		Left       *Boundary
		Right      *Boundary
		Want       []float64
	}{
		{
			Degree: 2,
			Xs:     []float64{0, 2, 4},
			K:      2,
			S:      5,
			Area:   0.1,
			Left:   DirichletBC(0),
			Right:  NeumannBC(5),
			Want:   []float64{0, 145, 190},
		},
	}

	for i, test := range tests {
		t.Logf("test %v:", i+1)
		mesh, err := NewMeshSimple1D(test.Xs, test.Degree)
		if err != nil {
			t.Errorf("    FAIL: %v", err)
		}

		hc := &HeatConduction{
			X:     test.Xs,
			K:     ConstVal(test.K),
			S:     ConstVal(test.S),
			Area:  test.Area,
			Left:  test.Left,
			Right: test.Right,
		}
		t.Logf("\n            k=%v", mat64.Formatted(mesh.StiffnessMatrix(hc, DefaultPenalty), mat64.Prefix("              ")))
		t.Logf("\n            f=%v", mat64.Formatted(mesh.ForceMatrix(hc, DefaultPenalty), mat64.Prefix("              ")))

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
