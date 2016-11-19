package main

import (
	"math"
	"testing"
)

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
		n := NewNode(test.Index, test.Xs)
		y := n.Sample(test.SampleX)
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
		elem := NewElementSimple(test.Xs)
		for i, n := range elem.Nodes {
			n.Val = test.Ys[i]
		}
		y := elem.Interpolate(test.SampleX)
		if y != test.Want {
			t.Errorf("FAIL test %v (xs=%v, ys=%v): f(%v)=%v, want %v", i+1, test.Xs, test.Ys, test.SampleX, y, test.Want)
		} else {
			t.Logf("     test %v (xs=%v, ys=%v): f(%v)=%v", i+1, test.Xs, test.Ys, test.SampleX, y)
		}
	}
}

func TestMeshSolve(t *testing.T) {
	tol := 1e-10

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
			Left:   EssentialBC(0),
			Right:  DirichletBC(5),
			Want:   []float64{0, 145, 190},
		},
	}

	for i, test := range tests {
		t.Logf("test %v:", i+1)
		mesh, err := NewMeshSimple(test.Xs, test.Degree)
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

		err = mesh.Solve(hc)
		if err != nil {
			t.Errorf("    FAIL: %v", err)
			continue
		}

		for i, x := range test.Xs {
			y := mesh.Interpolate(x)
			if math.Abs(y-test.Want[i]) > tol {
				t.Errorf("    FAIL f(%v)=%v, want %v", x, y, test.Want[i])
			} else {
				t.Logf("         f(%v)=%v", x, y)
			}
		}
	}
}
