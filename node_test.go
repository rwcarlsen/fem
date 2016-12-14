package main

import "testing"

func TestLagrangeNode(t *testing.T) {
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

func TestBilinRectNode(t *testing.T) {
	tests := []struct {
		X1, Y1  float64
		X2, Y2  float64
		Index   int
		SampleX [][]float64
		Want    []float64
	}{
		{
			X1: 0, X2: 1, Y1: 0, Y2: 1,
			Index:   0,
			SampleX: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{1.0, 0, 0, 0, 0.25, 0.5, 0},
		}, {
			X1: 0, X2: 1, Y1: 0, Y2: 1,
			Index:   1,
			SampleX: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{0, 1.0, 0, 0, 0.25, 0.5, 0},
		}, {
			X1: 0, X2: 1, Y1: 0, Y2: 1,
			Index:   2,
			SampleX: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{0, 0, 1.0, 0, 0.25, 0, 0.5},
		}, {
			X1: 0, X2: 1, Y1: 0, Y2: 1,
			Index:   3,
			SampleX: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{0, 0, 0, 1.0, 0.25, 0, 0.5},
		},
	}

	for i, test := range tests {
		var n *BilinRectNode
		switch test.Index {
		case 0:
			n = NewBilinRectNode(test.X1, test.Y1, test.X2, test.Y2)
		case 1:
			n = NewBilinRectNode(test.X1, test.Y2, test.X2, test.Y1)
		case 2:
			n = NewBilinRectNode(test.X2, test.Y1, test.X1, test.Y2)
		case 3:
			n = NewBilinRectNode(test.X2, test.Y2, test.X1, test.Y1)
		}
		t.Logf("test %v x1,x2,y1,y2=%v,%v,%v,%v:", i+1, n.X1, n.X2, n.Y1, n.Y2)
		for j, point := range test.SampleX {
			v := n.Sample(point)
			if v != test.Want[j] {
				t.Errorf("    FAIL sample %v: f(%v)=%v, want %v", j+1, point, v, test.Want[j])
			} else {
				t.Logf("         sample %v: f(%v)=%v", j+1, point, v)
			}
		}
	}
}
