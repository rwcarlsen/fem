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
			t.Errorf("FAIL case %v (xs=%v, i=%v): f(%v)=%v, want %v", i+1, test.Xs, test.Index, test.SampleX, y, test.Want)
		} else {
			t.Logf("     case %v (xs=%v, i=%v): f(%v)=%v", i+1, test.Xs, test.Index, test.SampleX, y)
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
		t.Logf("case %v x1,x2,y1,y2=%v,%v,%v,%v:", i+1, n.X1, n.X2, n.Y1, n.Y2)
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

func TestBilinQuadNode(t *testing.T) {
	tests := []struct {
		X1, Y1  float64
		X2, Y2  float64
		X3, Y3  float64
		X4, Y4  float64
		Index   int
		SampleX [][]float64
		Want    []float64
	}{
		{
			X1: 0, Y1: 0,
			X2: 1, Y2: 0,
			X3: 1, Y3: 1,
			X4: 0, Y4: 1,
			Index:   1,
			SampleX: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{1.0, 0, 0, 0, 0.25, 0.5, 0},
		}, {
			X1: 0, Y1: 0,
			X2: 1, Y2: 0,
			X3: 1, Y3: 1,
			X4: 0, Y4: 1,
			Index:   2,
			SampleX: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{0, 1.0, 0, 0, 0.25, 0, 0.5},
		}, {
			X1: 0, Y1: 0,
			X2: 1, Y2: 0,
			X3: 1, Y3: 1,
			X4: 0, Y4: 1,
			Index:   3,
			SampleX: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{0, 0, 1.0, 0, 0.25, 0, 0.5},
		}, {
			X1: 0, Y1: 0,
			X2: 1, Y2: 0,
			X3: 1, Y3: 1,
			X4: 0, Y4: 1,
			Index:   4,
			SampleX: [][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}, {0, 0.5}, {1, 0.5}},
			Want:    []float64{0, 0, 0, 1.0, 0.25, 0.5, 0},
		},
	}

	for i, test := range tests {
		ts := test
		t.Logf("case %v p1,p2,p3,p4=%v,%v;%v,%v;%v,%v;%v,%v:", i+1, ts.X1, ts.Y1, ts.X2, ts.Y2, ts.X3, ts.Y3, ts.X4, ts.Y4)

		var n *BilinQuadNode
		var err error
		switch test.Index {
		case 1:
			n, err = NewBilinQuadNode(ts.X1, ts.Y1, ts.X2, ts.Y2, ts.X3, ts.Y3, ts.X4, ts.Y4)
		case 2:
			n, err = NewBilinQuadNode(ts.X2, ts.Y2, ts.X3, ts.Y3, ts.X4, ts.Y4, ts.X1, ts.Y1)
		case 3:
			n, err = NewBilinQuadNode(ts.X3, ts.Y3, ts.X4, ts.Y4, ts.X1, ts.Y1, ts.X2, ts.Y2)
		case 4:
			n, err = NewBilinQuadNode(ts.X4, ts.Y4, ts.X1, ts.Y1, ts.X2, ts.Y2, ts.X3, ts.Y3)
		}
		if err != nil {
			t.Error(err)
			continue
		}
		for j, point := range test.SampleX {
			v := n.Sample(point)
			e, n := n.Transform.Reverse(point[0], point[1])
			if v != test.Want[j] {
				t.Errorf("    FAIL sample %v: f(%v) = %v, want %v  (e,n = %v,%v)", j+1, point, v, test.Want[j], e, n)
			} else {
				t.Logf("         sample %v: f(%v) = %v", j+1, point, v)
			}
		}
	}
}