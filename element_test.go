package main

import (
	"math"
	"testing"
)

func TestElementND_Interpolate(t *testing.T) {
	tests := []struct {
		Xs      [][]float64
		Ys      []float64
		SampleX float64
		Want    float64
	}{
		{Xs: [][]float64{[]float64{0}, []float64{1}}, Ys: []float64{1, 2}, SampleX: -1, Want: 1.0},
		{Xs: [][]float64{[]float64{0}, []float64{1}}, Ys: []float64{1, 2}, SampleX: 0, Want: 1.5},
		{Xs: [][]float64{[]float64{0}, []float64{1}}, Ys: []float64{1, 2}, SampleX: 1.0, Want: 2.0},
		{Xs: [][]float64{[]float64{0}, []float64{1}, []float64{2}}, Ys: []float64{1, 2, 9}, SampleX: -1, Want: 1},
		{Xs: [][]float64{[]float64{0}, []float64{1}, []float64{2}}, Ys: []float64{1, 2, 9}, SampleX: 0, Want: 2},
		{Xs: [][]float64{[]float64{0}, []float64{1}, []float64{2}}, Ys: []float64{1, 2, 9}, SampleX: 1, Want: 9},
	}

	for i, test := range tests {
		order := len(test.Xs) - 1
		elem := NewElementND(order, nil, nil, test.Xs...)
		for i, n := range elem.Nodes() {
			n.U = test.Ys[i]
			n.W = 1
		}
		y := Interpolate(elem, []float64{test.SampleX})
		if y != test.Want {
			t.Errorf("FAIL case %v (xs=%v, ys=%v): f(%v)=%v, want %v", i+1, test.Xs, test.Ys, test.SampleX, y, test.Want)
		} else {
			t.Logf("     case %v (xs=%v, ys=%v): f(%v)=%v", i+1, test.Xs, test.Ys, test.SampleX, y)
		}
	}
}

func TestElementND_Contains(t *testing.T) {
	tests := []struct {
		Xs     [][]float64
		Points [][]float64
		Inside []bool
	}{
		{ // unit square
			Xs: [][]float64{
				{0, 0},
				{1, 0},
				{1, 1},
				{0, 1},
			},
			Points: [][]float64{{0, 1}, {.5, .5}, {2, .5}, {.5, 2}},
			Inside: []bool{true, true, false, false},
		}, { // unregular quadrilateral
			Xs: [][]float64{
				{0, 0},
				{1, 0},
				{-1, 2},
				{1, 1},
			},
			Points: [][]float64{{0, 0}, {-.1, 0}, {-.5, 1}, {-.6, 1}, {0, 1.5}, {.1, 1.5}},
			Inside: []bool{true, false, true, false, true, false},
		},
	}

	for i, test := range tests {
		e := NewElementND(1, nil, nil, test.Xs...)
		t.Logf("case %v (nodes=%v):", i+1, test.Xs)
		for j, point := range test.Points {
			if e.Contains(point) != test.Inside[j] {
				t.Errorf("    FAIL contains point %v: got %v, want %v", point, e.Contains(point), test.Inside[j])
			} else {
				t.Logf("         contains point %v: got %v", point, e.Contains(point))
			}
		}
	}
}

func TestElementND_Coord(t *testing.T) {
	tests := []struct {
		x1, x2, x3, x4 []float64
		RefPoints      [][]float64
		RealPoints     [][]float64
	}{
		{ // unit square
			x1:         []float64{0, 0},
			x2:         []float64{1, 0},
			x3:         []float64{0, 1},
			x4:         []float64{1, 1},
			RefPoints:  [][]float64{{-1, -1}, {0, 0}, {0.5, -.5}},
			RealPoints: [][]float64{{0, 0}, {.5, .5}, {.75, .25}},
		}, { // rotate node order - but still proper order structure
			x1:         []float64{0, 1},
			x2:         []float64{1, 1},
			x3:         []float64{0, 0},
			x4:         []float64{1, 0},
			RefPoints:  [][]float64{{-1, -1}, {0, 0}, {0.5, -.5}},
			RealPoints: [][]float64{{0, 1}, {.5, .5}, {.75, .75}},
		}, { // unregular, translated quadrilateral
			x1:         []float64{2, 2},
			x2:         []float64{3, 2},
			x3:         []float64{1, 4},
			x4:         []float64{3, 3},
			RefPoints:  [][]float64{{-1, -1}, {0, 0}, {-1, 1}},
			RealPoints: [][]float64{{2, 2}, {2.25, 2.75}, {1, 4}},
		},
	}

	for i, test := range tests {
		e := NewElementND(1, nil, nil, test.x1, test.x2, test.x3, test.x4)
		t.Logf("case %v (x1=%v, x2=%v, x3=%v, x4=%v):", i+1, test.x1, test.x2, test.x3, test.x4)
		for j, refx := range test.RefPoints {
			want := test.RealPoints[j]
			if x := e.Coord(nil, refx, -1); x[0] != want[0] || x[1] != want[1] {
				t.Errorf("FAIL point %v (refx=%v) real coords: got %v, want %v", j+1, refx, x, want)
			} else {
				t.Logf("     point %v (refx=%v) real coords: got %v", j+1, refx, x)
			}
		}
	}
}

type constKernel float64

func (k constKernel) VolIntU(p *KernelParams) float64          { return float64(k) }
func (k constKernel) VolInt(p *KernelParams) float64           { return float64(k) }
func (k constKernel) BoundaryIntU(p *KernelParams) float64     { return float64(k) }
func (k constKernel) BoundaryInt(p *KernelParams) float64      { return float64(k) }
func (k constKernel) IsDirichlet(xs []float64) (bool, float64) { return false, 0 }

type testKernel float64

func (k testKernel) VolIntU(p *KernelParams) float64          { return float64(k) * p.U }
func (k testKernel) VolInt(p *KernelParams) float64           { return float64(k) * p.U }
func (k testKernel) BoundaryIntU(p *KernelParams) float64     { return float64(k) * p.U }
func (k testKernel) BoundaryInt(p *KernelParams) float64      { return float64(k) * p.U }
func (k testKernel) IsDirichlet(xs []float64) (bool, float64) { return false, 0 }

func TestElementND_IntegrateBoundary(t *testing.T) {
	const volume = 1
	const boundary = 2
	tests := []struct {
		X1 []float64
		X2 []float64
		X3 []float64
		X4 []float64
		V1 float64
		V2 float64
		V3 float64
		V4 float64

		Integral  int
		WantConst float64
		WantU1    float64
		WantU2    float64
		WantU3    float64
		WantU4    float64
	}{
		{ // scale+translate
			Integral: boundary,
			X1:       []float64{0, 0},
			X2:       []float64{1, 0},
			X3:       []float64{0, 1},
			X4:       []float64{1, 1},
			V1:       1, V2: 1, V3: 1, V4: 1,
			WantConst: 4,
			WantU1:    1,
			WantU2:    1,
			WantU3:    1,
			WantU4:    1,
		}, { // translate
			Integral: boundary,
			X1:       []float64{0, 0},
			X2:       []float64{2, 0},
			X3:       []float64{0, 2},
			X4:       []float64{2, 2},
			V1:       1, V2: 1, V3: 1, V4: 1,
			WantConst: 8,
			WantU1:    2,
			WantU2:    2,
			WantU3:    2,
			WantU4:    2,
		}, { // translate, scale, non-uniform node U values
			Integral: boundary,
			X1:       []float64{0, 0},
			X2:       []float64{2, 0},
			X3:       []float64{0, 2},
			X4:       []float64{2, 2},
			V1:       1, V2: 2, V3: 1, V4: 2,
			WantConst: 8,
			WantU1:    2,
			WantU2:    4,
			WantU3:    2,
			WantU4:    4,
		}, { // translate, scale, distort
			Integral: boundary,
			X1:       []float64{0, 0},
			X2:       []float64{2, 0},
			X3:       []float64{0, 2},
			X4:       []float64{1, 3},
			V1:       1, V2: 1, V3: 1, V4: 1,
			WantConst: 2 + math.Sqrt(10) + math.Sqrt(2) + 2,
			WantU1:    2,
			WantU2:    1 + .5*math.Sqrt(10),
			WantU3:    .5*math.Sqrt(2) + 1,
			WantU4:    .5*math.Sqrt(10) + .5*math.Sqrt(2),
		}, { // identity mapping
			Integral: volume,
			X1:       []float64{0, 0},
			X2:       []float64{2, 0},
			X3:       []float64{0, 2},
			X4:       []float64{2, 2},
			V1:       1, V2: 2, V3: 1, V4: 2,
			WantConst: 4,
			WantU1:    1,
			WantU2:    2,
			WantU3:    1,
			WantU4:    2,
		},
	}

	const eps = 1e-6

	for i, test := range tests {
		ts := test
		elem := NewElementND(1, nil, nil, ts.X1, ts.X2, ts.X3, ts.X4)

		nds := elem.Nodes()
		nds[0].Set(test.V1, 1)
		nds[1].Set(test.V2, 1)
		nds[2].Set(test.V3, 1)
		nds[3].Set(test.V4, 1)

		// test integration of constant value around the boundary
		var kconst constKernel = 1
		val := elem.integrateBoundary(kconst, 0, 0)
		if test.Integral == volume {
			val = elem.integrateVol(kconst, 0, 0)
		}
		t.Logf("case %v", i+1)
		if math.Abs(val-test.WantConst) > eps {
			t.Errorf("    FAIL const kernel: got %v want %v", val, test.WantConst)
		} else {
			t.Logf("         const kernel: got %v", val)
		}

		// test integration around the boundary separately for each node's shape function
		var ku testKernel = 1
		testu := func(j int, want float64) {
			val := elem.integrateBoundary(ku, 0, j)
			if test.Integral == volume {
				val = elem.integrateVol(ku, 0, j)
			}
			if math.Abs(val-want) > eps {
				t.Errorf("    FAIL u kernel node %v: got %v want %v", j, val, want)
			} else {
				t.Logf("         u kernel node %v: got %v", j, val)
			}
		}

		testu(0, test.WantU1)
		testu(1, test.WantU2)
		testu(2, test.WantU3)
		testu(3, test.WantU4)
	}
}
