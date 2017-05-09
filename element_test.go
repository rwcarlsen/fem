package main

import "testing"

func TestElement1D(t *testing.T) {
	tests := []struct {
		Xs      []float64
		Ys      []float64
		SampleX float64
		Want    float64
	}{
		{Xs: []float64{0, 1}, Ys: []float64{1, 2}, SampleX: -1, Want: 1.0},
		{Xs: []float64{0, 1}, Ys: []float64{1, 2}, SampleX: 0, Want: 1.5},
		{Xs: []float64{0, 1}, Ys: []float64{1, 2}, SampleX: 1.0, Want: 2.0},
		{Xs: []float64{0, 1, 2}, Ys: []float64{1, 2, 9}, SampleX: -1, Want: 1},
		{Xs: []float64{0, 1, 2}, Ys: []float64{1, 2, 9}, SampleX: 0, Want: 2},
		{Xs: []float64{0, 1, 2}, Ys: []float64{1, 2, 9}, SampleX: 1, Want: 9},
	}

	for i, test := range tests {
		elem := NewElementSimple1D(test.Xs)
		for i, n := range elem.Nodes() {
			n.U = test.Ys[i]
			n.W = 1
		}
		y, err := Interpolate(elem, []float64{test.SampleX})
		if err != nil {
			t.Errorf("FAIL case %v (xs=%v, ys=%v): %v", i+1, test.Xs, test.Ys, err)
		} else if y != test.Want {
			t.Errorf("FAIL case %v (xs=%v, ys=%v): f(%v)=%v, want %v", i+1, test.Xs, test.Ys, test.SampleX, y, test.Want)
		} else {
			t.Logf("     case %v (xs=%v, ys=%v): f(%v)=%v", i+1, test.Xs, test.Ys, test.SampleX, y)
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

func TestElement2D_IntegrateBoundary(t *testing.T) {
	return // TODO: implement

	const volume = 1
	const boundary = 2
	tests := []struct {
		X1, Y1, V1 float64
		X2, Y2, V2 float64
		X3, Y3, V3 float64
		X4, Y4, V4 float64

		Integral  int
		WantConst float64
		WantU1    float64
		WantU2    float64
		WantU3    float64
		WantU4    float64
	}{
		{ // scale+translate
			Integral: boundary,
			X1:       0, Y1: 0, X2: 1, Y2: 0, X3: 1, Y3: 1, X4: 0, Y4: 1,
			V1: 1, V2: 1, V3: 1, V4: 1,
			WantConst: 4,
			WantU1:    1,
			WantU2:    1,
			WantU3:    1,
			WantU4:    1,
		}, { // translate
			Integral: boundary,
			X1:       0, Y1: 0, X2: 2, Y2: 0, X3: 2, Y3: 2, X4: 0, Y4: 2,
			V1: 1, V2: 1, V3: 1, V4: 1,
			WantConst: 8,
			WantU1:    2,
			WantU2:    2,
			WantU3:    2,
			WantU4:    2,
		}, { // translate, scale, non-uniform node U values
			Integral: boundary,
			X1:       0, Y1: 0, X2: 2, Y2: 0, X3: 2, Y3: 2, X4: 0, Y4: 2,
			V1: 1, V2: 2, V3: 2, V4: 1,
			WantConst: 8,
			WantU1:    2,
			WantU2:    4,
			WantU3:    4,
			WantU4:    2,
		}, { // translate, scale, distort
			Integral: boundary,
			X1:       0, Y1: 0, X2: 2, Y2: 0, X3: 1, Y3: 3, X4: 0, Y4: 2,
			V1: 1, V2: 1, V3: 1, V4: 1,
			WantConst: 8,
			WantU1:    2,
			WantU2:    4,
			WantU3:    4,
			WantU4:    2,
		}, { // identity mapping
			Integral: volume,
			X1:       0, Y1: 0, X2: 2, Y2: 0, X3: 2, Y3: 2, X4: 0, Y4: 2,
			V1: 1, V2: 2, V3: 2, V4: 1,
			WantConst: 8,
			WantU1:    2,
			WantU2:    4,
			WantU3:    4,
			WantU4:    2,
		},
	}

	for i, test := range tests {
		ts := test
		elem, err := NewElementSimple2D(ts.X1, ts.Y1, ts.X2, ts.Y2, ts.X3, ts.Y3, ts.X4, ts.Y4)
		if err != nil {
			t.Error(err)
			continue
		}

		nds := elem.Nodes()
		nds[0].Set(test.V1, 1)
		nds[1].Set(test.V2, 1)
		nds[2].Set(test.V3, 1)
		nds[3].Set(test.V4, 1)

		var kconst constKernel = 1
		val := elem.integrateBoundary(kconst, 0, 0)
		if test.Integral == volume {
			val = elem.integrateVol(kconst, 0, 0)
		}
		t.Logf("case %v", i+1)
		if val != test.WantConst {
			t.Errorf("    FAIL const kernel: got %v want %v", val, test.WantConst)
		} else {
			t.Logf("         const kernel: got %v", val)
		}

		var ku testKernel = 1
		testu := func(j int, want float64) {
			val := elem.integrateBoundary(ku, 0, j)
			if test.Integral == volume {
				val = elem.integrateVol(ku, 0, j)
			}
			if val != want {
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
