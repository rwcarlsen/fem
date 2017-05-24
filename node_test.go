package main

import "testing"

func TestLagrange1D_Value(t *testing.T) {
	tests := []struct {
		Order   int
		Index   int
		SampleX float64
		Want    float64
	}{
		{Order: 1, Index: 0, SampleX: -1., Want: 1.0},
		{Order: 1, Index: 0, SampleX: 0.0, Want: 0.5},
		{Order: 1, Index: 0, SampleX: 1.0, Want: 0.0},
		{Order: 1, Index: 1, SampleX: -1., Want: 0.0},
		{Order: 1, Index: 1, SampleX: 0.0, Want: 0.5},
		{Order: 1, Index: 1, SampleX: 1.0, Want: 1.0},
		{Order: 2, Index: 0, SampleX: -1., Want: 1.0},
		{Order: 2, Index: 0, SampleX: 0.0, Want: 0.0},
		{Order: 2, Index: 0, SampleX: 1.0, Want: 0.0},
		{Order: 2, Index: 1, SampleX: -1., Want: 0.0},
		{Order: 2, Index: 1, SampleX: 0.0, Want: 1.0},
		{Order: 2, Index: 1, SampleX: 1.0, Want: 0.0},
		{Order: 2, Index: 2, SampleX: -1., Want: 0.0},
		{Order: 2, Index: 2, SampleX: 0.0, Want: 0.0},
		{Order: 2, Index: 2, SampleX: 1.0, Want: 1.0},
	}

	for i, test := range tests {
		fn := Lagrange1D{test.Index, test.Order}
		y := fn.Value([]float64{test.SampleX})
		if y != test.Want {
			t.Errorf("FAIL case %v (order=%v, index=%v): f(%v)=%v, want %v", i+1, test.Order, test.Index, test.SampleX, y, test.Want)
		} else {
			t.Logf("     case %v (order=%v, index=%v): f(%v)=%v", i+1, test.Order, test.Index, test.SampleX, y)
		}
	}
}

func TestLagrange1D_Deriv(t *testing.T) {
	tests := []struct {
		Order   int
		Index   int
		SampleX float64
		Want    float64
	}{
		////////// x1=-1, x2=1 ///////////
		//            1
		// dN/dx =  -----
		//          x1-x2
		{Order: 1, Index: 0, SampleX: -1., Want: -0.5},
		{Order: 1, Index: 0, SampleX: 0.0, Want: -0.5},
		{Order: 1, Index: 0, SampleX: 1.0, Want: -0.5},
		//            1
		// dN/dx =  -----
		//          x2-x1
		{Order: 1, Index: 1, SampleX: -1., Want: 0.5},
		{Order: 1, Index: 1, SampleX: 0.0, Want: 0.5},
		{Order: 1, Index: 1, SampleX: 1.0, Want: 0.5},

		////////// x1=-1, x2=0, x3=1 ///////////
		//          x-x2      1     x-x3      1
		// dN/dx =  ----- * ----- + ----- * -----
		//          x1-x2   x1-x3   x1-x3   x1-x2
		{Order: 2, Index: 0, SampleX: -1., Want: -1.5},
		{Order: 2, Index: 0, SampleX: 0.0, Want: -.5},
		{Order: 2, Index: 0, SampleX: 1.0, Want: 0.5},
		//          x-x1      1     x-x3      1
		// dN/dx =  ----- * ----- + ----- * -----
		//          x2-x1   x2-x3   x2-x3   x2-x1
		{Order: 2, Index: 1, SampleX: -1., Want: 2.0},
		{Order: 2, Index: 1, SampleX: 0.0, Want: 0.0},
		{Order: 2, Index: 1, SampleX: 1.0, Want: -2.},
		//          x-x1      1     x-x2      1
		// dN/dx =  ----- * ----- + ----- * -----
		//          x3-x1   x3-x2   x3-x2   x3-x1
		{Order: 2, Index: 2, SampleX: -1., Want: -.5},
		{Order: 2, Index: 2, SampleX: 0.0, Want: 0.5},
		{Order: 2, Index: 2, SampleX: 1.0, Want: 1.5},
	}

	for i, test := range tests {
		fn := Lagrange1D{test.Index, test.Order}
		y := fn.Deriv([]float64{test.SampleX})[0]
		if y != test.Want {
			t.Errorf("FAIL case %v (order=%v, index=%v): df/dx(%v)=%v, want %v", i+1, test.Order, test.Index, test.SampleX, y, test.Want)
		} else {
			t.Logf("     case %v (order=%v, index=%v): df/dx(%v)=%v", i+1, test.Order, test.Index, test.SampleX, y)
		}
	}
}

func TestBilinear_Value(t *testing.T) {
	tests := []struct {
		Index   int
		SampleX []float64
		Want    float64
	}{
		{Index: 0, SampleX: []float64{-1, -1}, Want: 1},
		{Index: 0, SampleX: []float64{-1, 1}, Want: 0},
		{Index: 0, SampleX: []float64{1, -1}, Want: 0},
		{Index: 0, SampleX: []float64{1, 1}, Want: 0},
		{Index: 0, SampleX: []float64{0, 0}, Want: .25},
		{Index: 0, SampleX: []float64{-1, 0}, Want: .5},
		{Index: 0, SampleX: []float64{1, 0}, Want: 0},

		{Index: 1, SampleX: []float64{-1, -1}, Want: 0},
		{Index: 1, SampleX: []float64{-1, 1}, Want: 0},
		{Index: 1, SampleX: []float64{1, -1}, Want: 1},
		{Index: 1, SampleX: []float64{1, 1}, Want: 0},
		{Index: 1, SampleX: []float64{0, 0}, Want: .25},
		{Index: 1, SampleX: []float64{-1, 0}, Want: 0},
		{Index: 1, SampleX: []float64{1, 0}, Want: .5},

		{Index: 2, SampleX: []float64{-1, -1}, Want: 0},
		{Index: 2, SampleX: []float64{-1, 1}, Want: 0},
		{Index: 2, SampleX: []float64{1, -1}, Want: 0},
		{Index: 2, SampleX: []float64{1, 1}, Want: 1},
		{Index: 2, SampleX: []float64{0, 0}, Want: .25},
		{Index: 2, SampleX: []float64{-1, 0}, Want: 0},
		{Index: 2, SampleX: []float64{1, 0}, Want: .5},

		{Index: 3, SampleX: []float64{-1, -1}, Want: 0},
		{Index: 3, SampleX: []float64{-1, 1}, Want: 1},
		{Index: 3, SampleX: []float64{1, -1}, Want: 0},
		{Index: 3, SampleX: []float64{1, 1}, Want: 0},
		{Index: 3, SampleX: []float64{0, 0}, Want: .25},
		{Index: 3, SampleX: []float64{-1, 0}, Want: .5},
		{Index: 3, SampleX: []float64{1, 0}, Want: 0},
	}

	for i, test := range tests {
		n := Bilinear{Index: test.Index}
		v := n.Value(test.SampleX)
		if v != test.Want {
			t.Errorf("    FAIL case %2v (index %v): f(%2v) = %4v, want %2v", i+1, test.Index, test.SampleX, v, test.Want)
		} else {
			t.Logf("         case %2v (index %v): f(%2v) = %4v", i+1, test.Index, test.SampleX, v)
		}
	}
}

func TestBilinear_Deriv(t *testing.T) {
	tests := []struct {
		Index   int
		SampleX []float64
		Want    []float64
	}{
		{Index: 0, SampleX: []float64{-1, -1}, Want: []float64{-.5, -.5}},
		{Index: 0, SampleX: []float64{-1, 1}, Want: []float64{0, -.5}},
		{Index: 0, SampleX: []float64{1, -1}, Want: []float64{-.5, 0}},
		{Index: 0, SampleX: []float64{1, 1}, Want: []float64{0, 0}},
		{Index: 0, SampleX: []float64{0, 0}, Want: []float64{-.25, -.25}},
		{Index: 0, SampleX: []float64{-1, 0}, Want: []float64{-.25, -.5}},
		{Index: 0, SampleX: []float64{1, 0}, Want: []float64{-.25, 0}},

		{Index: 1, SampleX: []float64{-1, -1}, Want: []float64{.5, 0}},
		{Index: 1, SampleX: []float64{-1, 1}, Want: []float64{0, 0}},
		{Index: 1, SampleX: []float64{1, -1}, Want: []float64{.5, -.5}},
		{Index: 1, SampleX: []float64{1, 1}, Want: []float64{0, -.5}},
		{Index: 1, SampleX: []float64{0, 0}, Want: []float64{.25, -.25}},
		{Index: 1, SampleX: []float64{-1, 0}, Want: []float64{.25, 0}},
		{Index: 1, SampleX: []float64{1, 0}, Want: []float64{.25, -.5}},

		{Index: 2, SampleX: []float64{-1, -1}, Want: []float64{0, 0}},
		{Index: 2, SampleX: []float64{-1, 1}, Want: []float64{.5, 0}},
		{Index: 2, SampleX: []float64{1, -1}, Want: []float64{0, .5}},
		{Index: 2, SampleX: []float64{1, 1}, Want: []float64{.5, .5}},
		{Index: 2, SampleX: []float64{0, 0}, Want: []float64{.25, .25}},
		{Index: 2, SampleX: []float64{-1, 0}, Want: []float64{.25, 0}},
		{Index: 2, SampleX: []float64{1, 0}, Want: []float64{.25, .5}},

		{Index: 3, SampleX: []float64{-1, -1}, Want: []float64{0, .5}},
		{Index: 3, SampleX: []float64{-1, 1}, Want: []float64{-.5, .5}},
		{Index: 3, SampleX: []float64{1, -1}, Want: []float64{0, 0}},
		{Index: 3, SampleX: []float64{1, 1}, Want: []float64{-.5, 0}},
		{Index: 3, SampleX: []float64{0, 0}, Want: []float64{-.25, .25}},
		{Index: 3, SampleX: []float64{-1, 0}, Want: []float64{-.25, .5}},
		{Index: 3, SampleX: []float64{1, 0}, Want: []float64{-.25, 0}},
	}

	for i, test := range tests {
		n := Bilinear{Index: test.Index}
		d := n.Deriv(test.SampleX)
		dx, dy := d[0], d[1]
		if dx != test.Want[0] || dy != test.Want[1] {
			t.Errorf("    FAIL case %2v (index %v): f(%2v) = %5v, want %5v", i+1, test.Index, test.SampleX, d, test.Want)
		} else {
			t.Logf("         case %2v (index %v): f(%2v) = %5v", i+1, test.Index, test.SampleX, d)
		}
	}
}
