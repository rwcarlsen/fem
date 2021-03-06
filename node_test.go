package main

import (
	"fmt"
	"testing"
)

func TestLagrange2D_PrintVal(t *testing.T) {
	return // comment out to generate plottable data for Lagrange2D
	nodes := []LagrangeND{}
	for i := 0; i < 9; i++ {
		nodes = append(nodes, LagrangeND{Order: 2, Index: i})
	}

	nsamples := 100
	for i := 0; i < nsamples; i++ {
		x := -1 + 2*float64(i)/float64(nsamples-1)
		for j := 0; j < nsamples; j++ {
			y := -1 + 2*float64(j)/float64(nsamples-1)
			fmt.Printf("%v\t%v", x, y)
			for _, n := range nodes {
				val := n.Value([]float64{x, y}, -1)
				fmt.Printf("\t%v", val)
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

func TestLagrangeND_Value(t *testing.T) {
	tests := []struct {
		Order   int
		Index   int
		SampleX []float64
		Want    float64
	}{
		// 1D tests
		{Order: 1, Index: 0, SampleX: []float64{-1.}, Want: 1.0},
		{Order: 1, Index: 0, SampleX: []float64{0.0}, Want: 0.5},
		{Order: 1, Index: 0, SampleX: []float64{1.0}, Want: 0.0},
		{Order: 1, Index: 1, SampleX: []float64{-1.}, Want: 0.0},
		{Order: 1, Index: 1, SampleX: []float64{0.0}, Want: 0.5},
		{Order: 1, Index: 1, SampleX: []float64{1.0}, Want: 1.0},
		{Order: 2, Index: 0, SampleX: []float64{-1.}, Want: 1.0},
		{Order: 2, Index: 0, SampleX: []float64{0.0}, Want: 0.0},
		{Order: 2, Index: 0, SampleX: []float64{1.0}, Want: 0.0},
		{Order: 2, Index: 1, SampleX: []float64{-1.}, Want: 0.0},
		{Order: 2, Index: 1, SampleX: []float64{0.0}, Want: 1.0},
		{Order: 2, Index: 1, SampleX: []float64{1.0}, Want: 0.0},
		{Order: 2, Index: 2, SampleX: []float64{-1.}, Want: 0.0},
		{Order: 2, Index: 2, SampleX: []float64{0.0}, Want: 0.0},
		{Order: 2, Index: 2, SampleX: []float64{1.0}, Want: 1.0},

		// 2D tests
		{Order: 1, Index: 0, SampleX: []float64{-1, -1}, Want: 1},
		{Order: 1, Index: 0, SampleX: []float64{-1, 1}, Want: 0},
		{Order: 1, Index: 0, SampleX: []float64{1, -1}, Want: 0},
		{Order: 1, Index: 0, SampleX: []float64{1, 1}, Want: 0},
		{Order: 1, Index: 0, SampleX: []float64{0, 0}, Want: .25},
		{Order: 1, Index: 0, SampleX: []float64{-1, 0}, Want: .5},
		{Order: 1, Index: 0, SampleX: []float64{1, 0}, Want: 0},

		{Order: 1, Index: 1, SampleX: []float64{-1, -1}, Want: 0},
		{Order: 1, Index: 1, SampleX: []float64{-1, 1}, Want: 0},
		{Order: 1, Index: 1, SampleX: []float64{1, -1}, Want: 1},
		{Order: 1, Index: 1, SampleX: []float64{1, 1}, Want: 0},
		{Order: 1, Index: 1, SampleX: []float64{0, 0}, Want: .25},
		{Order: 1, Index: 1, SampleX: []float64{-1, 0}, Want: 0},
		{Order: 1, Index: 1, SampleX: []float64{1, 0}, Want: .5},

		{Order: 1, Index: 2, SampleX: []float64{-1, -1}, Want: 0},
		{Order: 1, Index: 2, SampleX: []float64{-1, 1}, Want: 1},
		{Order: 1, Index: 2, SampleX: []float64{1, -1}, Want: 0},
		{Order: 1, Index: 2, SampleX: []float64{1, 1}, Want: 0},
		{Order: 1, Index: 2, SampleX: []float64{0, 0}, Want: .25},
		{Order: 1, Index: 2, SampleX: []float64{-1, 0}, Want: .5},
		{Order: 1, Index: 2, SampleX: []float64{1, 0}, Want: 0},

		{Order: 1, Index: 3, SampleX: []float64{-1, -1}, Want: 0},
		{Order: 1, Index: 3, SampleX: []float64{-1, 1}, Want: 0},
		{Order: 1, Index: 3, SampleX: []float64{1, -1}, Want: 0},
		{Order: 1, Index: 3, SampleX: []float64{1, 1}, Want: 1},
		{Order: 1, Index: 3, SampleX: []float64{0, 0}, Want: .25},
		{Order: 1, Index: 3, SampleX: []float64{-1, 0}, Want: 0},
		{Order: 1, Index: 3, SampleX: []float64{1, 0}, Want: .5},
	}

	for i, test := range tests {
		n := LagrangeND{Order: test.Order, Index: test.Index}
		v := n.Value(test.SampleX, -1)
		if v != test.Want {
			t.Errorf("    FAIL case %2v (order %v, index %v): f(%2v) = %4v, want %2v", i+1, test.Order, test.Index, test.SampleX, v, test.Want)
		} else {
			t.Logf("         case %2v (order %v, index %v): f(%2v) = %4v", i+1, test.Order, test.Index, test.SampleX, v)
		}
	}
}

func TestLagrangeND_Deriv(t *testing.T) {
	tests := []struct {
		Order   int
		Index   int
		SampleX []float64
		Want    []float64
	}{
		{Order: 1, Index: 0, SampleX: []float64{-1, -1}, Want: []float64{-.5, -.5}},
		{Order: 1, Index: 0, SampleX: []float64{-1, 1}, Want: []float64{0, -.5}},
		{Order: 1, Index: 0, SampleX: []float64{1, -1}, Want: []float64{-.5, 0}},
		{Order: 1, Index: 0, SampleX: []float64{1, 1}, Want: []float64{0, 0}},
		{Order: 1, Index: 0, SampleX: []float64{0, 0}, Want: []float64{-.25, -.25}},
		{Order: 1, Index: 0, SampleX: []float64{-1, 0}, Want: []float64{-.25, -.5}},
		{Order: 1, Index: 0, SampleX: []float64{1, 0}, Want: []float64{-.25, 0}},

		{Order: 1, Index: 1, SampleX: []float64{-1, -1}, Want: []float64{.5, 0}},
		{Order: 1, Index: 1, SampleX: []float64{-1, 1}, Want: []float64{0, 0}},
		{Order: 1, Index: 1, SampleX: []float64{1, -1}, Want: []float64{.5, -.5}},
		{Order: 1, Index: 1, SampleX: []float64{1, 1}, Want: []float64{0, -.5}},
		{Order: 1, Index: 1, SampleX: []float64{0, 0}, Want: []float64{.25, -.25}},
		{Order: 1, Index: 1, SampleX: []float64{-1, 0}, Want: []float64{.25, 0}},
		{Order: 1, Index: 1, SampleX: []float64{1, 0}, Want: []float64{.25, -.5}},

		{Order: 1, Index: 2, SampleX: []float64{-1, -1}, Want: []float64{0, .5}},
		{Order: 1, Index: 2, SampleX: []float64{-1, 1}, Want: []float64{-.5, .5}},
		{Order: 1, Index: 2, SampleX: []float64{1, -1}, Want: []float64{0, 0}},
		{Order: 1, Index: 2, SampleX: []float64{1, 1}, Want: []float64{-.5, 0}},
		{Order: 1, Index: 2, SampleX: []float64{0, 0}, Want: []float64{-.25, .25}},
		{Order: 1, Index: 2, SampleX: []float64{-1, 0}, Want: []float64{-.25, .5}},
		{Order: 1, Index: 2, SampleX: []float64{1, 0}, Want: []float64{-.25, 0}},

		{Order: 1, Index: 3, SampleX: []float64{-1, -1}, Want: []float64{0, 0}},
		{Order: 1, Index: 3, SampleX: []float64{-1, 1}, Want: []float64{.5, 0}},
		{Order: 1, Index: 3, SampleX: []float64{1, -1}, Want: []float64{0, .5}},
		{Order: 1, Index: 3, SampleX: []float64{1, 1}, Want: []float64{.5, .5}},
		{Order: 1, Index: 3, SampleX: []float64{0, 0}, Want: []float64{.25, .25}},
		{Order: 1, Index: 3, SampleX: []float64{-1, 0}, Want: []float64{.25, 0}},
		{Order: 1, Index: 3, SampleX: []float64{1, 0}, Want: []float64{.25, .5}},

		{Order: 2, Index: 0, SampleX: []float64{-1, -1}, Want: []float64{-1.5, -1.5}},
		{Order: 2, Index: 0, SampleX: []float64{-1, 1}, Want: []float64{0, .5}},
		{Order: 2, Index: 0, SampleX: []float64{1, -1}, Want: []float64{.5, 0}},
		{Order: 2, Index: 0, SampleX: []float64{1, 1}, Want: []float64{0, 0}},
		{Order: 2, Index: 0, SampleX: []float64{0, 0}, Want: []float64{0, 0}},
		{Order: 2, Index: 0, SampleX: []float64{-1, 0}, Want: []float64{0, -.5}},
		{Order: 2, Index: 0, SampleX: []float64{1, 0}, Want: []float64{0, 0}},

		{Order: 2, Index: 1, SampleX: []float64{-1, -1}, Want: []float64{2, 0}},
		{Order: 2, Index: 1, SampleX: []float64{-1, 1}, Want: []float64{0, 0}},
		{Order: 2, Index: 1, SampleX: []float64{1, -1}, Want: []float64{-2, 0}},
		{Order: 2, Index: 1, SampleX: []float64{1, 1}, Want: []float64{0, 0}},
		{Order: 2, Index: 1, SampleX: []float64{0, 0}, Want: []float64{0, -.5}},
		{Order: 2, Index: 1, SampleX: []float64{-1, 0}, Want: []float64{0, 0}},
		{Order: 2, Index: 1, SampleX: []float64{1, 0}, Want: []float64{0, 0}},

		{Order: 2, Index: 2, SampleX: []float64{-1, -1}, Want: []float64{-.5, 0}},
		{Order: 2, Index: 2, SampleX: []float64{-1, 1}, Want: []float64{0, 0}},
		{Order: 2, Index: 2, SampleX: []float64{1, -1}, Want: []float64{1.5, -1.5}},
		{Order: 2, Index: 2, SampleX: []float64{1, 1}, Want: []float64{0, .5}},
		{Order: 2, Index: 2, SampleX: []float64{0, 0}, Want: []float64{0, 0}},
		{Order: 2, Index: 2, SampleX: []float64{-1, 0}, Want: []float64{0, 0}},
		{Order: 2, Index: 2, SampleX: []float64{1, 0}, Want: []float64{0, -.5}},

		{Order: 2, Index: 3, SampleX: []float64{-1, -1}, Want: []float64{0, 2}},
		{Order: 2, Index: 3, SampleX: []float64{-1, 1}, Want: []float64{0, -2}},
		{Order: 2, Index: 3, SampleX: []float64{1, -1}, Want: []float64{0, 0}},
		{Order: 2, Index: 3, SampleX: []float64{1, 1}, Want: []float64{0, 0}},
		{Order: 2, Index: 3, SampleX: []float64{0, 0}, Want: []float64{-.5, 0}},
		{Order: 2, Index: 3, SampleX: []float64{-1, 0}, Want: []float64{-1.5, 0}},
		{Order: 2, Index: 3, SampleX: []float64{1, 0}, Want: []float64{.5, 0}},
	}

	for i, test := range tests {
		n := LagrangeND{Order: test.Order, Index: test.Index}
		d := n.Deriv(test.SampleX, nil, -1)
		dx, dy := d[0], d[1]
		if dx != test.Want[0] || dy != test.Want[1] {
			t.Errorf("    FAIL case %2v (order %v, index %v): f(%2v) = %5v, want %5v", i+1, test.Order, test.Index, test.SampleX, d, test.Want)
		} else {
			t.Logf("         case %2v (order %v, index %v): f(%2v) = %5v", i+1, test.Order, test.Index, test.SampleX, d)
		}
	}
}
