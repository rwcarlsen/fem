package main

import "testing"

func TestCombinations(t *testing.T) {
	ndims := 2
	nsplits := 3
	combs := combinations(ndims, nsplits, nil)
	for i := range combs {
		t.Logf("%v", combs[i])
	}
}

func TestSplitBox(t *testing.T) {
	b := &box{Low: []float64{0, 0}, Up: []float64{1, 1}}
	nsplits := 3
	b.splitBox(nsplits)
	for _, b := range b.children {
		t.Errorf("low=%v, up=%v", b.Low, b.Up)
	}
}

func TestNewBox(t *testing.T) {
	mesh, _ := NewMeshSimple1D([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 2)
	b := newBox(mesh.Elems)
	for
}
