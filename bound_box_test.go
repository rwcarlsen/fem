package main

import (
	"io/ioutil"
	"testing"
)

func TestCombinations(t *testing.T) {
	ndims := 2
	nsplits := 3
	combs := combinations(ndims, nsplits, nil)
	for i := range combs {
		t.Logf("%v", combs[i])
	}
}

func TestSplitBox(t *testing.T) {
	b := &Box{Low: []float64{0, 0}, Up: []float64{1, 1}}
	nsplits := 3
	b.splitBox(nsplits)
	for _, b := range b.children {
		t.Logf("low=%v, up=%v", b.Low, b.Up)
	}
}

func TestNewBox(t *testing.T) {
	mesh, _ := NewMeshStructured(2, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	elemTarget := 3
	nsplit := 2
	b := NewBox(mesh.Elems, elemTarget, nsplit)
	b.printTree(ioutil.Discard, 0)
	//b.printTree(os.Stdout, 0)
}
