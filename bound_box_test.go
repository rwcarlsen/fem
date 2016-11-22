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
	boxes := splitBox(b, nsplits)
	for _, b := range boxes {
		t.Errorf("low=%v, up=%v", b.Low, b.Up)
	}
}
