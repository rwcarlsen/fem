package main

import (
	"fmt"
	"io"
	"strings"
)

// Box represents a bounding-box tree useful for quickly determining which
// element contains a particular point (in ~log([num_elems]) time).
type Box struct {
	Low      []float64
	Up       []float64
	Elems    []Element
	children []*Box
}

// NewBox builds an initialized bounding box tree that includes all given
// elements.  It splits a box into "nsplit^ndim" (ndim is the number of
// dimensions in the problem (i.e. len(node.X())) child bounding boxes if the
// number of elements in the box are greater than elemTarget.
func NewBox(elems []Element, elemTarget, nsplit int) *Box {
	l, u := elems[0].Bounds()
	lowest := make([]float64, len(l))
	upest := make([]float64, len(u))
	copy(lowest, l)
	copy(upest, u)

	for _, elem := range elems[1:] {
		low, up := elem.Bounds()
		for i := range low {
			if low[i] < lowest[i] {
				lowest[i] = low[i]
			}
			if up[i] > upest[i] {
				upest[i] = up[i]
			}
		}
	}

	b := &Box{Low: lowest, Up: upest, Elems: elems}
	b.split(elemTarget, nsplit)
	return b
}

func (b *Box) Find(x []float64) (Element, error) {
	for _, child := range b.children {
		for i := range x {
			if child.Low[i] <= x[i] && x[i] <= child.Up[i] {
				elem, err := child.Find(x)
				if err == nil {
					return elem, nil
				}
			}
		}
	}
	if len(b.children) == 0 {
		for _, e := range b.Elems {
			if e.Contains(x) {
				return e, nil
			}
		}
	}
	return nil, fmt.Errorf("element not found in bounding Box tree for x=%v", x)
}

func (b *Box) printTree(w io.Writer, level int) {
	prefix := strings.Repeat(" ", level*4)
	fmt.Fprintf(w, prefix+"Box (low=%v, up=%v)\n", b.Low, b.Up)
	for _, e := range b.Elems {
		low, up := e.Bounds()
		fmt.Fprintf(w, prefix+"    Elem (low=%v, up=%v)\n", low, up)
	}
	for _, child := range b.children {
		child.printTree(w, level+1)
	}
}

// contains returns true if there is a non-null intersection between this Box
// and e's bounding Box.
func (b *Box) contains(e Element) bool {
	low, up := e.Bounds()
	for i := range low {
		if up[i] < b.Low[i] || b.Up[i] < low[i] {
			return false
		}
	}
	return true
}

func (b *Box) split(elemTarget, nsplit int) {
	if len(b.Elems) <= elemTarget {
		return
	}

	b.splitBox(nsplit)
	for _, elem := range b.Elems {
		for _, child := range b.children {
			if child.contains(elem) {
				child.Elems = append(child.Elems, elem)
			}
		}
	}

	for _, child := range b.children {
		child.split(elemTarget, nsplit)
	}
}

func (b *Box) splitBox(n int) {
	ndim := len(b.Low)
	tot := 1
	for i := 0; i < n; i++ {
		tot *= ndim
	}
	if tot > len(b.Elems) {
		n = 2
	}
	combs := combinations(ndim, n, nil)

	b.children = make([]*Box, len(combs))
	for i, comb := range combs {
		b.children[i] = &Box{}
		sub := b.children[i]
		sub.Elems = make([]Element, 0, len(b.Elems))
		sub.Low = make([]float64, ndim)
		sub.Up = make([]float64, ndim)
		for dim, section := range comb {
			dx := b.Up[dim] - b.Low[dim]
			sub.Low[dim] = b.Low[dim] + dx/float64(n)*float64(section)
			sub.Up[dim] = sub.Low[dim] + dx/float64(n)
		}
	}
}

func combinations(ndims, nsplits int, prefix []int) [][]int {
	if len(prefix) == ndims {
		return [][]int{prefix}
	}

	combs := [][]int{}
	for i := 0; i < nsplits; i++ {
		combs = append(combs, combinations(ndims, nsplits, append(prefix, i))...)
	}
	return combs
}
