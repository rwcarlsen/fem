package main

import (
	"fmt"
	"io"
	"strings"
)

type box struct {
	Low      []float64
	Up       []float64
	Elems    []Element
	children []*box
}

func newBox(elems []Element, elemTarget, nsplit int) *box {
	lowest, upest := elems[0].Bounds()
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

	b := &box{Low: lowest, Up: upest, Elems: elems}
	b.split(elemTarget, nsplit)
	return b
}

func (b *box) printTree(w io.Writer, level int) {
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

// contains returns true if there is a non-null intersection between this box
// and e's bounding box.
func (b *box) contains(e Element) bool {
	low, up := e.Bounds()
	for i := range low {
		if up[i] < b.Low[i] || b.Up[i] < low[i] {
			return false
		}
	}
	return true
}

func (b *box) Find(x []float64) (Element, error) {
	for _, child := range b.children {
		for i := range x {
			if child.Low[i] <= x[i] && x[i] <= child.Up[i] {
				return child.Find(x)
			}
		}
	}
	for _, e := range b.Elems {
		if e.Contains(x) {
			return e, nil
		}
	}
	return nil, fmt.Errorf("element not found in bounding box tree for x=%v")
}

func (b *box) split(elemTarget, nsplit int) {
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

func (b *box) splitBox(n int) {
	ndim := len(b.Low)
	combs := combinations(ndim, n, nil)
	b.children = make([]*box, len(combs))
	for i, comb := range combs {
		b.children[i] = &box{}
		sub := b.children[i]
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
