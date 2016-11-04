package main

import (
	"fmt"
	"io"
	"os"
)

func main() {
	pts = []Point{
		{0, 2},
		{1, 3},
		{2, 9},
		{3, 1},
	}
	n := 100

	elem := NewElement(pts)
	elem.PrintShapeFuncs(os.Stdout, n)
	//elem.PrintFunc(os.Stdout, n)
}

type Point struct {
	X, Y float64
}

// Node represents a finite element node.  It holds a polynomial shape
// function that can be sampled.  It represents a shape function of the
// following form:
//
//    (x - x2)    (x - x3)    (x - x4)
//    --------- * --------- * ---------  * ...
//    (x1 - x2)   (x1 - x3)   (x1 - x4)
//
type Node struct {
	Xmain float64
	Xzero []float64
	Val   float64
}

func NewNode(p Point, xZeros []float64) Node {
	return Node{
		Xmain: p.X,
		Val:   p.Y,
		Xzero: xZeros,
	}
}

// Sample returns the value of the shape function at x.
func (n Node) Sample(x float64) float64 {
	u := n.Val
	for _, x0 := range n.Xzero {
		u *= (x - x0) / (n.Xmain - x0)
	}
	return u
}

// Element holds a collection of nodes comprising a finite element. The
// element is calibrated to a particular (approximate) solution and can be
// queried to provide said solution at various points within the element.
type Element struct {
	Nodes       []Node
	Left, Right float64
}

func NewElement(pts []Point) *Element {
	e := &Element{Left: pts[0].X, Right: pts[len(pts)-1].X}
	xs := []float64{}
	for _, p := range pts {
		xs = append(xs, p.X)
	}

	for i, p := range pts {
		xZeros := append([]float64{}, xs[:i]...)
		xZeros = append(xZeros, xs[i+1:]...)
		e.Nodes = append(e.Nodes, NewNode(p, xZeros))
	}
	return e
}

func (e *Element) Interpolate(x float64) float64 {
	if x < e.Left || x > e.Right {
		return 0
	}
	u := 0.0
	for _, n := range e.Nodes {
		u += n.Sample(x)
	}
	return u
}

func (e *Element) PrintFunc(w io.Writer, nsamples int) {
	xrange := e.Right - e.Left
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := e.Left + xrange*float64(i)/float64(nsamples)
		fmt.Fprintf(w, "%v\t%v\n", x, e.Interpolate(x))
	}
}

func (e *Element) PrintShapeFuncs(w io.Writer, nsamples int) {
	xrange := e.Right - e.Left
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := e.Left + xrange*float64(i)/float64(nsamples)
		fmt.Fprintf(w, "%v", x)
		for _, n := range e.Nodes {
			if x < e.Left || x > e.Right {
				fmt.Fprintf(w, "\t0")
			} else {
				fmt.Fprintf(w, "\t%v", n.Sample(x))
			}
		}
		fmt.Fprintf(w, "\n")
	}
}
