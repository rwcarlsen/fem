package main

import (
	"fmt"
	"io"
	"log"

	"github.com/gonum/integrate/quad"
	"github.com/gonum/matrix/mat64"
)

func main() {
	xs := []float64{0, 1, 2, 3}

	//n := 100
	//elem := NewElement(xs)
	//elem.PrintShapeFuncs(os.Stdout, n)
	//elem.PrintFunc(os.Stdout, n)

	xs = []float64{0, 1, 2, 3, 4, 5, 6}
	mesh, err := NewMesh(xs, 3, Boundary{}, Boundary{})
	if err != nil {
		log.Fatal(err)
	}

	for i, elem := range mesh.Elems {
		fmt.Printf("elem %v\n", i)
		for _, n := range elem.Nodes {
			fmt.Printf("    node %p at x=%v\n", n, n.X())
		}
	}

	fmt.Println("NodeList:")
	for _, nds := range mesh.Nodes {
		for _, n := range nds {
			fmt.Printf("    node %p at x=%v\n", n, n.X())
		}
	}
}

type KernelParams struct {
	X     float64
	U     float64
	GradU float64
	W     float64
	GradW float64
}

type Kerneler interface {
	Kernel(p *KernelParams) float64
}

type NodeId int

type BoundaryType int

const (
	Essential BoundaryType = iota
	Dirichlet
)

type Boundary struct {
	Val  float64
	Type BoundaryType
}

type Mesh struct {
	Elems       []*Element
	Nodes       map[float64][]*Node
	Left, Right Boundary
}

// aka essential boundary condition matrix
func (m *Mesh) essentialBC() *mat64.Dense {
	panic("unimplemented")
}

func (m *Mesh) Interpolate(x float64) float64 {
	for _, e := range m.Elems {
		if e.Left() <= x && x <= e.Right() {
			return e.Interpolate(x)
		}
	}
	panic("cannot interpolate out of bounds on mesh")
}

// NewMesh creates a simply-connected mesh with nodes at the specified points and degree nodes per element.
func NewMesh(nodePos []float64, degree int, left, right Boundary) (*Mesh, error) {
	m := &Mesh{Nodes: map[float64][]*Node{}, Left: left, Right: right}
	if (len(nodePos)-1)%(degree-1) != 0 {
		return nil, fmt.Errorf("incompatible mesh degree (%v) and node count (%v)", degree, len(nodePos))
	}

	nElems := (len(nodePos) - 1) / (degree - 1)
	for i := 0; i < nElems; i++ {
		xs := make([]float64, degree)
		for j := 0; j < degree; j++ {
			xs[j] = nodePos[i*(degree-1)+j]
		}
		elem := NewElement(xs)
		m.Elems = append(m.Elems, elem)
		for _, n := range elem.Nodes {
			m.Nodes[n.X()] = append(m.Nodes[n.X()], n)
		}
	}
	return m, nil
}

func (m *Mesh) StiffnessMatrix(k Kerneler) (A, b mat64.Dense) {
	for _, e := range m.Elems {
		for _, n := range e.Nodes {
			fn := func(x float64) float64 {
				pars := &KernelParams{
					X:     x,
					U:     n.Sample(x),
					GradU: n.Deriv(x),
					W:     n.SampleWeight(x),
					GradW: n.DerivWeight(x),
				}
				return k.Kernel(pars)
			}
			k := quad.Fixed(fn, e.Left(), e.Right(), len(e.Nodes), quad.Legendre{}, 0)
			_ = k
		}
	}
	panic("unimplemented")
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
	// Index is the x-value where the node is equal to Val
	Index  int
	Xvals  []float64
	Val    float64
	Weight float64
}

func NewNode(xIndex int, xs []float64) *Node {
	return &Node{
		Index:  xIndex,
		Xvals:  append([]float64{}, xs...),
		Val:    1,
		Weight: 1,
	}
}

func (n *Node) X() float64 { return n.Xvals[n.Index] }

// Sample returns the value of the shape function at x.
func (n *Node) Sample(x float64) float64 {
	u := n.Val
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		u *= (x - x0) / (n.X() - x0)
	}
	return u
}

// SampleWeight returns the value of the weight function at x.
func (n *Node) SampleWeight(x float64) float64 { return n.Sample(x) / n.Val * n.Weight }

// Deriv returns the derivative of the shape function at x.
func (n *Node) Deriv(x float64) float64 {
	u := n.Val
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()-x0)*u + (x-x0)/(n.X()-x0)*dudx
		u *= (x - x0) / (n.X() - x0)
	}
	return dudx
}

// DerivWeight returns the derivative of the weight function at x.
func (n *Node) DerivWeight(x float64) float64 {
	u := n.Weight
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()-x0)*u + (x-x0)/(n.X()-x0)*dudx
		u *= (x - x0) / (n.X() - x0)
	}
	return dudx
}

// Element holds a collection of nodes comprising a finite element. The
// element is calibrated to a particular (approximate) solution and can be
// queried to provide said solution at various points within the element.
type Element struct {
	Nodes   []*Node
	NodePos []float64
}

func (e *Element) Left() float64  { return e.NodePos[0] }
func (e *Element) Right() float64 { return e.NodePos[len(e.NodePos)-1] }

func NewElement(xs []float64) *Element {
	e := &Element{NodePos: append([]float64{}, xs...)}
	for i := range xs {
		n := NewNode(i, xs)
		e.Nodes = append(e.Nodes, n)
	}
	return e
}

func (e *Element) Interpolate(x float64) float64 {
	if x < e.Left() || x > e.Right() {
		return 0
	}
	u := 0.0
	for _, n := range e.Nodes {
		u += n.Sample(x)
	}
	return u
}

func (e *Element) Deriv(x float64) float64 {
	if x < e.Left() || x > e.Right() {
		return 0
	}
	u := 0.0
	for _, n := range e.Nodes {
		u += n.Deriv(x)
	}
	return u
}

func (e *Element) PrintFunc(w io.Writer, nsamples int) {
	xrange := e.Right() - e.Left()
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := e.Left() + xrange*float64(i)/float64(nsamples)
		fmt.Fprintf(w, "%v\t%v\t%v\n", x, e.Interpolate(x), e.Deriv(x))
	}
}

// PrintShapeFuncs prints the shape functions and their derivatives in tab-separated form
// with nsamples evenly spaced over the element's domain (one sample per line) in the form:
//
//    [x]	[Node1-shape(x)]	[Node1-shapederiv(x)]	[Node2-shape(x)]	...
//
func (e *Element) PrintShapeFuncs(w io.Writer, nsamples int) {
	xrange := e.Right() - e.Left()
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := e.Left() + xrange*float64(i)/float64(nsamples)
		fmt.Fprintf(w, "%v", x)
		for _, n := range e.Nodes {
			if x < e.Left() || x > e.Right() {
				fmt.Fprintf(w, "\t0\t0")
			} else {
				fmt.Fprintf(w, "\t%v\t%v", n.Sample(x), n.Deriv(x))
			}
		}
		fmt.Fprintf(w, "\n")
	}
}
