package main

import (
	"fmt"
	"io"
	"os"
	"log"
	
	"github.com/gonum/mat/mat64"
)

func main() {
	pts := []Point{
		{0, 2},
		{1, 3},
		{2, 9},
		{3, 1},
	}
	weights := []float64{
		1,
		1,
		1,
		1,
	}

	n := 100

	elem := NewElement(pts, weights)
	elem.PrintShapeFuncs(os.Stdout, n)
	//elem.PrintFunc(os.Stdout, n)
	
	xs := []float64{0, 1, 2, 3, 4, 5, 6}
	mesh, err := NewMesh(xs, 3)
	if err != nil {
		log.Fatal(err)
	}
	
	for i, elem := range mesh.Elems {
		fmt.Printf("elem %v\n", i)
		for _, n := range elem.Nodes {
			fmt.Printf("    node %p at x=%v\n", n, n.Xmain)
		}
	}
	
	fmt.Println("NodeList:")
	for _, n := range mesh.Nodes {
		fmt.Printf("    node %p at x=%v\n", n, n.Xmain)
	}
}

type KernelParams struct {
	X float64
	U float64
	DU float64
	W float64
	DW float64
}

type Kernel Interface {
	Residual(p *KernelParams) float64
}

type NodeId int

type Mesh struct {
	Elems []*Element
	Nodes []*Node
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
func NewMesh(nodePos []float64, degree int) (*Mesh, error) {
	m := &Mesh{}
	if (len(nodePos)-1) % (degree-1) != 0 {
		return nil, fmt.Errorf("incompatible mesh degree (%v) and node count (%v)", degree, len(nodePos))
	}
	nElems := (len(nodePos)-1) / (degree-1)
	for i := 0; i < nElems; i++ {
		pts := make([]Point, degree)
		weights := make([]float64, degree)
		for j := 0; j < degree; j++ {
			pts[j] = Point{nodePos[i*(degree-1)+j], 1}
			weights[j] = 1
		}
		elem := NewElement(pts, weights)
		m.Elems = append(m.Elems, elem)
		
		if i == 0 {
			m.Nodes = append(m.Nodes, elem.Nodes[0])
		}
		m.Nodes = append(m.Nodes, elem.Nodes[1:]...)
		
		// have adjacent elements share edge node
		if i > 0 {
			prevNodes := m.Elems[i-1].Nodes
			elem.Nodes[0] = prevNodes[len(prevNodes)-1]
		}
	}
	return m, nil
}

func (m *Mesh) BuildSystem(k Kernel) (A, b mat64.Dense) {
	for i, e := range m.Elems {
		fn := func(x float64) float64 {
			pars := &KernelParams{
				X: x,
				U: e.Interpolate(x),
				DU: e.Deriv(x),
				W: e.InterpolateWeight(x),
				DW: e.DerivWeight(x),
			}
			return k.Residual(pars)
		}
		residual := quad.Fixed(fn, e.Left(), e.Right(), len(e.Nodes), quad.Legendre{}, 0)
	}
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
	Weight float64
}

func NewNode(p Point, xZeros []float64, weight float64) *Node {
	return &Node{
		Xmain: p.X,
		Xzero: xZeros,
		Val:   p.Y,
		Weight: weight,
	}
}

func (n *Node) Update(val, weight float64) {
	n.Val = val
	n.Weight = weight
}

// Deriv returns the derivative of the shape function at x.
func (n *Node) Deriv(x float64) float64 {
	u := n.Val
	dudx := 0.0
	for _, x0 := range n.Xzero {
		dudx = 1 / (n.Xmain - x0) * u + (x - x0) / (n.Xmain - x0) * dudx
		u *= (x - x0) / (n.Xmain - x0)
	}
	return dudx
}

// Sample returns the value of the shape function at x.
func (n *Node) Sample(x float64) float64 {
	u := n.Val
	for _, x0 := range n.Xzero {
		u *= (x - x0) / (n.Xmain - x0)
	}
	return u
}

// DerivWeight returns the derivative of the weight function at x.
func (n *Node) DerivWeight(x float64) float64 {
	u := n.Weight
	dudx := 0.0
	for _, x0 := range n.Xzero {
		dudx = 1 / (n.Xmain - x0) * u + (x - x0) / (n.Xmain - x0) * dudx
		u *= (x - x0) / (n.Xmain - x0)
	}
	return dudx
}

// SampleWeight returns the value of the weight function at x.
func (n *Node) SampleWeight(x float64) float64 {return n.Sample(x) / n.Val * n.Weight}

// Element holds a collection of nodes comprising a finite element. The
// element is calibrated to a particular (approximate) solution and can be
// queried to provide said solution at various points within the element.
type Element struct {
	Nodes       []*Node
	NodePos []float64
}

func (e *Element) Left() float64 {return e.NodePos[0]}
func (e *Element) Right() float64 {return e.NodePos[len(e.NodePos)-1]}

func NewElementNodes(nodes []*Node) *Element {
	nodePos := make([]float64, len(nodes))
	for i := range nodes {
		nodePos[i] = nodes[i].Xmain
	}
	return &Element{
		Nodes: nodes,
		NodePos: nodePos,
	}
}

func NewElement(pts []Point, weights []float64) *Element {
	e := &Element{}
	for _, p := range pts {
		e.NodePos = append(e.NodePos, p.X)
	}

	for i, p := range pts {
		xZeros := append([]float64{}, e.NodePos[:i]...)
		xZeros = append(xZeros, e.NodePos[i+1:]...)
		e.Nodes = append(e.Nodes, NewNode(p, xZeros, weights[i]))
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
