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
	//elem := NewElementSimple(xs)
	//elem.PrintShapeFuncs(os.Stdout, n)
	//elem.PrintFunc(os.Stdout, n)

	xs = []float64{0, 1, 2, 3, 4, 5, 6}
	mesh, err := NewMeshSimple(xs, 3, Boundary{}, Boundary{})
	if err != nil {
		log.Fatal(err)
	}

	for i, elem := range mesh.Elems {
		fmt.Printf("elem %v\n", i)
		for _, n := range elem.Nodes {
			fmt.Printf("    node %p at x=%v\n", n, n.X())
		}
	}

	PrintStiffness(xs, []float64{7, 8, 9, 11, 13, 19}, 3)
	PrintStiffness([]float64{0, 1, 2}, []float64{7, 8}, 2)
}

func PrintStiffness(xs, ks []float64, degree int) {
	k := &SpringKernel{X: xs, K: ks}
	mesh, err := NewMeshSimple(xs, degree, Boundary{}, Boundary{})
	if err != nil {
		log.Fatal(err)
	}
	stiffness := mesh.StiffnessMatrix(k)
	fmt.Printf("%v\n", mat64.Formatted(stiffness))
}

type SpringKernel struct {
	X []float64
	K []float64
}

func (k *SpringKernel) VolIntU(p *KernelParams) float64 {
	for i := 1; i < len(k.X); i++ {
		if k.X[i-1] <= p.X && p.X <= k.X[i] {
			return p.GradW * p.GradU * k.K[i-1]
		}
	}
	return 1e100
}
func (k *SpringKernel) VolInt(p *KernelParams) float64       { return 0 }
func (k *SpringKernel) BoundaryIntU(p *KernelParams) float64 { return 0 }
func (k *SpringKernel) BoundaryInt(p *KernelParams) float64  { return 0 }

type KernelParams struct {
	X float64
	// U is value of the solution shape function if solving a linear system and
	// it is the current guess for the solution (i.e. shape function val times current solution guess).
	// For linear systems, an explicit solve will actually determine the nodal U values directly.
	// For nonlinear systems, an Newton or similar will be used to iterate toward better U guesses.
	U     float64
	GradU float64
	// W holds the value of the weight/test function
	W float64
	// GradW holds the derivative of the weight/test function
	GradW float64
	// Penalty represents a penalty factor for converting essential boundary
	// conditions to natural/traction boundary conditions.
	Penalty float64
}

// K
type Kerneler interface {
	// Kernel returns the value of the volumetric integration portion of the weak form
	// (i.e. everything but the boundary/surface integration).
	VolIntU(p *KernelParams) float64
	VolInt(p *KernelParams) float64
	BoundaryIntU(p *KernelParams) float64
	BoundaryInt(p *KernelParams) float64
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
	Elems []*Element
	// NodeIndex maps all nodes to a global index/ID
	NodeIndex map[*Node]int
	// IndexNode maps all global node indices to a list of nodes at the corresponding position
	IndexNode   map[int][]*Node
	Left, Right Boundary
}

// NewMeshSimple creates a simply-connected mesh with nodes at the specified points and degree nodes per element.
func NewMeshSimple(nodePos []float64, degree int, left, right Boundary) (*Mesh, error) {
	m := &Mesh{NodeIndex: map[*Node]int{}, IndexNode: map[int][]*Node{}, Left: left, Right: right}
	if (len(nodePos)-1)%(degree-1) != 0 {
		return nil, fmt.Errorf("incompatible mesh degree (%v) and node count (%v)", degree, len(nodePos))
	}

	nextId := 0
	ids := map[float64]int{}

	nElems := (len(nodePos) - 1) / (degree - 1)
	for i := 0; i < nElems; i++ {
		xs := make([]float64, degree)
		for j := 0; j < degree; j++ {
			xs[j] = nodePos[i*(degree-1)+j]
		}
		elem := NewElementSimple(xs)
		m.Elems = append(m.Elems, elem)
		for _, n := range elem.Nodes {
			if id, ok := ids[n.X()]; ok {
				m.NodeIndex[n] = id
				m.IndexNode[id] = append(m.IndexNode[id], n)
				continue
			}
			m.NodeIndex[n] = nextId
			m.IndexNode[nextId] = append(m.IndexNode[nextId], n)
			ids[n.X()] = nextId
			nextId++
		}
	}
	return m, nil
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

func (m *Mesh) StiffnessMatrix(k Kerneler) *mat64.Dense {
	size := len(m.IndexNode)
	mat := mat64.NewDense(size, size, nil)
	for _, e := range m.Elems {
		for i := range e.Nodes {
			for j := i; j < len(e.Nodes); j++ {
				w, u := e.Nodes[i], e.Nodes[j]
				a, b := m.NodeIndex[w], m.NodeIndex[u]
				fn := func(x float64) float64 {
					pars := &KernelParams{X: x, U: u.Sample(x), GradU: u.Deriv(x), W: w.SampleWeight(x), GradW: w.DerivWeight(x)}
					return k.VolIntU(pars)
				}
				volU := quad.Fixed(fn, e.Left(), e.Right(), len(e.Nodes), quad.Legendre{}, 0)
				x1 := e.Left()
				x2 := e.Right()
				pars1 := &KernelParams{X: x1, U: u.Sample(x1), GradU: u.Deriv(x1), W: w.SampleWeight(x1), GradW: w.DerivWeight(x1)}
				pars2 := &KernelParams{X: x2, U: u.Sample(x2), GradU: u.Deriv(x2), W: w.SampleWeight(x2), GradW: w.DerivWeight(x2)}
				boundU1 := k.BoundaryIntU(pars1)
				boundU2 := k.BoundaryIntU(pars2)
				mat.Set(a, b, mat.At(a, b)+volU+boundU1-boundU2)
				if a != b {
					mat.Set(b, a, mat.At(a, b))
				}
			}
		}
	}
	return mat
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
	Nodes []*Node
	// Connectivity matrix for Nodes - true if two node i (index into Nodes slice) is connected to node j.
	Conn [][]bool
}

func (e *Element) Left() float64  { return e.Nodes[0].X() }
func (e *Element) Right() float64 { return e.Nodes[len(e.Nodes)-1].X() }

func NewElementSimple(xs []float64) *Element {
	e := &Element{Conn: make([][]bool, len(xs))}
	for i := range xs {
		e.Conn[i] = make([]bool, len(xs))
		n := NewNode(i, xs)
		e.Nodes = append(e.Nodes, n)
		e.Conn[i][i] = true
		if i > 0 {
			e.Conn[i-1][i] = true
			e.Conn[i][i-1] = true
		}
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
