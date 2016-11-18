package main

import (
	"fmt"
	"io"

	"github.com/gonum/integrate/quad"
	"github.com/gonum/matrix/mat64"
)

type Mesh struct {
	Elems []*Element
	// NodeIndex maps all nodes to a global index/ID
	NodeIndex map[*Node]int
	// IndexNode maps all global node indices to a list of nodes at the corresponding position
	IndexNode map[int][]*Node
}

// Finalize generates node ID's and indices for mapping nodes to matrix
// indices.  It should be called after all elements have been added to the
// mesh.
func (m *Mesh) Finalize() {
	if len(m.NodeIndex) > 0 {
		return
	}
	nextId := 0
	ids := map[float64]int{}
	for _, e := range m.Elems {
		for _, n := range e.Nodes {
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
}

// AddElement is for adding custom-built elements to a mesh.  When all
// elements have been added.  New elements
func (m *Mesh) AddElement(e *Element) error {
	if len(m.NodeIndex) > 0 {
		return fmt.Errorf("cannot add elements to a finalized mesh")
	}
	m.Elems = append(m.Elems, e)
	return nil
}

// NewMeshSimple creates a simply-connected mesh with nodes at the specified
// points and degree nodes per element. The returned mesh has been finalized.
func NewMeshSimple(nodePos []float64, degree int) (*Mesh, error) {
	m := &Mesh{NodeIndex: map[*Node]int{}, IndexNode: map[int][]*Node{}}
	if (len(nodePos)-1)%(degree-1) != 0 {
		return nil, fmt.Errorf("incompatible mesh degree (%v) and node count (%v)", degree, len(nodePos))
	}

	nElems := (len(nodePos) - 1) / (degree - 1)
	for i := 0; i < nElems; i++ {
		xs := make([]float64, degree)
		for j := 0; j < degree; j++ {
			xs[j] = nodePos[i*(degree-1)+j]
		}
		m.AddElement(NewElementSimple(xs))
	}
	m.Finalize()
	return m, nil
}

func (m *Mesh) Interpolate(x float64) float64 {
	for _, e := range m.Elems {
		if e.Left() <= x && x <= e.Right() {
			return e.Interpolate(x)
		}
	}
	panic("cannot interpolate out of bounds on mesh")
}

func (m *Mesh) ForceMatrix(k Kernel) *mat64.Dense {
	m.Finalize()
	size := len(m.IndexNode)
	mat := mat64.NewDense(size, 1, nil)
	for _, e := range m.Elems {
		for i := range e.Nodes {
			w := e.Nodes[i]

			fn := func(x float64) float64 {
				pars := &KernelParams{X: x, U: 0, GradU: 0, W: w.SampleWeight(x), GradW: w.DerivWeight(x), Penalty: DefaultPenalty}
				return k.VolInt(pars)
			}
			volU := quad.Fixed(fn, e.Left(), e.Right(), len(e.Nodes), quad.Legendre{}, 0)

			x1 := e.Left()
			x2 := e.Right()
			pars1 := &KernelParams{X: x1, U: 0, GradU: 0, W: w.SampleWeight(x1), GradW: w.DerivWeight(x1), Penalty: DefaultPenalty}
			pars2 := &KernelParams{X: x2, U: 0, GradU: 0, W: w.SampleWeight(x2), GradW: w.DerivWeight(x2), Penalty: DefaultPenalty}
			boundU1 := k.BoundaryInt(pars1)
			boundU2 := k.BoundaryInt(pars2)

			a := m.NodeIndex[w]
			mat.Set(a, 0, mat.At(a, 0)+volU+boundU1+boundU2)
		}
	}
	return mat
}

func (m *Mesh) StiffnessMatrix(k Kernel) *mat64.Dense {
	m.Finalize()
	size := len(m.IndexNode)
	mat := mat64.NewDense(size, size, nil)
	for _, e := range m.Elems {
		for i := range e.Nodes {
			for j := i; j < len(e.Nodes); j++ {
				w, u := e.Nodes[i], e.Nodes[j]

				fn := func(x float64) float64 {
					pars := &KernelParams{X: x, U: u.Sample(x), GradU: u.Deriv(x), W: w.SampleWeight(x), GradW: w.DerivWeight(x), Penalty: DefaultPenalty}
					return k.VolIntU(pars)
				}
				volU := quad.Fixed(fn, e.Left(), e.Right(), len(e.Nodes), quad.Legendre{}, 0)

				x1 := e.Left()
				x2 := e.Right()
				pars1 := &KernelParams{X: x1, U: u.Sample(x1), GradU: u.Deriv(x1), W: w.SampleWeight(x1), GradW: w.DerivWeight(x1), Penalty: DefaultPenalty}
				pars2 := &KernelParams{X: x2, U: u.Sample(x2), GradU: u.Deriv(x2), W: w.SampleWeight(x2), GradW: w.DerivWeight(x2), Penalty: DefaultPenalty}
				boundU1 := k.BoundaryIntU(pars1)
				boundU2 := k.BoundaryIntU(pars2)

				a, b := m.NodeIndex[w], m.NodeIndex[u]
				mat.Set(a, b, mat.At(a, b)+volU+boundU1+boundU2)
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
	// Index identifies the interpolation point in Xvals where the node's
	// shape function is equal to Val
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

// X returns the x interpolation point where the node's shape function is
// non-zero.
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
}

func (e *Element) Left() float64  { return e.Nodes[0].X() }
func (e *Element) Right() float64 { return e.Nodes[len(e.Nodes)-1].X() }

// NewElementSimple generates a lagrange polynomial interpolating element of
// degree len(xs)-1 using the values in xs as the interpolation points/nodes.
func NewElementSimple(xs []float64) *Element {
	e := &Element{}
	for i := range xs {
		n := NewNode(i, xs)
		e.Nodes = append(e.Nodes, n)
	}
	return e
}

// Interpolate returns the value of the element at x - i.e. the superposition
// of samples from each of the element nodes.
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

// Deriv returns the derivative of the element at x - i.e. the superposition
// of derivatives from each of the element nodes.
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

// PrintFunc prints the element value and derivative in tab-separated form
// with nsamples evenly spaced over the element's domain (one sample per line)
// in the form:
//
//    [x]	[value]	[derivative]
//    ...
func (e *Element) PrintFunc(w io.Writer, nsamples int) {
	xrange := e.Right() - e.Left()
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := e.Left() + xrange*float64(i)/float64(nsamples)
		fmt.Fprintf(w, "%v\t%v\t%v\n", x, e.Interpolate(x), e.Deriv(x))
	}
}

// PrintShapeFuncs prints the shape functions and their derivatives in
// tab-separated form with nsamples evenly spaced over the element's domain
// (one sample per line) in the form:
//
//    [x]	[Node1-shape(x)]	[Node1-shapederiv(x)]	[Node2-shape(x)]
//    ...
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
