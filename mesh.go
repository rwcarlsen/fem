package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/gonum/integrate/quad"
	"github.com/gonum/matrix/mat64"
)

// Mesh represents a collection of elements constituting an approximation for
// a differential equation solution over a closed, contiguous volume.
type Mesh struct {
	// Elems is an (arbitrarily) ordered list of elements that make up the
	// mesh.
	Elems []Element
	// nodeIndex maps all nodes to a global index/ID
	nodeIndex map[Node]int
	// indexNode maps all global node indices to a list of nodes at the
	// corresponding position
	indexNode map[int][]Node
	// box is a helper to speed up the identification of elements that enclose
	// certain points in the mesh.  This helps lookups to be more performant.
	box *Box
}

// nodeId returns the global node id for the given element index and its local
// node index.
func (m *Mesh) nodeId(elem, node int) int {
	return m.nodeIndex[m.Elems[elem].Nodes()[node]]
}

type posHash [sha1.Size]byte

// hashX generates a unique identifier for each node position in a finite
// element mesh.  This is used to corroborate nodes at the same position x in
// different elements (i.e. nodes that are shared between elements).
func hashX(x []float64) posHash {
	var buf bytes.Buffer
	binary.Write(&buf, binary.BigEndian, x)
	return sha1.Sum(buf.Bytes())
}

// finalize generates node ID's and indices for mapping nodes to matrix
// indices.  It should be called after all elements have been added to the
// mesh.
func (m *Mesh) finalize() {
	if len(m.nodeIndex) > 0 {
		return
	}
	nextId := 0
	ids := map[posHash]int{}
	for _, e := range m.Elems {
		for _, n := range e.Nodes() {
			hx := hashX(n.X())
			if id, ok := ids[hx]; ok {
				m.nodeIndex[n] = id
				m.indexNode[id] = append(m.indexNode[id], n)
				continue
			}

			m.nodeIndex[n] = nextId
			m.indexNode[nextId] = append(m.indexNode[nextId], n)
			ids[hx] = nextId
			nextId++
		}
	}

	m.box = NewBox(m.Elems, 10, 10)
}

// AddElement is for adding custom-built elements to a mesh.  When all
// elements have been added.  Elements must form a single, contiguous domain
// (i.e.  with no holes/gaps).
func (m *Mesh) AddElement(e Element) error {
	if len(m.nodeIndex) > 0 {
		return fmt.Errorf("cannot add elements to a finalized mesh")
	}
	m.Elems = append(m.Elems, e)
	return nil
}

// NewMeshSimple1D creates a simply-connected mesh with nodes at the specified
// points and degree nodes per element.
func NewMeshSimple1D(nodePos []float64, degree int) (*Mesh, error) {
	m := &Mesh{nodeIndex: map[Node]int{}, indexNode: map[int][]Node{}}
	if (len(nodePos)-1)%(degree-1) != 0 {
		return nil, fmt.Errorf("incompatible mesh degree (%v) and node count (%v)", degree, len(nodePos))
	}

	nElems := (len(nodePos) - 1) / (degree - 1)
	for i := 0; i < nElems; i++ {
		xs := make([]float64, degree)
		for j := 0; j < degree; j++ {
			xs[j] = nodePos[i*(degree-1)+j]
		}
		m.AddElement(NewElementSimple1D(xs))
	}
	return m, nil
}

// Interpolate returns the finite element approximate for the solution at x.
// Solve must have been called before this for it to return meaningful
// results.
func (m *Mesh) Interpolate(x []float64) (float64, error) {
	elem, err := m.box.Find(x)
	if err != nil {
		return 0, err
	}
	return Interpolate(elem, x)
}

// reset renormalizes all the node shape functions in the mesh to one - i.e.
// resets and throws away any previously computed approximations via Solve.
func (m *Mesh) reset() {
	for _, e := range m.Elems {
		for _, n := range e.Nodes() {
			n.Set(1, 1)
		}
	}
}

// Solve computes the finite element approximation for the the differential
// equation represented by k.  This can be called multiple times with
// different kernels, but any previously computed DE approximations will be
// overwritten.  This solves the system K*u=f for u where K is the stiffness matrix
// and f is the force matrix.
func (m *Mesh) Solve(k Kernel) error {
	m.reset()
	A := m.StiffnessMatrix(k)
	b := m.ForceMatrix(k)
	var x mat64.Dense
	err := x.Solve(A, b)
	if err != nil {
		return err
	}

	r, _ := x.Dims()

	for i := 0; i < r; i++ {
		nodes := m.indexNode[i]
		for _, n := range nodes {
			n.Set(x.At(i, 0), 1)
		}
	}
	return nil
}

// ForceMatrix builds the matrix with one entry for each node representing the
// result of the integration terms of the weak form of the differential
// equation in k that do *not* include/depend on u(x).  This is the f column
// vector in the equation the K*u=f.
func (m *Mesh) ForceMatrix(k Kernel) *mat64.Dense {
	m.finalize()
	size := len(m.indexNode)
	mat := mat64.NewDense(size, 1, nil)
	for e, elem := range m.Elems {
		for i := range elem.Nodes() {
			v := elem.IntegrateForce(k, i)
			a := m.nodeId(e, i)
			mat.Set(a, 0, mat.At(a, 0)+v)
		}
	}
	return mat
}

// StiffnessMatrix builds the matrix with one entry for each combination of
// node test and weight functions representing the result of the integration
// terms of the weak form of the differential equation in k that
// include/depend on u(x).  This is the K matrix in the equation the K*u=f.
func (m *Mesh) StiffnessMatrix(k Kernel) *mat64.Dense {
	m.finalize()
	size := len(m.indexNode)
	mat := mat64.NewDense(size, size, nil)
	for e, elem := range m.Elems {
		for i := range elem.Nodes() {
			for j := i; j < len(elem.Nodes()); j++ {
				v := elem.IntegrateStiffness(k, i, j)
				a, b := m.nodeId(e, i), m.nodeId(e, j)
				mat.Set(a, b, mat.At(a, b)+v)
				if a != b {
					mat.Set(b, a, mat.At(a, b))
				}
			}
		}
	}
	return mat
}

// Node represents a node and its interpolant within a finite element.
type Node interface {
	// X returns the global/absolute position of the node.
	X() []float64
	// Sample returns the value of the node's shape/solution function at x.
	Sample(x []float64) float64
	// Weight returns the value of the node'd weight/test function at x.
	Weight(x []float64) float64
	// DerivSample returns the partial derivative for the dim'th dimension
	// of the node's shape function at x.
	DerivSample(x []float64, dim int) float64
	// DerivWeight returns the partial derivative for the dim'th dimension of
	// the node's weight function at x.
	DerivWeight(x []float64, dim int) float64
	// Set normalizes the node's shape/solution and weight/test function to be
	// equal to sample and weight at the node's position X().
	Set(sample, weight float64)
}

// LagrangeNode represents a finite element node.  It holds a polynomial shape
// function that can be sampled.  It represents a shape function of the
// following form:
//
//    (x - x2)    (x - x3)    (x - x4)
//    --------- * --------- * ---------  * ...
//    (x1 - x2)   (x1 - x3)   (x1 - x4)
//
type LagrangeNode struct {
	// Index identifies the interpolation point in Xvals where the node's
	// shape function is equal to U or W for the solution and weight
	// respectively.
	Index int
	// Xvals represent all the interpolation points where the node's
	// polynomial shape function is fixed/specified (i.e. either zero or U/W).
	// When constructing elements, all the points in Xvals must each have a
	// node that the Index is set to.
	Xvals []float64
	// U is the value of the node's solution shape function at Xvals[Index].
	U float64
	// W is the value of the node's weight shape function at Xvals[Index].
	W float64
}

// NewLagrangeNode returns a lagrange interpolating polynomial shape function
// backed node with U and W set to 1.0.  xs represents the interpolation
// points and xIndex identifies the point in xs where the polynomial is equal
// to U/W instead of zero.
func NewLagrangeNode(xIndex int, xs []float64) *LagrangeNode {
	return &LagrangeNode{
		Index: xIndex,
		Xvals: append([]float64{}, xs...),
		U:     1,
		W:     1,
	}
}

func (n *LagrangeNode) X() []float64 { return []float64{n.Xvals[n.Index]} }

func (n *LagrangeNode) Set(sample, weight float64) { n.U, n.W = sample, weight }

func (n *LagrangeNode) Sample(x []float64) float64 {
	xx, u := x[0], n.U
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return u
}

func (n *LagrangeNode) Weight(x []float64) float64 { return n.Sample(x) / n.U * n.W }

func (n *LagrangeNode) DerivSample(x []float64, dim int) float64 {
	xx, u := x[0], n.U
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()[0]-x0)*u + (xx-x0)/(n.X()[0]-x0)*dudx
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return dudx
}

func (n *LagrangeNode) DerivWeight(x []float64, dim int) float64 {
	xx, u := x[0], n.U
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()[0]-x0)*u + (xx-x0)/(n.X()[0]-x0)*dudx
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return dudx
}

// Element represents an element and provides integration and bounds related
// functionality required for approximating differential equation solutions.
type Element interface {
	// Nodes returns a persistent list of nodes that comprise this
	// element in no particular order but in consistent order.
	Nodes() []Node
	// IntegrateStiffness returns the result of the integration terms of the
	// weak form of the differential equation that include/depend on u(x) (the
	// solution or dependent variable).
	IntegrateStiffness(k Kernel, wNode, uNode int) float64
	// IntegrateForce returns the result of the integration terms of the weak
	// form of the differential equation that do *not* include/depend on u(x).
	IntegrateForce(k Kernel, wNode int) float64
	// Bounds returns a hyper-cubic bounding box defined by low and up values
	// in each dimension.
	Bounds() (low, up []float64)
	// Contains returns true if x is inside this element and false otherwise.
	Contains(x []float64) bool
}

// Element1D represents a 1D finite element.  It assumes len(x) == 1 (i.e.
// only one dimension of independent variables.
type Element1D struct {
	nodes []Node
}

func (e *Element1D) Bounds() (low, up []float64) { return []float64{e.left()}, []float64{e.right()} }

func (e *Element1D) Nodes() []Node { return e.nodes }

func (e *Element1D) Contains(x []float64) bool {
	xx := x[0]
	return e.left() <= xx && xx <= e.right()
}

func (e *Element1D) left() float64  { return e.nodes[0].X()[0] }
func (e *Element1D) right() float64 { return e.nodes[len(e.nodes)-1].X()[0] }

// NewElementSimple1D generates a lagrange polynomial interpolating element of
// degree len(xs)-1 using the values in xs as the interpolation points/nodes.
func NewElementSimple1D(xs []float64) *Element1D {
	e := &Element1D{}
	for i := range xs {
		n := NewLagrangeNode(i, xs)
		e.nodes = append(e.nodes, n)
	}
	return e
}

// Interpolate returns the (approximated) value of the function within the
// element at position x.  An error is returned if x is not contained inside
// the element.
func Interpolate(e Element, x []float64) (float64, error) {
	if !e.Contains(x) {
		return 0, fmt.Errorf("point %v is not inside the element", x)
	}
	u := 0.0
	for _, n := range e.Nodes() {
		u += n.Sample(x)
	}
	return u, nil
}

// Deriv returns the derivative of the element at x - i.e. the superposition
// of derivatives from each of the element nodes. An error is returned if x is
// not contained inside the element.
func Deriv(e Element, x []float64, dim int) (float64, error) {
	if !e.Contains(x) {
		return 0, fmt.Errorf("point %v is not inside the element", x)
	}
	u := 0.0
	for _, n := range e.Nodes() {
		u += n.DerivSample(x, dim)
	}
	return u, nil
}

func (e *Element1D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	w, u := e.nodes[wNode], e.nodes[uNode]

	fn := func(x float64) float64 {
		xs := []float64{x}
		pars := &KernelParams{X: xs, U: u.Sample(xs), GradU: u.DerivSample(xs, 0), W: w.Weight(xs), GradW: w.DerivWeight(xs, 0), Penalty: DefaultPenalty}
		return k.VolIntU(pars)
	}
	volU := quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)

	x1 := []float64{e.left()}
	x2 := []float64{e.right()}
	pars1 := &KernelParams{X: x1, U: u.Sample(x1), GradU: u.DerivSample(x1, 0), W: w.Weight(x1), GradW: w.DerivWeight(x1, 0), Penalty: DefaultPenalty}
	pars2 := &KernelParams{X: x2, U: u.Sample(x2), GradU: u.DerivSample(x2, 0), W: w.Weight(x2), GradW: w.DerivWeight(x2, 0), Penalty: DefaultPenalty}
	boundU1 := k.BoundaryIntU(pars1)
	boundU2 := k.BoundaryIntU(pars2)
	return volU + boundU1 + boundU2
}

func (e *Element1D) IntegrateForce(k Kernel, wNode int) float64 {
	w := e.nodes[wNode]

	fn := func(x float64) float64 {
		xvec := []float64{x}
		pars := &KernelParams{X: xvec, U: 0, GradU: 0, W: w.Weight(xvec), GradW: w.DerivWeight(xvec, 0), Penalty: DefaultPenalty}
		return k.VolInt(pars)
	}
	vol := quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)

	x1 := []float64{e.left()}
	x2 := []float64{e.right()}
	pars1 := &KernelParams{X: x1, U: 0, GradU: 0, W: w.Weight(x1), GradW: w.DerivWeight(x1, 0), Penalty: DefaultPenalty}
	pars2 := &KernelParams{X: x2, U: 0, GradU: 0, W: w.Weight(x2), GradW: w.DerivWeight(x2, 0), Penalty: DefaultPenalty}
	bound1 := k.BoundaryInt(pars1)
	bound2 := k.BoundaryInt(pars2)

	return vol + bound1 + bound2
}

// PrintFunc prints the element value and derivative in tab-separated form
// with nsamples evenly spaced over the element's domain (one sample per line)
// in the form:
//
//    [x]	[value]	[derivative]
//    ...
func (e *Element1D) PrintFunc(w io.Writer, nsamples int) {
	xrange := e.right() - e.left()
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := []float64{e.left() + xrange*float64(i)/float64(nsamples)}
		v, err := Interpolate(e, x)
		if err != nil {
			panic(err)
		}
		d, err := Deriv(e, x, 0)
		if err != nil {
			panic(err)
		}
		fmt.Fprintf(w, "%v\t%v\t%v\n", x, v, d)
	}
}

// PrintShapeFuncs prints the shape functions and their derivatives in
// tab-separated form with nsamples evenly spaced over the element's domain
// (one sample per line) in the form:
//
//    [x]	[LagrangeNode1-shape(x)]	[LagrangeNode1-shapederiv(x)]	[LagrangeNode2-shape(x)]
//    ...
func (e *Element1D) PrintShapeFuncs(w io.Writer, nsamples int) {
	xrange := e.right() - e.left()
	for i := -1 * nsamples / 10; i < nsamples+2*nsamples/10; i++ {
		x := []float64{e.left() + xrange*float64(i)/float64(nsamples)}
		fmt.Fprintf(w, "%v", x)
		for _, n := range e.nodes {
			if x[0] < e.left() || x[0] > e.right() {
				fmt.Fprintf(w, "\t0\t0")
			} else {
				fmt.Fprintf(w, "\t%v\t%v", n.Sample(x), n.DerivSample(x, 0))
			}
		}
		fmt.Fprintf(w, "\n")
	}
}
