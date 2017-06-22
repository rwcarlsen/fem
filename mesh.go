package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"fmt"

	"github.com/rwcarlsen/fem/sparse"
)

// Mesh represents a collection of elements constituting an approximation for
// a differential equation solution over a closed, contiguous volume.
type Mesh struct {
	// Elems is an (arbitrarily) ordered list of elements that make up the
	// mesh.
	Elems []Element
	// notEdges stores a hint for each element that is true if the element is
	// not on a boundary.
	notEdges []bool
	// nodeIndex maps all nodes to a global index/ID
	nodeIndex map[*Node]int
	// indexNode maps all global node indices to a list of nodes at the
	// corresponding position
	indexNode map[int][]*Node
	// box is a helper to speed up the identification of elements that enclose
	// certain points in the mesh.  This helps lookups to be more performant.
	box *Box
	// Conv is the coordinate conversion function/logic used for calculating solutions at
	// arbitrary points on the mesh.
	Conv Converter
	// Solver is the solver to use - if nil, a two-pass Gaussian-Jordan elimination algorithm is
	// used.
	Solver sparse.Solver
	// Bandwidth is the maximum off-diagonal distance that will be used for solving - other
	// entries are assumed zero.  If bandwidth is zero, the full dense matrix will be
	// computed/solved.
	Bandwidth int
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
	if m.nodeIndex == nil {
		m.nodeIndex = map[*Node]int{}
		m.indexNode = map[int][]*Node{}
	}
	nextId := 0
	ids := map[posHash]int{}
	for _, e := range m.Elems {
		for _, n := range e.Nodes() {
			hx := hashX(n.X)
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

func (m *Mesh) NumDOF() int { return len(m.indexNode) }

// AddElement is for adding custom-built elements to a mesh.  When all
// elements have been added.  Elements must form a single, contiguous domain
// (i.e.  with no holes/gaps).  notEdge is a hint that, if true, the mesh assumes
// the element is not on an edge and can be skipped for boundary operations.
func (m *Mesh) AddElement(e Element, notEdge bool) error {
	if len(m.nodeIndex) > 0 {
		return fmt.Errorf("cannot add elements to a finalized mesh")
	}
	m.Elems = append(m.Elems, e)
	m.notEdges = append(m.notEdges, notEdge)
	return nil
}

// NewMeshStructured creates a structured mesh with hyper-rectangular lagrange elements of the
// specified order using a grid of points defined by axis-perpendicular planes defined by as a
// series of points in each dimension.  len(divs) is the number of dimensions with each dimension
// having a slice of points defining the points at which the planes intersect the axis.
func NewMeshStructured(order int, divs ...[]float64) (*Mesh, error) {
	ndim := len(divs)
	m := &Mesh{Conv: StructuredConverter}
	nelems := 1
	strides := make([]int, ndim) // strides for offsets into divs for section for each element
	for i, xs := range divs {
		if (len(xs)-1)%order != 0 {
			return nil, fmt.Errorf("incompatible mesh order (%v) and dim division count (dim %v nx=%v)", order, i, len(xs))
		} else if len(xs) < 2 {
			return nil, fmt.Errorf("simple 2D mesh requires at least two divisions in each dimension")
		}
		strides[i] = nelems
		nelems *= (len(xs) - 1) / order
	}

	nnodes := pow(order+1, ndim)

	indices := make([]int, ndim)
	points := make([][]float64, nnodes)
	// substrides for offsets into divs on top of strides for each node in an element
	nodestrides := make([][]int, ndim)
	stride := 1
	for d := range nodestrides {
		nodestrides[d] = make([]int, nnodes)
		for i := range nodestrides[d] {
			nodestrides[d][i] = (i / stride) % (order + 1)
		}
		stride *= order + 1
	}

	shapecache := LagrangeNDCache{}
	elemcache := NewElementCache()

	for count := 0; count < nelems; count++ {
		for d, stride := range strides {
			indices[d] = ((count / stride) % ((len(divs[d]) - 1) / order)) * order
		}

		edge := false
		for i := range points {
			// each element needs its own slices so they aren't overwriting each other
			points[i] = make([]float64, ndim)
			for d, index := range indices {
				offset := index + nodestrides[d][i]
				if offset == 0 || offset == len(divs[d])-1 {
					edge = true
				}
				points[i][d] = divs[d][offset]
			}
		}
		e := NewElementND(order, elemcache, shapecache, points...)
		e.Conv = StructuredConverter
		m.AddElement(e, !edge)
	}
	return m, nil
}

// Interpolate returns the finite element approximate for the solution at x.
// Solve must have been called before this for it to return meaningful
// results.
func (m *Mesh) Interpolate(x []float64) (float64, error) {
	if m.Conv == nil {
		m.Conv = OptimConverter
	}
	elem, err := m.box.Find(x)
	if err != nil {
		return 0, err
	}
	refx, err := m.Conv(elem, x)
	if err != nil {
		return 0, err
	}
	return Interpolate(elem, refx), nil
}

// reset renormalizes all the node shape functions in the mesh to one - i.e.
// resets and throws away any previously computed approximations via Solve.
func (m *Mesh) reset() {
	for _, e := range m.Elems {
		for _, n := range e.Nodes() {
			n.U = 1
			n.W = 1
		}
	}
}

// Solve computes the finite element approximation for the the differential
// equation represented by k.  This can be called multiple times with
// different kernels, but any previously computed DE approximations will be
// overwritten.  This solves the system K*u=f for u where K is the stiffness matrix
// and f is the force matrix.
func (m *Mesh) Solve(k Kernel) error {
	solver := m.Solver
	if solver == nil {
		solver = sparse.GaussJordan{}
	}

	m.reset()
	A := m.StiffnessMatrix(k)
	b := m.ForceVector(k)

	// eliminate nonzeros in all columns of known/dirichlet dofs.
	knowns := m.knowns(k)
	for i := range knowns {
		sparse.ApplyPivot(A, b, i, i, -1, nil)
		sparse.ApplyPivot(A, b, i, i, 1, nil)
	}

	// remove knowns (i.e. dirichlet BCs) from matrix system to preserve symmetry
	size := m.NumDOF() - len(knowns)
	AA := sparse.NewSparse(size)
	bb := make([]float64, size)

	subindex := 0
	subindices := make([]int, m.NumDOF())
	rsubindices := make([]int, size)
	for i := 0; i < m.NumDOF(); i++ {
		if _, ok := knowns[i]; !ok {
			subindices[i] = subindex
			rsubindices[subindex] = i
			subindex++
		}
	}

	for i := 0; i < m.NumDOF(); i++ {
		if _, ok := knowns[i]; ok {
			continue
		}

		bb[subindices[i]] = b[i]
		for _, nonzero := range A.SweepRow(i) {
			if _, ok := knowns[nonzero.J]; ok {
				continue
			}
			AA.Set(subindices[i], subindices[nonzero.J], nonzero.Val)
		}
	}

	// solve symmetric system
	x, err := solver.Solve(AA, bb)
	if err != nil {
		return err
	}

	// populate mesh with node/DOF solution values from solver
	for i, val := range x {
		nodes := m.indexNode[rsubindices[i]]
		for _, n := range nodes {
			n.U = val
			n.W = 1
		}
	}
	// populate node/DOF solutions from known/dirichlet conditions
	for i, val := range knowns {
		nodes := m.indexNode[i]
		for _, n := range nodes {
			n.U = val
			n.W = 1
		}
	}

	return nil
}

func (m *Mesh) knowns(k Kernel) map[int]float64 {
	m.finalize()
	knowns := make(map[int]float64)
	for e, elem := range m.Elems {
		for i, n := range elem.Nodes() {
			if is, val := k.IsDirichlet(n.X); is {
				knowns[m.nodeId(e, i)] = val
			}
		}
	}

	return knowns
}

// ForceVector builds the matrix with one entry for each node representing the
// result of the integration terms of the weak form of the differential
// equation in k that do *not* include/depend on u(x).  This is the f column
// vector in the equation the K*u=f.
func (m *Mesh) ForceVector(k Kernel) []float64 {
	m.finalize()
	size := m.NumDOF()
	f := make([]float64, size)
	for e, elem := range m.Elems {
		for i, n := range elem.Nodes() {
			a := m.nodeId(e, i)
			if ok, v := k.IsDirichlet(n.X); ok {
				f[a] = v
				continue
			}
			f[a] += elem.IntegrateForce(k, i, m.notEdges[e])
		}
	}
	return f
}

// StiffnessMatrix builds the matrix with one entry for each combination of
// node test and weight functions representing the result of the integration
// terms of the weak form of the differential equation in k that
// include/depend on u(x).  This is the K matrix in the equation the K*u=f.
func (m *Mesh) StiffnessMatrix(k Kernel) *sparse.Sparse {
	m.finalize()
	size := m.NumDOF()
	mat := sparse.NewSparse(size)

	for e, elem := range m.Elems {
		for i, n := range elem.Nodes() {
			a := m.nodeId(e, i)
			for j := i; j < len(elem.Nodes()); j++ {
				b := m.nodeId(e, j)
				if m.Bandwidth > 0 && absInt(a-b) > m.Bandwidth {
					continue
				}
				v := elem.IntegrateStiffness(k, i, j, m.notEdges[e])
				mat.Set(a, b, mat.At(a, b)+v)
				mat.Set(b, a, mat.At(a, b))
				if ok, _ := k.IsDirichlet(n.X); ok {
					for _, nonzero := range mat.SweepRow(a) {
						mat.Set(a, nonzero.J, 0.0)
					}
					mat.Set(a, a, 1.0)
					continue
				}
			}
		}
	}
	return mat
}
