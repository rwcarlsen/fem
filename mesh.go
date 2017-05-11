package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// Mesh represents a collection of elements constituting an approximation for
// a differential equation solution over a closed, contiguous volume.
type Mesh struct {
	// Elems is an (arbitrarily) ordered list of elements that make up the
	// mesh.
	Elems []Element
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
// (i.e.  with no holes/gaps).
func (m *Mesh) AddElement(e Element) error {
	if len(m.nodeIndex) > 0 {
		return fmt.Errorf("cannot add elements to a finalized mesh")
	}
	m.Elems = append(m.Elems, e)
	return nil
}

// NewMeshSimple1D creates a simply-connected mesh with nodes at the specified
// points and order specifies the polynomial shape function order used in each element to
// approximate the solution.
func NewMeshSimple1D(nodePos []float64, order int) (*Mesh, error) {
	m := &Mesh{nodeIndex: map[*Node]int{}, indexNode: map[int][]*Node{}}
	if (len(nodePos)-1)%order != 0 {
		return nil, fmt.Errorf("incompatible mesh order (%v) and node count (%v)", order, len(nodePos))
	}

	nElems := (len(nodePos) - 1) / order
	for i := 0; i < nElems; i++ {
		xs := make([]float64, order+1)
		for j := 0; j < order+1; j++ {
			xs[j] = nodePos[i*order+j]
		}
		m.AddElement(NewElementSimple1D(xs))
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
	m.reset()
	A := m.StiffnessMatrix(k)
	b := m.ForceMatrix(k)

	var u mat64.Vector
	if err := u.SolveVec(A, b); err != nil {
		return err
	}

	for i := 0; i < u.Len(); i++ {
		nodes := m.indexNode[i]
		for _, n := range nodes {
			n.U = u.At(i, 0)
			n.W = 1
		}
	}
	return nil
}

func (m *Mesh) SolveIter(k Kernel, maxiter int, tol float64) (iter int, err error) {
	m.reset()
	b := m.ForceMatrix(k)

	prev := mat64.NewVector(m.NumDOF(), nil)
	soln := mat64.NewVector(m.NumDOF(), nil)

	n := 0
	acceleration := 1.7 // between 1.0 and 2.0
	for ; n < maxiter; n++ {
		for i := 0; i < m.NumDOF(); i++ {
			row := m.StiffnessRow(k, i)
			xold := soln.At(i, 0)
			soln.SetVec(i, 0)
			xnew := (1-acceleration)*xold +
				acceleration/row.At(i, 0)*(b.At(i, 0)-mat64.Dot(row, soln))
			soln.SetVec(i, xnew)
		}

		var diff mat64.Vector
		diff.SubVec(soln, prev)
		if er := mat64.Norm(&diff, 2) / mat64.Norm(soln, 2); er < tol {
			break
		}
		prev.CloneVec(soln)
	}

	for i := 0; i < soln.Len(); i++ {
		nodes := m.indexNode[i]
		for _, n := range nodes {
			n.U = soln.At(i, 0)
			n.W = 1
		}
	}
	return n, nil
}

// ForceMatrix builds the matrix with one entry for each node representing the
// result of the integration terms of the weak form of the differential
// equation in k that do *not* include/depend on u(x).  This is the f column
// vector in the equation the K*u=f.
func (m *Mesh) ForceMatrix(k Kernel) *mat64.Vector {
	m.finalize()
	size := m.NumDOF()
	mat := mat64.NewVector(size, nil)
	for e, elem := range m.Elems {
		for i, n := range elem.Nodes() {
			a := m.nodeId(e, i)
			if ok, v := k.IsDirichlet(n.X); ok {
				mat.SetVec(a, v)
				continue
			}
			v := elem.IntegrateForce(k, i)
			mat.SetVec(a, mat.At(a, 0)+v)
		}
	}
	return mat
}

func (m *Mesh) StiffnessRow(k Kernel, row int) *mat64.Vector {
	m.finalize()
	size := m.NumDOF()
	mat := mat64.NewVector(size, nil)
	for e, elem := range m.Elems {
		for i, n := range elem.Nodes() {
			a := m.nodeId(e, i)
			if a != row {
				continue
			} else if ok, _ := k.IsDirichlet(n.X); ok {
				mat.SetVec(row, 1.0)
				return mat
			}

			for j := 0; j < len(elem.Nodes()); j++ {
				b := m.nodeId(e, j)
				v := elem.IntegrateStiffness(k, i, j)
				mat.SetVec(b, mat.At(b, 0)+v)
			}
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
	size := m.NumDOF()
	mat := mat64.NewDense(size, size, nil)
	for e, elem := range m.Elems {
		for i, n := range elem.Nodes() {
			for j := i; j < len(elem.Nodes()); j++ {
				a, b := m.nodeId(e, i), m.nodeId(e, j)
				v := elem.IntegrateStiffness(k, i, j)
				mat.Set(a, b, mat.At(a, b)+v)
				mat.Set(b, a, mat.At(a, b))
				if ok, _ := k.IsDirichlet(n.X); ok {
					for c := 0; c < len(elem.Nodes()); c++ {
						mat.Set(a, c, 0.0)
					}
					mat.Set(a, a, 1.0)
					continue
				}
			}
		}
	}
	return mat
}
