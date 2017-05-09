package main

import (
	"fmt"
	"io"

	"github.com/gonum/integrate/quad"
)

// Element represents an element and provides integration and bounds related
// functionality required for approximating differential equation solutions.
type Element interface {
	// Nodes returns a persistent list of nodes that comprise this
	// element in no particular order but in stable/consistent order.
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

// Deriv returns the partial derivatives of the element at x for each
// dimension - i.e. the superposition of partial derivatives from each of the
// element nodes. An error is returned if x is not contained inside the
// element.
func Deriv(e Element, x []float64) ([]float64, error) {
	if !e.Contains(x) {
		return nil, fmt.Errorf("point %v is not inside the element", x)
	}
	u := e.Nodes()[0].DerivSample(x)
	for _, n := range e.Nodes()[1:] {
		subu := n.DerivSample(x)
		for i := range subu {
			u[i] += subu[i]
		}
	}
	return u, nil
}

// Element1D represents a 1D finite element.  It assumes len(x) == 1 (i.e.
// only one dimension of independent variables.
type Element1D struct {
	nodes []Node
}

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

func (e *Element1D) Bounds() (low, up []float64) { return []float64{e.left()}, []float64{e.right()} }

func (e *Element1D) Nodes() []Node { return e.nodes }

func (e *Element1D) Contains(x []float64) bool {
	xx := x[0]
	return e.left() <= xx && xx <= e.right()
}

func (e *Element1D) left() float64  { return e.nodes[0].X()[0] }
func (e *Element1D) right() float64 { return e.nodes[len(e.nodes)-1].X()[0] }

func (e *Element1D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	return e.integrateVol(k, wNode, uNode) + e.integrateBoundary(k, wNode, uNode)
}

func (e *Element1D) IntegrateForce(k Kernel, wNode int) float64 {
	return e.integrateVol(k, wNode, -1) + e.integrateBoundary(k, wNode, -1)
}

func (e *Element1D) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	var w, u Node = e.nodes[wNode], nil
	x1 := []float64{e.left()}
	x2 := []float64{e.right()}
	pars1 := &KernelParams{X: x1, W: w.Weight(x1), GradW: w.DerivWeight(x1)}
	pars2 := &KernelParams{X: x2, W: w.Weight(x2), GradW: w.DerivWeight(x2)}

	if uNode < 0 {
		return k.BoundaryInt(pars1) + k.BoundaryInt(pars2)
	}
	u = e.nodes[uNode]
	pars1.U = u.Sample(x1)
	pars1.GradU = u.DerivSample(x1)
	pars2.U = u.Sample(x2)
	pars2.GradU = u.DerivSample(x2)
	return k.BoundaryIntU(pars1) + k.BoundaryIntU(pars2)
}

func (n *LagrangeNode) coord(refx float64) float64 {
	return (e.left()*(1-refx) + e.right()*(1+refx)) / 2
}

func (e *Element1D) integrateVol(k Kernel, wNode, uNode int) float64 {
	fn := func(ref float64) float64 {
		xs := []float64{e.coord(ref)}
		var w, u Node = e.nodes[wNode], nil
		pars := &KernelParams{X: xs, W: w.Weight(xs), GradW: w.DerivWeight(xs)}
		if uNode < 0 {
			return k.VolInt(pars)
		}
		u = e.nodes[uNode]
		pars.U = u.Sample(xs)
		pars.GradU = u.DerivSample(xs)
		return k.VolIntU(pars)
	}
	return quad.Fixed(fn, -1, 1, len(e.nodes), quad.Legendre{}, 0)
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
		d, err := Deriv(e, x)
		if err != nil {
			panic(err)
		}
		fmt.Fprintf(w, "%v\t%v\t%v\n", x, v, d[0])
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
				fmt.Fprintf(w, "\t%v\t%v", n.Sample(x), n.DerivSample(x)[0])
			}
		}
		fmt.Fprintf(w, "\n")
	}
}

type Element2D struct {
	nodes      []Node
	trans      *ProjectiveTransform
	xmin, xmax float64
	ymin, ymax float64
}

func NewElementSimple2D(x1, y1, x2, y2, x3, y3, x4, y4 float64) (*Element2D, error) {
	n1, err := NewBilinQuadNode(x1, y1, x2, y2, x3, y3, x4, y4)
	if err != nil {
		return nil, err
	}
	n2, err := NewBilinQuadNode(x2, y2, x3, y3, x4, y4, x1, y1)
	if err != nil {
		return nil, err
	}
	n3, err := NewBilinQuadNode(x3, y3, x4, y4, x1, y1, x2, y2)
	if err != nil {
		return nil, err
	}
	n4, err := NewBilinQuadNode(x4, y4, x1, y1, x2, y2, x3, y3)
	if err != nil {
		return nil, err
	}

	return &Element2D{
		nodes: []Node{n1, n2, n3, n4},
		trans: n1.Transform,
		xmin:  min(x1, x2, x3, x4),
		xmax:  max(x1, x2, x3, x4),
		ymin:  min(y1, y2, y3, y4),
		ymax:  max(y1, y2, y3, y4),
	}, nil
}

func (e *Element2D) Bounds() (low, up []float64) {
	return []float64{e.xmin, e.ymin}, []float64{e.xmax, e.ymax}
}

func (e *Element2D) Nodes() []Node { return e.nodes }

func (e *Element2D) Contains(x []float64) bool {
	xx, yy := e.trans.Reverse(x[0], x[1])
	return -1 <= xx && xx <= 1 && -1 <= yy && yy <= 1
}

func (e *Element2D) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	var w, u Node = e.nodes[wNode], nil
	fnFactory := func(iFree int, fixedVar float64) func(x float64) float64 {
		xs := []float64{fixedVar, fixedVar}
		en := []float64{fixedVar, fixedVar}
		return func(x float64) float64 {
			en[iFree] = x
			ee, nn := en[0], en[1]
			xs[0], xs[1] = e.trans.Transform(ee, nn)
			jac := e.trans.ReverseJacobian(ee, nn)
			dxdfree := jac.At(iFree, 0)
			dydfree := jac.At(iFree, 1)
			pars := &KernelParams{X: xs, W: w.Weight(xs), GradW: w.DerivWeight(xs)}
			if uNode < 0 {
				return (dxdfree + dydfree) * k.BoundaryInt(pars)
			}
			u = e.nodes[uNode]
			pars.U = u.Sample(xs)
			pars.GradU = u.DerivSample(xs)
			return 1 / (dxdfree + dydfree) * k.BoundaryIntU(pars)
		}
	}

	xFree, yFree := 0, 1
	x1, x2, y1, y2 := -1.0, 1.0, -1.0, 1.0

	bound := 0.0
	bound += quad.Fixed(fnFactory(xFree, y1), x1, x2, 2, quad.Legendre{}, 0)
	bound += quad.Fixed(fnFactory(xFree, y2), x1, x2, 2, quad.Legendre{}, 0)
	bound += quad.Fixed(fnFactory(yFree, x1), y1, y2, 2, quad.Legendre{}, 0)
	bound += quad.Fixed(fnFactory(yFree, x2), y1, y2, 2, quad.Legendre{}, 0)
	return bound
}

func (e *Element2D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	return e.integrateVol(k, wNode, uNode) + e.integrateBoundary(k, wNode, uNode)
}

func (e *Element2D) IntegrateForce(k Kernel, wNode int) float64 {
	return e.integrateVol(k, wNode, -1) + e.integrateBoundary(k, wNode, -1)
}

func (e *Element2D) integrateVol(k Kernel, wNode, uNode int) float64 {
	outer := func(ee float64) float64 {
		inner := func(nn float64) float64 {
			xs := make([]float64, 2)
			xs[0], xs[1] = e.trans.Transform(ee, nn)
			jacdet := e.trans.ReverseJacobianDet(ee, nn)

			var w, u Node = e.nodes[wNode], nil
			pars := &KernelParams{X: xs, W: w.Weight(xs), GradW: w.DerivWeight(xs)}
			if uNode < 0 {
				return jacdet * k.VolInt(pars)
			}
			u = e.nodes[uNode]
			pars.U = u.Sample(xs)
			pars.GradU = u.DerivSample(xs)
			return jacdet * k.VolIntU(pars)
		}
		return quad.Fixed(inner, -1, 1, len(e.nodes), quad.Legendre{}, 0)
	}
	return quad.Fixed(outer, -1, 1, len(e.nodes), quad.Legendre{}, 0)
}
