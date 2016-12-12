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
	w, u := e.nodes[wNode], e.nodes[uNode]

	fn := func(x float64) float64 {
		xs := []float64{x}
		pars := &KernelParams{
			X: xs, U: u.Sample(xs), W: w.Weight(xs),
			GradU: []float64{u.DerivSample(xs, 0)},
			GradW: []float64{w.DerivWeight(xs, 0)},
		}
		return k.VolIntU(pars)
	}
	volU := quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)

	x1 := []float64{e.left()}
	x2 := []float64{e.right()}
	pars1 := &KernelParams{
		X: x1, U: u.Sample(x1), W: w.Weight(x1),
		GradU: []float64{u.DerivSample(x1, 0)},
		GradW: []float64{w.DerivWeight(x1, 0)},
	}
	pars2 := &KernelParams{
		X: x2, U: u.Sample(x2), W: w.Weight(x2),
		GradU: []float64{u.DerivSample(x2, 0)},
		GradW: []float64{w.DerivWeight(x2, 0)},
	}
	boundU1 := k.BoundaryIntU(pars1)
	boundU2 := k.BoundaryIntU(pars2)
	return volU + boundU1 + boundU2
}

func (e *Element1D) IntegrateForce(k Kernel, wNode int) float64 {
	w := e.nodes[wNode]

	fn := func(x float64) float64 {
		xvec := []float64{x}
		pars := &KernelParams{
			X: xvec, U: 0, W: w.Weight(xvec),
			GradU: []float64{0},
			GradW: []float64{w.DerivWeight(xvec, 0)},
		}
		return k.VolInt(pars)
	}
	vol := quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)

	x1 := []float64{e.left()}
	x2 := []float64{e.right()}
	pars1 := &KernelParams{
		X: x1, U: 0, W: w.Weight(x1),
		GradU: []float64{0},
		GradW: []float64{w.DerivWeight(x1, 0)},
	}
	pars2 := &KernelParams{
		X: x2, U: 0, W: w.Weight(x2),
		GradU: []float64{0},
		GradW: []float64{w.DerivWeight(x2, 0)},
	}
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

type Element2D struct {
	nodes []Node
}

func (e *Element2D) Bounds() (low, up []float64) {
	return []float64{e.x1(), e.y1()}, []float64{e.x2(), e.y2()}
}

func (e *Element2D) Nodes() []Node { return e.nodes }

func (e *Element2D) Contains(x []float64) bool {
	xx, yy := x[0], x[1]
	return e.x1() <= xx && xx <= e.x2() && e.y1() <= yy && yy <= e.y2()
}

func (e *Element2D) x1() float64 { return e.nodes[0].X()[0] }
func (e *Element2D) y1() float64 { return e.nodes[0].X()[1] }
func (e *Element2D) x2() float64 { return e.nodes[len(e.nodes)-1].X()[0] }
func (e *Element2D) y2() float64 { return e.nodes[len(e.nodes)-1].X()[1] }

func (e *Element2D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	panic("unimplemented")
	//w, u := e.nodes[wNode], e.nodes[uNode]

	//fn := func(x float64) float64 {
	//	xs := []float64{x}
	//	pars := &KernelParams{X: xs, U: u.Sample(xs), GradU: u.DerivSample(xs, 0), W: w.Weight(xs), GradW: w.DerivWeight(xs, 0)}
	//	return k.VolIntU(pars)
	//}
	//volU := quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)

	//x1 := []float64{e.left()}
	//x2 := []float64{e.right()}
	//pars1 := &KernelParams{X: x1, U: u.Sample(x1), GradU: u.DerivSample(x1, 0), W: w.Weight(x1), GradW: w.DerivWeight(x1, 0)}
	//pars2 := &KernelParams{X: x2, U: u.Sample(x2), GradU: u.DerivSample(x2, 0), W: w.Weight(x2), GradW: w.DerivWeight(x2, 0)}
	//boundU1 := k.BoundaryIntU(pars1)
	//boundU2 := k.BoundaryIntU(pars2)
	//return volU + boundU1 + boundU2
}

func (e *Element2D) IntegrateForce(k Kernel, wNode int) float64 {
	panic("unimplemented")
	//w := e.nodes[wNode]

	//fn := func(x float64) float64 {
	//	xvec := []float64{x}
	//	pars := &KernelParams{X: xvec, U: 0, GradU: 0, W: w.Weight(xvec), GradW: w.DerivWeight(xvec, 0)}
	//	return k.VolInt(pars)
	//}
	//vol := quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)

	//x1 := []float64{e.left()}
	//x2 := []float64{e.right()}
	//pars1 := &KernelParams{X: x1, U: 0, GradU: 0, W: w.Weight(x1), GradW: w.DerivWeight(x1, 0)}
	//pars2 := &KernelParams{X: x2, U: 0, GradU: 0, W: w.Weight(x2), GradW: w.DerivWeight(x2, 0)}
	//bound1 := k.BoundaryInt(pars1)
	//bound2 := k.BoundaryInt(pars2)

	//return vol + bound1 + bound2
}
