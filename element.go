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
	Nodes() []*Node
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
func Interpolate(e Element, refx []float64) (float64, error) {
	u := 0.0
	for _, n := range e.Nodes() {
		u += n.Value(refx)
	}
	return u, nil
}

// Deriv returns the partial derivatives of the element at x for each
// dimension - i.e. the superposition of partial derivatives from each of the
// element nodes. An error is returned if x is not contained inside the
// element.
func Deriv(e Element, refx []float64) ([]float64, error) {
	u := e.Nodes()[0].ValueDeriv(refx)
	for _, n := range e.Nodes()[1:] {
		subu := n.ValueDeriv(refx)
		for i := range subu {
			u[i] += subu[i]
		}
	}
	return u, nil
}

// Element1D represents a 1D finite element.  It assumes len(x) == 1 (i.e.
// only one dimension of independent variables.
type Element1D struct {
	Nds []*Node
}

// NewElementSimple1D generates a lagrange polynomial interpolating element of
// degree len(xs)-1 using the values in xs as the interpolation points/nodes.
func NewElementSimple1D(xs []float64) *Element1D {
	e := &Element1D{}
	for i := range xs {
		n := &Node{X: xs, U: 1.0, W: 1.0, ShapeFunc: Lagrange1D{Index: i, Order: len(xs) - 1}}
		e.Nds = append(e.Nds, n)
	}
	return e
}

func (e *Element1D) Bounds() (low, up []float64) { return []float64{e.left()}, []float64{e.right()} }

func (e *Element1D) Nodes() []*Node { return e.Nds }

func (e *Element1D) Contains(x []float64) bool {
	xx := x[0]
	return e.left() <= xx && xx <= e.right()
}

func (e *Element1D) left() float64  { return e.Nds[0].X[0] }
func (e *Element1D) right() float64 { return e.Nds[len(e.Nds)-1].X[0] }

func (e *Element1D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	return e.integrateVol(k, wNode, uNode) + e.integrateBoundary(k, wNode, uNode)
}

func (e *Element1D) IntegrateForce(k Kernel, wNode int) float64 {
	return e.integrateVol(k, wNode, -1) + e.integrateBoundary(k, wNode, -1)
}

func (e *Element1D) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	var refLeft = []float64{-1}
	var refRight = []float64{1}

	var w, u *Node = e.Nds[wNode], nil
	x1 := []float64{e.left()}
	x2 := []float64{e.right()}
	pars1 := &KernelParams{X: x1, W: w.Weight(refLeft), GradW: w.WeightDeriv(refRight)}
	pars2 := &KernelParams{X: x2, W: w.Weight(refLeft), GradW: w.WeightDeriv(refRight)}

	if uNode < 0 {
		return k.BoundaryInt(pars1) + k.BoundaryInt(pars2)
	}
	u = e.Nds[uNode]
	pars1.U = u.Value(refLeft)
	pars1.GradU = u.ValueDeriv(refLeft)
	pars2.U = u.Value(refRight)
	pars2.GradU = u.ValueDeriv(refRight)
	return k.BoundaryIntU(pars1) + k.BoundaryIntU(pars2)
}

func (e *Element1D) coord(refx float64) float64 {
	return (e.left()*(1-refx) + e.right()*(1+refx)) / 2
}

func (e *Element1D) integrateVol(k Kernel, wNode, uNode int) float64 {
	fn := func(ref float64) float64 {
		refxs := []float64{ref}
		xs := []float64{e.coord(ref)}
		var w, u *Node = e.Nds[wNode], nil
		pars := &KernelParams{X: xs, W: w.Weight(refxs), GradW: w.WeightDeriv(refxs)}
		if uNode < 0 {
			return k.VolInt(pars)
		}
		u = e.Nds[uNode]
		pars.U = u.Value(refxs)
		pars.GradU = u.ValueDeriv(refxs)
		return k.VolIntU(pars)
	}
	return quad.Fixed(fn, -1, 1, len(e.Nds), quad.Legendre{}, 0)
}

// PrintFunc prints the element value and derivative in tab-separated form
// with nsamples evenly spaced over the element's domain (one sample per line)
// in the form:
//
//    [x]	[value]	[derivative]
//    ...
func (e *Element1D) PrintFunc(w io.Writer, nsamples int) {
	drefx := 2 / (float64(nsamples) - 1)
	for i := 0; i < nsamples; i++ {
		refx := []float64{-1 + float64(i)*drefx}
		x := e.coord(refx[0])

		v, err := Interpolate(e, refx)
		if err != nil {
			panic(err)
		}
		d, err := Deriv(e, refx)
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
	drefx := 2 / (float64(nsamples) - 1)
	for i := 0; i < nsamples; i++ {
		refx := []float64{-1 + float64(i)*drefx}
		x := e.coord(refx[0])
		fmt.Fprintf(w, "%v", x)
		for _, n := range e.Nds {
			if x < e.left() || x > e.right() {
				fmt.Fprintf(w, "\t0\t0")
			} else {
				fmt.Fprintf(w, "\t%v\t%v", n.Value(refx), n.ValueDeriv(refx)[0])
			}
		}
		fmt.Fprintf(w, "\n")
	}
}

type Element2D struct {
	Nds []*Node
}

func (e *Element2D) xmin() float64 { return e.min(0, true) }
func (e *Element2D) xmax() float64 { return e.min(0, false) }
func (e *Element2D) ymin() float64 { return e.min(1, true) }
func (e *Element2D) ymax() float64 { return e.min(1, false) }

func (e *Element2D) min(coord int, less bool) float64 {
	extreme := e.Nds[coord].X[0]
	for _, n := range e.Nds[1:] {
		if n.X[coord] < extreme && less || n.X[coord] > extreme && !less {
			extreme = n.X[coord]
		}
	}
	return extreme
}

func NewElementSimple2D(x1, y1, x2, y2, x3, y3, x4, y4 float64) (*Element2D, error) {
	panic("unimplemented")
	//n1, err := NewBilinQuadNode(x1, y1, x2, y2, x3, y3, x4, y4)
	//if err != nil {
	//	return nil, err
	//}
	//n2, err := NewBilinQuadNode(x2, y2, x3, y3, x4, y4, x1, y1)
	//if err != nil {
	//	return nil, err
	//}
	//n3, err := NewBilinQuadNode(x3, y3, x4, y4, x1, y1, x2, y2)
	//if err != nil {
	//	return nil, err
	//}
	//n4, err := NewBilinQuadNode(x4, y4, x1, y1, x2, y2, x3, y3)
	//if err != nil {
	//	return nil, err
	//}

	//return &Element2D{
	//	nodes: []Node{n1, n2, n3, n4},
	//}, nil
}

func (e *Element2D) Bounds() (low, up []float64) {
	return []float64{e.xmin(), e.ymin()}, []float64{e.xmax(), e.ymax()}
}

func (e *Element2D) Nodes() []*Node { return e.Nds }

func (e *Element2D) Contains(x []float64) bool {
	panic("unimplemented")
}

func (e *Element2D) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	panic("unimplemented")
	//	var w, u Node = e.Nodes[wNode], nil
	//	fnFactory := func(iFree int, fixedVar float64) func(x float64) float64 {
	//		xs := []float64{fixedVar, fixedVar}
	//		en := []float64{fixedVar, fixedVar}
	//		return func(x float64) float64 {
	//			en[iFree] = x
	//			ee, nn := en[0], en[1]
	//			xs[0], xs[1] = e.trans.Transform(ee, nn)
	//			jac := e.trans.ReverseJacobian(ee, nn)
	//			dxdfree := jac.At(iFree, 0)
	//			dydfree := jac.At(iFree, 1)
	//			pars := &KernelParams{X: xs, W: w.Weight(xs), GradW: w.DerivWeight(xs)}
	//			if uNode < 0 {
	//				return (dxdfree + dydfree) * k.BoundaryInt(pars)
	//			}
	//			u = e.Nodes[uNode]
	//			pars.U = u.Sample(xs)
	//			pars.GradU = u.DerivSample(xs)
	//			return 1 / (dxdfree + dydfree) * k.BoundaryIntU(pars)
	//		}
	//	}
	//
	//	xFree, yFree := 0, 1
	//	x1, x2, y1, y2 := -1.0, 1.0, -1.0, 1.0
	//
	//	bound := 0.0
	//	bound += quad.Fixed(fnFactory(xFree, y1), x1, x2, 2, quad.Legendre{}, 0)
	//	bound += quad.Fixed(fnFactory(xFree, y2), x1, x2, 2, quad.Legendre{}, 0)
	//	bound += quad.Fixed(fnFactory(yFree, x1), y1, y2, 2, quad.Legendre{}, 0)
	//	bound += quad.Fixed(fnFactory(yFree, x2), y1, y2, 2, quad.Legendre{}, 0)
	//	return bound
}

//
func (e *Element2D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	panic("unimplemented")
	//	return e.integrateVol(k, wNode, uNode) + e.integrateBoundary(k, wNode, uNode)
}

//
func (e *Element2D) IntegrateForce(k Kernel, wNode int) float64 {
	panic("unimplemented")
	//	return e.integrateVol(k, wNode, -1) + e.integrateBoundary(k, wNode, -1)
}

func (e *Element2D) integrateVol(k Kernel, wNode, uNode int) float64 {
	panic("unimplemented")
	//	outer := func(ee float64) float64 {
	//		inner := func(nn float64) float64 {
	//			xs := make([]float64, 2)
	//			xs[0], xs[1] = e.trans.Transform(ee, nn)
	//			jacdet := e.trans.ReverseJacobianDet(ee, nn)
	//
	//			var w, u Node = e.Nodes[wNode], nil
	//			pars := &KernelParams{X: xs, W: w.Weight(xs), GradW: w.DerivWeight(xs)}
	//			if uNode < 0 {
	//				return jacdet * k.VolInt(pars)
	//			}
	//			u = e.Nodes[uNode]
	//			pars.U = u.Sample(xs)
	//			pars.GradU = u.DerivSample(xs)
	//			return jacdet * k.VolIntU(pars)
	//		}
	//		return quad.Fixed(inner, -1, 1, len(e.Nodes), quad.Legendre{}, 0)
	//	}
	//	return quad.Fixed(outer, -1, 1, len(e.Nodes), quad.Legendre{}, 0)
}
