package main

import (
	"fmt"
	"io"
	"math"

	"github.com/gonum/integrate/quad"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
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
	// Coord returns the actual coordinates in the element for the given reference coordinates
	// (between -1 and 1).
	Coord(refx []float64) []float64
}

// Converter represents functions that can generate/provide the (approximate) reference
// coordinates for a given real coordinate on element e.
type Converter func(e Element, x []float64) (refx []float64, err error)

// PermConverter generates an element converter function which returns the (approximated)
// reference coordinates of within an element for real coordinate position x.  An error is
// returned if x is not contained inside the element.  The returned converter divides each
// dimension into ndiv segments forming a multi-dimensional grid over the element's reference
// coordinate domain.  Each grid point will be checked and the grid point corresponding to a real
// coordinate closest to x will be returned.
func PermConverter(ndiv int) Converter {
	return func(e Element, x []float64) ([]float64, error) {
		if !e.Contains(x) {
			return nil, fmt.Errorf("cannot convert coordinates - element does not contain X=%v", x)
		}
		dims := make([]int, len(x))
		for i := range dims {
			dims[i] = ndiv
		}
		convert := func(perm []int) []float64 {
			x := make([]float64, len(perm))
			for i := range x {
				x[i] = -1 + 2*float64(perm[i])/(float64(ndiv)-1)
			}
			return x
		}

		perms := Permute(nil, dims...)
		best := make([]float64, len(x))
		bestnorm := math.Inf(1)
		for _, p := range perms {
			norm := vecL2Norm(vecSub(x, e.Coord(convert(p))))
			if norm < bestnorm {
				best = convert(p)
				bestnorm = norm
			}
		}
		return best, nil
	}
}

// OptimConverter performs a local optimization using vanilla algorithms (e.g. gradient descent,
// , newton, etc.) to find the reference coordinates for x.
func OptimConverter(e Element, x []float64) ([]float64, error) {
	p := optimize.Problem{
		Func: func(trial []float64) float64 {
			return vecL2Norm(vecSub(x, e.Coord(trial)))
		},
	}

	initial := make([]float64, len(x))
	result, err := optimize.Local(p, initial, nil, nil)
	if err != nil {
		return nil, err
	} else if err = result.Status.Err(); err != nil {
		return nil, err
	}
	return result.X, nil
}

// Interpolate returns the solution of the element at refx (reference coordinates [-1,1]).
func Interpolate(e Element, refx []float64) float64 {
	u := 0.0
	for _, n := range e.Nodes() {
		u += n.Value(refx)
	}
	return u
}

// InterpolateDeriv returns the partial derivatives of the element at refx (reference coordinates
// [-1,1]) for each dimension - i.e. the superposition of partial derivatives from each of the
// element nodes.
func InterpolateDeriv(e Element, refx []float64) []float64 {
	u := e.Nodes()[0].ValueDeriv(refx)
	for _, n := range e.Nodes()[1:] {
		subu := n.ValueDeriv(refx)
		for i := range subu {
			u[i] += subu[i]
		}
	}
	return u
}

// Element1D represents a 1D finite element.  It assumes len(x) == 1 (i.e.
// only one dimension of independent variables.
type Element1D struct {
	Nds               []*Node
	lowbound, upbound []float64
	// jacdet holds the determinant of jacobian to used convert from ref
	// element integral to real coord integral
	jacdet    float64
	invjacdet float64
	// all below fields are used as state-holders for quadrature integration func
	wNode, uNode int
	kern         Kernel
	refxs        []float64
	pars         *KernelParams
	pars2        *KernelParams
}

// NewElement1D generates a lagrange polynomial interpolating element of
// degree len(xs)-1 using the values in xs as the interpolation points/nodes.
func NewElement1D(xs []float64) *Element1D {
	e := &Element1D{refxs: make([]float64, 1), pars: &KernelParams{}, pars2: &KernelParams{}}
	for i := range xs {
		order := len(xs) - 1
		nodepos := []float64{xs[i]}
		n := &Node{X: nodepos, U: 1.0, W: 1.0, ShapeFunc: LagrangeND{Index: i, Order: order}}
		e.Nds = append(e.Nds, n)
	}

	e.jacdet = (e.right() - e.left()) / 2
	e.invjacdet = 1 / e.jacdet

	return e
}

func (e *Element1D) Bounds() (low, up []float64) {
	if e.lowbound == nil {
		e.lowbound = []float64{e.left()}
		e.upbound = []float64{e.right()}
	}
	return e.lowbound, e.upbound
}

func (e *Element1D) Nodes() []*Node { return e.Nds }

func (e *Element1D) Contains(x []float64) bool {
	xx := x[0]
	return e.left() <= xx && xx <= e.right()
}

func (e *Element1D) Coord(refx []float64) []float64 {
	return []float64{(e.left()*(1-refx[0]) + e.right()*(1+refx[0])) / 2}
}

func (e *Element1D) left() float64  { return e.Nds[0].X[0] }
func (e *Element1D) right() float64 { return e.Nds[len(e.Nds)-1].X[0] }

func (e *Element1D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	return e.integrateVol(k, wNode, uNode) + e.integrateBoundary(k, wNode, uNode)
}

func (e *Element1D) IntegrateForce(k Kernel, wNode int) float64 {
	return e.integrateVol(k, wNode, -1) + e.integrateBoundary(k, wNode, -1)
}

var refLeft = []float64{-1}
var refRight = []float64{1}

func (e *Element1D) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	var w, u *Node = e.Nds[wNode], nil
	if e.pars2.X == nil {
		e.pars.X = make([]float64, 1)
		e.pars2.X = make([]float64, 1)
	}

	e.pars.X[0] = e.left()
	e.pars.W = w.Weight(refLeft)
	e.pars.GradW = vecMult(w.WeightDeriv(refLeft), e.invjacdet)
	e.pars2.X[0] = e.right()
	e.pars2.W = w.Weight(refRight)
	e.pars2.GradW = vecMult(w.WeightDeriv(refRight), e.invjacdet)

	if uNode < 0 {
		return k.BoundaryInt(e.pars) + k.BoundaryInt(e.pars2)
	}
	u = e.Nds[uNode]
	e.pars.U = u.Value(refLeft)
	e.pars.GradU = vecMult(u.ValueDeriv(refLeft), e.invjacdet)

	e.pars2.U = u.Value(refRight)
	e.pars2.GradU = vecMult(u.ValueDeriv(refRight), e.invjacdet)
	return k.BoundaryIntU(e.pars) + k.BoundaryIntU(e.pars2)
}

func (e *Element1D) volQuadFunc(ref float64) float64 {
	var w, u *Node = e.Nds[e.wNode], nil
	e.refxs[0] = ref
	e.pars.X = e.Coord(e.refxs)
	e.pars.W = w.Weight(e.refxs)
	e.pars.GradW = vecMult(w.WeightDeriv(e.refxs), e.invjacdet)

	if e.uNode < 0 {
		return e.kern.VolInt(e.pars)
	}
	u = e.Nds[e.uNode]
	e.pars.U = u.Value(e.refxs)
	e.pars.GradU = vecMult(u.ValueDeriv(e.refxs), e.invjacdet)
	return e.kern.VolIntU(e.pars)
}

func (e *Element1D) integrateVol(k Kernel, wNode, uNode int) float64 {
	e.wNode, e.uNode = wNode, uNode
	e.kern = k
	return quad.Fixed(e.volQuadFunc, -1, 1, len(e.Nds), quad.Legendre{}, 0) * e.jacdet
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
		x := e.Coord(refx)[0]

		v := Interpolate(e, refx)
		d := InterpolateDeriv(e, refx)
		fmt.Fprintf(w, "%v\t%v\t%v\n", x, v, d[0])
	}
}

// PrintShapeFuncs prints the shape functions and their derivatives in
// tab-separated form with nsamples evenly spaced over the element's domain
// (one sample per line) in the form:
//
//    [x]	[LagrangeNode1-shape(x)]	[LagrangeNode1-shapederiv(x)]	[LagrangeNode2-shape(x)]   ...
func (e *Element1D) PrintShapeFuncs(w io.Writer, nsamples int) {
	drefx := 2 / (float64(nsamples) - 1)
	for i := 0; i < nsamples; i++ {
		refx := []float64{-1 + float64(i)*drefx}
		x := e.Coord(refx)[0]
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
	Nds   []*Node
	Order int
}

// NewElement2D creates a new 2D bilinear quad element.
// (x1[0],x1[1]);(x2[0],x2[1]);... must specify coordinates for the
// nodes running left to right (increasing x) in rows starting at the bottom and iterating towards
// the top (increasing y).
func NewElement2D(order int, points ...[]float64) *Element2D {
	nodes := make([]*Node, len(points))
	for i, x := range points {
		nodes[i] = &Node{X: x, U: 1.0, W: 1.0, ShapeFunc: LagrangeND{Order: order, Index: i}}
	}
	return &Element2D{Nds: nodes, Order: order}
}

func (e *Element2D) Nodes() []*Node { return e.Nds }

func (e *Element2D) Contains(x []float64) bool {
	refxs, err := OptimConverter(e, x)
	if err != nil {
		panic(err)
	}

	const eps = 1e-9
	for _, refx := range refxs {
		if refx < -1-eps || 1+eps < refx {
			return false
		}
	}
	return true
}

func (e *Element2D) Bounds() (low, up []float64) {
	return []float64{e.xmin(), e.ymin()}, []float64{e.xmax(), e.ymax()}
}

func (e *Element2D) xmin() float64 { return e.min(0, true) }
func (e *Element2D) xmax() float64 { return e.min(0, false) }
func (e *Element2D) ymin() float64 { return e.min(1, true) }
func (e *Element2D) ymax() float64 { return e.min(1, false) }

// TODO: handle cases of higher order elements where this doesn't account for the fact that the
// curved element edge could extend beyond the extreme node values.
func (e *Element2D) min(coord int, less bool) float64 {
	extreme := e.Nds[0].X[coord]
	for _, n := range e.Nds[1:] {
		if n.X[coord] < extreme && less || n.X[coord] > extreme && !less {
			extreme = n.X[coord]
		}
	}
	return extreme
}

func (e *Element2D) Coord(refx []float64) []float64 {
	const ndim = 2
	realcoords := make([]float64, ndim)
	for i := 0; i < ndim; i++ {
		tot := 0.0
		for _, n := range e.Nds {
			tot += n.ShapeFunc.Value(refx) * n.X[i]
		}
		realcoords[i] = tot
	}
	return realcoords
}

func (e *Element2D) IntegrateStiffness(k Kernel, wNode, uNode int) float64 {
	return e.integrateVol(k, wNode, uNode) + e.integrateBoundary(k, wNode, uNode)
}

func (e *Element2D) IntegrateForce(k Kernel, wNode int) float64 {
	return e.integrateVol(k, wNode, -1) + e.integrateBoundary(k, wNode, -1)
}

func (e *Element2D) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	const ndim = 2
	type minmaxvar bool
	const minvar minmaxvar = true
	const maxvar minmaxvar = false

	var w, u *Node = e.Nds[wNode], nil

	fnFactory := func(fixedDim int, minmax minmaxvar) func(x float64) float64 {
		varDim := 0
		if fixedDim == 0 {
			varDim = 1
		}

		refxs := []float64{1, 1}
		if minmax == minvar {
			refxs[fixedDim] = -1
		}

		return func(ref float64) float64 {
			refxs[varDim] = ref
			xs := e.Coord(refxs)

			jac := e.jacobian(refxs)
			jacdet := 0.0
			for i := 0; i < ndim; i++ {
				v := jac.At(varDim, i)
				jacdet += v * v
			}
			jacdet = math.Sqrt(jacdet)

			gradw := w.WeightDeriv(refxs)
			e.convertderiv(gradw, refxs)
			pars := &KernelParams{X: xs, W: w.Weight(refxs), GradW: gradw}
			if uNode < 0 {
				return k.BoundaryInt(pars) * jacdet
			}
			u = e.Nds[uNode]
			pars.U = u.Value(refxs)
			pars.GradU = u.ValueDeriv(refxs)
			e.convertderiv(pars.GradU, refxs)
			return k.BoundaryIntU(pars) * jacdet
		}
	}

	bound := 0.0
	nquadpoints := int(math.Ceil((float64(e.Order) + 1) / 2))
	for d := 0; d < ndim; d++ {
		// integrate over the face/side corresponding to pinning the variable in each dimension
		// to its min and max values
		bound += quad.Fixed(fnFactory(d, minvar), -1, 1, nquadpoints, quad.Legendre{}, 0)
		bound += quad.Fixed(fnFactory(d, maxvar), -1, 1, nquadpoints, quad.Legendre{}, 0)
	}
	return bound
}

func (e *Element2D) integrateVol(k Kernel, wNode, uNode int) float64 {
	outer := func(refx float64) float64 {
		inner := func(refy float64) float64 {
			refxs := []float64{refx, refy}
			xs := e.Coord(refxs)

			// determinant of jacobian to convert from ref element integral to
			// real coord integral
			jacdet := e.jacdet(refxs)

			var w, u *Node = e.Nds[wNode], nil
			gradw := w.WeightDeriv(refxs)
			e.convertderiv(gradw, refxs)
			pars := &KernelParams{X: xs, W: w.Weight(refxs), GradW: gradw}
			if uNode < 0 {
				return jacdet * k.VolInt(pars)
			}
			u = e.Nds[uNode]
			pars.U = u.Value(refxs)
			pars.GradU = u.ValueDeriv(refxs)
			e.convertderiv(pars.GradU, refxs)
			return jacdet * k.VolIntU(pars)
		}
		return quad.Fixed(inner, -1, 1, 2, quad.Legendre{}, 0)
	}
	return quad.Fixed(outer, -1, 1, 2, quad.Legendre{}, 0)
}

// convertderiv converts the dN/de and dN/dn (derivatives w.r.t. the reference coordinates) to
// dN/dx and dN/dy (derivatives w.r.t. the real coordinates).  This is used to convert the GradU
// and GradW terms to be the correct values when building the stiffness matrix.
func (e *Element2D) convertderiv(refgradu []float64, refxs []float64) {
	jac := e.jacobian(refxs)

	var soln mat64.Vector
	soln.SolveVec(jac, mat64.NewVector(2, refgradu))
	refgradu[0] = soln.At(0, 0)
	refgradu[1] = soln.At(1, 0)
}

func (e *Element2D) jacobian(refxs []float64) *mat64.Dense {
	ndim := 2
	mat := mat64.NewDense(ndim, ndim, nil)
	for i := 0; i < ndim; i++ {
		for j := 0; j < ndim; j++ {
			tot := 0.0
			for _, n := range e.Nds {
				tot += n.ShapeFunc.Deriv(refxs)[i] * n.X[j]
			}
			mat.Set(i, j, tot)
		}
	}
	return mat
}

// jacdet computes the determinant of the element's 2D jacobian:
// J = | dx/de  dy/de |
//     | dx/dn  dy/dn |
func (e *Element2D) jacdet(refxs []float64) float64 {
	return mat64.Det(e.jacobian(refxs))
}
