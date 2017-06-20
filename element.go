package main

import (
	"fmt"
	"io"
	"math"

	"github.com/gonum/integrate/quad"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
)

type IntegralLocationId int

// Element represents an element and provides integration and bounds related
// functionality required for approximating differential equation solutions.
type Element interface {
	// Nodes returns a persistent list of nodes that comprise this
	// element in no particular order but in stable/consistent order.
	Nodes() []*Node
	// IntegrateStiffness returns the result of the integration terms of the
	// weak form of the differential equation that include/depend on u(x) (the
	// solution or dependent variable).
	IntegrateStiffness(k Kernel, wNode, uNode int, skipBoundary bool) float64
	// IntegrateForce returns the result of the integration terms of the weak
	// form of the differential equation that do *not* include/depend on u(x).
	IntegrateForce(k Kernel, wNode int, skipBoundary bool) float64
	// Bounds returns a hyper-cubic bounding box defined by low and up values
	// in each dimension.
	Bounds() (low, up []float64)
	// Contains returns true if x is inside this element and false otherwise.
	Contains(x []float64) bool
	// Coord returns the actual coordinates in the element for the given
	// reference coordinates (between -1 and 1).  If x is not nil, it stores
	// the real coordinates there and returns x. If desired/available, a unique identifier
	// corresponding to the given reference coordinates can be passed in ids (the first entry) to
	// help speed up calculations.
	Coord(x, refx []float64, id IntegralLocationId) []float64
}

// Converter represents functions that can generate/provide the (approximate)
// reference coordinates for a given real coordinate on element e.
type Converter func(e Element, x []float64) (refx []float64, err error)

func StructuredConverter(e Element, x []float64) (refx []float64, err error) {
	low, up := e.Bounds()
	refx = make([]float64, len(x))
	for d := range low {
		refx[d] = -1 + 2*(x[d]-low[d])/(up[d]-low[d])
	}
	return refx, nil
}

// PermConverter generates an element converter function which returns the
// (approximated) reference coordinates of within an element for real
// coordinate position x.  An error is returned if x is not contained inside
// the element.  The returned converter divides each dimension into ndiv
// segments forming a multi-dimensional grid over the element's reference
// coordinate domain.  Each grid point will be checked and the grid point
// corresponding to a real coordinate closest to x will be returned.
func PermConverter(ndiv int) Converter {
	return func(e Element, x []float64) ([]float64, error) {
		realcoords := make([]float64, len(x))
		diff := make([]float64, len(x))
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

		perms := Permute(nil, nil, dims...)
		best := make([]float64, len(x))
		bestnorm := math.Inf(1)
		for _, p := range perms {
			norm := vecL2Norm(vecSub(diff, x, e.Coord(realcoords, convert(p), -1)))
			if norm < bestnorm {
				best = convert(p)
				bestnorm = norm
			}
		}
		return best, nil
	}
}

// OptimConverter performs a local optimization using vanilla algorithms (e.g.
// gradient descent, , newton, etc.) to find the reference coordinates for x.
func OptimConverter(e Element, x []float64) ([]float64, error) {
	realcoords := make([]float64, len(x))
	diff := make([]float64, len(x))
	p := optimize.Problem{
		Func: func(trial []float64) float64 {
			return vecL2Norm(vecSub(diff, x, e.Coord(realcoords, trial, -1)))
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

// Interpolate returns the solution of the element at refx (reference
// coordinates [-1,1]).
func Interpolate(e Element, refx []float64) float64 {
	u := 0.0
	for _, n := range e.Nodes() {
		u += n.Value(refx, -1)
	}
	return u
}

// InterpolateDeriv returns the partial derivatives of the element at refx
// (reference coordinates [-1,1]) for each dimension - i.e. the superposition
// of partial derivatives from each of the element nodes.
func InterpolateDeriv(e Element, refx []float64) []float64 {
	u := e.Nodes()[0].ValueDeriv(refx, nil, -1)
	for _, n := range e.Nodes()[1:] {
		subu := n.ValueDeriv(refx, nil, -1)
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
		n := &Node{X: nodepos, U: 1.0, W: 1.0, ShapeFunc: &LagrangeND{Order: order, Index: i}}
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

func (e *Element1D) Coord(x, refx []float64, id IntegralLocationId) []float64 {
	if x == nil {
		x = make([]float64, 1)
	}
	x[0] = (e.left()*(1-refx[0]) + e.right()*(1+refx[0])) / 2
	return x
}

func (e *Element1D) left() float64  { return e.Nds[0].X[0] }
func (e *Element1D) right() float64 { return e.Nds[len(e.Nds)-1].X[0] }

func (e *Element1D) IntegrateStiffness(k Kernel, wNode, uNode int, skipBoundary bool) float64 {
	I := e.integrateVol(k, wNode, uNode)
	if !skipBoundary {
		I += e.integrateBoundary(k, wNode, uNode)
	}
	return I
}

func (e *Element1D) IntegrateForce(k Kernel, wNode int, skipBoundary bool) float64 {
	I := e.integrateVol(k, wNode, -1)
	if !skipBoundary {
		I += e.integrateBoundary(k, wNode, -1)
	}
	return I
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
	e.pars.W = w.Weight(refLeft, -1)
	e.pars.GradW = vecMult(w.WeightDeriv(refLeft, nil, -1), e.invjacdet)
	e.pars2.X[0] = e.right()
	e.pars2.W = w.Weight(refRight, -1)
	e.pars2.GradW = vecMult(w.WeightDeriv(refRight, nil, -1), e.invjacdet)

	if uNode < 0 {
		return k.BoundaryInt(e.pars) + k.BoundaryInt(e.pars2)
	}
	u = e.Nds[uNode]
	e.pars.U = u.Value(refLeft, -1)
	e.pars.GradU = vecMult(u.ValueDeriv(refLeft, nil, -1), e.invjacdet)

	e.pars2.U = u.Value(refRight, -1)
	e.pars2.GradU = vecMult(u.ValueDeriv(refRight, nil, -1), e.invjacdet)
	return k.BoundaryIntU(e.pars) + k.BoundaryIntU(e.pars2)
}

func (e *Element1D) volQuadFunc(ref float64) float64 {
	var w, u *Node = e.Nds[e.wNode], nil
	e.refxs[0] = ref
	e.pars.X = e.Coord(e.pars.X, e.refxs, -1)
	e.pars.W = w.Weight(e.refxs, -1)
	e.pars.GradW = vecMult(w.WeightDeriv(e.refxs, nil, -1), e.invjacdet)

	if e.uNode < 0 {
		return e.kern.VolInt(e.pars)
	}
	u = e.Nds[e.uNode]
	e.pars.U = u.Value(e.refxs, -1)
	e.pars.GradU = vecMult(u.ValueDeriv(e.refxs, nil, -1), e.invjacdet)
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
		x := e.Coord(nil, refx, -1)[0]

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
		x := e.Coord(nil, refx, -1)[0]
		fmt.Fprintf(w, "%v", x)
		for _, n := range e.Nds {
			if x < e.left() || x > e.right() {
				fmt.Fprintf(w, "\t0\t0")
			} else {
				fmt.Fprintf(w, "\t%v\t%v", n.Value(refx, -1), n.ValueDeriv(refx, nil, -1)[0])
			}
		}
		fmt.Fprintf(w, "\n")
	}
}

type ElementND struct {
	Nds   []*Node
	Order int
	NDim  int
	Conv  Converter
	Cache *ElementCache
	// ndim
	low, up []float64
}

// NewElementND creates a new N-dimensional lagrange brick element (i.e. line,
// quadrilateral, brick, hyperbrick).
// (x1[0],x1[1],x[2],...);(x2[0],x2[1],x2[2],...);(...);... must specify
// coordinates for the nodes running left to right (increasing x) in rows
// starting at the lowest dimension and iterating recursively towards the
// highest dimension.
func NewElementND(order int, cache LagrangeNDCache, points ...[]float64) *ElementND {
	ndim := len(points[0])
	nodes := make([]*Node, len(points))
	for i, x := range points {
		n := &LagrangeND{Order: order, Index: i}
		if cache != nil {
			n = cache.New(order, i)
		}
		nodes[i] = &Node{X: x, U: 1.0, W: 1.0, ShapeFunc: n}
	}
	tmpderivs := make([][]float64, len(nodes))
	for i := range tmpderivs {
		tmpderivs[i] = make([]float64, ndim)
	}
	return &ElementND{
		Nds:   nodes,
		Order: order,
		NDim:  ndim,
		Conv:  OptimConverter,
	}
}

func (e *ElementND) Nodes() []*Node { return e.Nds }

func (e *ElementND) Contains(x []float64) bool {
	refxs, err := e.Conv(e, x)
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

func (e *ElementND) Bounds() (low, up []float64) {
	e.initCache()
	if e.low == nil {
		for d := 0; d < e.NDim; d++ {
			e.low = append(e.low, e.extreme(d, true))
			e.up = append(e.up, e.extreme(d, false))
		}
	}
	return e.low, e.up
}

// TODO: handle cases of higher order elements where this doesn't account for
// the fact that the curved element edge could extend beyond the extreme node
// values.
func (e *ElementND) extreme(coord int, less bool) float64 {
	extreme := e.Nds[0].X[coord]
	for _, n := range e.Nds[1:] {
		if n.X[coord] < extreme && less || n.X[coord] > extreme && !less {
			extreme = n.X[coord]
		}
	}
	return extreme
}

func (e *ElementND) Coord(x, refx []float64, id IntegralLocationId) []float64 {
	if x == nil {
		x = make([]float64, e.NDim)
	} else {
		if id >= 0 && e.Cache.HaveCoord[id] {
			for i, v := range e.Cache.Coords[id] {
				x[i] = v
			}
			return x
		}
		for i := range x {
			x[i] = 0
		}
	}

	for _, n := range e.Nds {
		val := n.ShapeFunc.Value(refx, id)
		for i := 0; i < e.NDim; i++ {
			x[i] += val * n.X[i]
		}
	}

	if id >= 0 {
		e.Cache.HaveCoord[id] = true
		cache := e.Cache.Coords[id]
		for i, v := range x {
			cache[i] = v
		}
	}

	return x
}

func (e *ElementND) IntegrateStiffness(k Kernel, wNode, uNode int, skipBoundary bool) float64 {
	I := e.integrateVol(k, wNode, uNode)
	if !skipBoundary {
		I += e.integrateBoundary(k, wNode, uNode)
	}
	return I
}

func (e *ElementND) IntegrateForce(k Kernel, wNode int, skipBoundary bool) float64 {
	I := e.integrateVol(k, wNode, -1)
	if !skipBoundary {
		I += e.integrateBoundary(k, wNode, -1)
	}
	return I
}

func (e *ElementND) integrateBoundary(k Kernel, wNode, uNode int) float64 {
	locid := IntegralLocationId(e.nquadpoints())
	fi := &FaceIntegrator{
		Elem: e,
		W:    e.Nds[wNode],
		K:    k,
		Loc:  &locid,
	}

	if uNode >= 0 {
		fi.U = e.Nds[uNode]
	}

	bound := 0.0
	nquadpoints := e.nquadpointsdim()
	xs := make([]float64, nquadpoints)
	weights := make([]float64, nquadpoints)
	for d := 0; d < e.NDim; d++ {
		// integrate over the face/side corresponding to pinning the variable
		// in each dimension to its min and max values
		fi.FixedDim = d
		fi.FixedVal = -1
		bound += QuadLegendre(e.NDim-1, fi.Func, -1, 1, nquadpoints, xs, weights)
		fi.FixedVal = 1
		bound += QuadLegendre(e.NDim-1, fi.Func, -1, 1, nquadpoints, xs, weights)
	}
	return bound
}

func (e *ElementND) nquadpoints() int {
	return pow(e.nquadpointsdim(), e.NDim)
}

func (e *ElementND) nquadpointsall() int {
	return e.nquadpoints() + 2*e.NDim*pow(e.nquadpointsdim(), e.NDim-1)
}

func (e *ElementND) nquadpointsdim() int {
	return int(math.Ceil((float64(e.Order) + 1) / 2))
}

func (e *ElementND) integrateVol(k Kernel, wNode, uNode int) float64 {
	var locid IntegralLocationId

	fn := func(refxs []float64) float64 {
		e.Coord(e.Cache.Pars.X, refxs, locid)

		jac := e.jacobian(refxs, locid)
		// determinant of jacobian to convert from ref element integral to
		// real coord integral:
		//     J = | dx/de  dy/de |
		//         | dx/dn  dy/dn |
		jacdet := det(jac)

		var w, u *Node = e.Nds[wNode], nil
		e.Cache.Pars.W = w.Weight(refxs, locid)
		w.WeightDeriv(refxs, e.Cache.Pars.GradW, locid)
		ConvertDeriv(jac, e.Cache.Pars.GradW)
		if uNode < 0 {
			e.Cache.Pars.U = 0
			for i := range e.Cache.Pars.GradU {
				e.Cache.Pars.GradU[i] = 0
			}
			locid++
			return jacdet * k.VolInt(e.Cache.Pars)
		}
		u = e.Nds[uNode]
		e.Cache.Pars.U = u.Value(refxs, locid)
		u.ValueDeriv(refxs, e.Cache.Pars.GradU, locid)
		ConvertDeriv(jac, e.Cache.Pars.GradU)

		locid++
		return jacdet * k.VolIntU(e.Cache.Pars)
	}
	nquadpoints := e.nquadpointsdim()
	xs := make([]float64, nquadpoints)
	weights := make([]float64, nquadpoints)
	integral := QuadLegendre(e.NDim, fn, -1, 1, nquadpoints, xs, weights)
	return integral
}

func (e *ElementND) initCache() {
	if e.Cache == nil {
		e.Cache = NewElementCache()
	}
	e.Cache.Init(e.NDim, len(e.Nds), e.nquadpointsall())
}

func (e *ElementND) jacobian(refxs []float64, id IntegralLocationId) *mat64.Dense {
	e.initCache()

	if e.Cache.HaveJac[id] {
		return e.Cache.Jacs[id]
	}
	e.Cache.HaveJac[id] = true

	mat := e.Cache.Jacs[id]
	data := mat.RawMatrix().Data
	for i := range data {
		data[i] = 0
	}
	for ni, n := range e.Nds {
		deriv := e.Cache.Derivs[ni]
		n.ShapeFunc.Deriv(refxs, deriv, id)
		for i := 0; i < e.NDim; i++ {
			dd := deriv[i]
			index := e.NDim * i
			for j := 0; j < e.NDim; j++ {
				data[index+j] += dd * n.X[j]
			}
		}
	}
	return mat
}

type ElementCache struct {
	// npoints x ndim
	Xs [][]float64
	// npoints x ndim
	Derivs [][]float64
	// ndim x ndim
	Mat *mat64.Dense
	// ndim x ndim
	Jacs      []*mat64.Dense
	HaveJac   []bool
	Coords    [][]float64
	HaveCoord []bool
	// ndim
	RefXs []float64
	Pars  *KernelParams
}

func NewElementCache() *ElementCache { return &ElementCache{} }

func (c *ElementCache) Init(ndim int, npoints int, nquadpoints int) {
	if c.Xs != nil {
		return
	}
	c.Xs = make([][]float64, npoints)
	c.Derivs = make([][]float64, npoints)
	for i := range c.Xs {
		c.Xs[i] = make([]float64, ndim)
		c.Derivs[i] = make([]float64, ndim)
	}

	c.HaveCoord = make([]bool, nquadpoints)
	c.Coords = make([][]float64, nquadpoints)
	c.HaveJac = make([]bool, nquadpoints)
	c.Jacs = make([]*mat64.Dense, nquadpoints)
	for i := range c.Jacs {
		c.Jacs[i] = mat64.NewDense(ndim, ndim, nil)
		c.Coords[i] = make([]float64, ndim)
	}

	c.RefXs = make([]float64, ndim)
	c.Mat = mat64.NewDense(ndim, ndim, nil)
	c.Pars = &KernelParams{GradW: make([]float64, ndim), GradU: make([]float64, ndim)}
}

type FaceIntegrator struct {
	FixedDim int
	FixedVal float64
	Elem     *ElementND
	U, W     *Node
	K        Kernel
	Loc      *IntegralLocationId
}

func (fi *FaceIntegrator) Func(partialrefxs []float64) float64 {
	loc := *fi.Loc
	ndim := len(partialrefxs) + 1
	refxs := make([]float64, ndim)

	// rebuild full-rank coordinates
	refi := 0
	for i := range refxs {
		if i == fi.FixedDim {
			refxs[i] = fi.FixedVal
		} else {
			refxs[i] = partialrefxs[refi]
			refi++
		}
	}

	jac := fi.Elem.jacobian(refxs, loc)
	jacdet := faceArea(fi.FixedDim, jac)

	pars := &KernelParams{}
	pars.X = fi.Elem.Coord(pars.X, refxs, loc)
	pars.GradW = fi.W.WeightDeriv(refxs, pars.GradW, loc)
	ConvertDeriv(jac, pars.GradW)
	pars.W = fi.W.Weight(refxs, loc)

	if fi.U == nil {
		// TODO: when we start caching KernelParams object, we will need to
		// zero out U and GradU here
		*fi.Loc++
		return fi.K.BoundaryInt(pars) * jacdet
	}
	pars.U = fi.U.Value(refxs, loc)
	pars.GradU = fi.U.ValueDeriv(refxs, pars.GradU, loc)
	ConvertDeriv(jac, pars.GradU)
	*fi.Loc++
	return fi.K.BoundaryIntU(pars) * jacdet
}

// ConvertDeriv converts the dN/de and dN/dn (derivatives w.r.t. the reference
// coordinates) to dN/dx and dN/dy (derivatives w.r.t. the real coordinates).
// This is used to convert the GradU and GradW terms to be the correct values
// when building the stiffness matrix.
func ConvertDeriv(jac *mat64.Dense, refgradu []float64) {
	ndim, _ := jac.Dims()
	switch ndim {
	case 1:
		refgradu[0] /= jac.At(0, 0)
	case 2:
		// fastpath for 2 dimensions
		a := jac.At(0, 0)
		b := jac.At(0, 1)
		c := jac.At(1, 0)
		d := jac.At(1, 1)
		invdet := 1 / (a*d - b*c)
		s1 := invdet * (d*refgradu[0] - b*refgradu[1])
		s2 := invdet * (-c*refgradu[0] + a*refgradu[1])
		refgradu[0] = s1
		refgradu[1] = s2
	case 3:
		a := jac.At(0, 0)
		b := jac.At(0, 1)
		c := jac.At(0, 2)
		d := jac.At(1, 0)
		e := jac.At(1, 1)
		f := jac.At(1, 2)
		g := jac.At(2, 0)
		h := jac.At(2, 1)
		i := jac.At(2, 2)

		invdet := 1 / (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)

		s1 := invdet * ((e*i-f*h)*refgradu[0] + (c*h-b*i)*refgradu[1] + (b*f-c*e)*refgradu[2])
		s2 := invdet * ((f*g-d*i)*refgradu[0] + (a*i-c*g)*refgradu[1] + (c*d-a*f)*refgradu[2])
		s3 := invdet * ((d*h-e*g)*refgradu[0] + (b*g-a*h)*refgradu[1] + (a*e-b*d)*refgradu[2])
		refgradu[0] = s1
		refgradu[1] = s2
		refgradu[2] = s3
	default:
		var soln mat64.Vector
		soln.SolveVec(jac, mat64.NewVector(ndim, refgradu))
		for i := range refgradu {
			refgradu[i] = soln.At(i, 0)
		}
	}
}

// This is used to compute the ratio of differential surface area at a
// particular point for which the numerical jacobian jac is given.  jac is
// d(real or parent coords)/d(reference coords) for the desired
// point/location.  fixedDim identifies the dimension of the reference
// coordinates that is fixed - the differential area being computed by tangent
// vectors (i.e. built from the jacobian entries) in the other dimensions.
//
// This is a generalization of the surface area-ratio formula for parametric
// surfaces (https://en.wikipedia.org/wiki/Parametric_surface#Surface_area) to
// multiple dimensions - we take a higher dimensional cross-product of the
// tangent vectors (i.e. jacobian terms) to our differential area of the
// element face using by replacing the jacobian row corresponding to the fixed
// dimension with i_had, j_hat, etc. vectors.  The multi-dimensional cross
// product is defined using the Hodge-star operator
// (https://en.wikipedia.org/wiki/Hodge_dual).  See also
// https://math.stackexchange.com/questions/185991/is-the-vector-cross-product-only-defined-for-3d#186000.
func faceArea(fixedDim int, jac *mat64.Dense) float64 {
	ndim, _ := jac.Dims()
	tot := 0.0
	subjac := mat64.NewDense(ndim-1, ndim-1, nil)
	for direc := 0; direc < ndim; direc++ {
		ii := 0
		for i := 0; i < ndim; i++ {
			if i == fixedDim {
				continue
			}
			jj := 0
			for j := 0; j < ndim; j++ {
				if j == direc {
					continue
				}
				subjac.Set(ii, jj, jac.At(i, j))
				jj++
			}
			ii++
		}
		d := det(subjac)
		tot += d * d
	}
	return math.Sqrt(tot)
}
