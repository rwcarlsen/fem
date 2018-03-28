package main

type Node struct {
	// X is the position/coordinates for the node.
	X []float64
	// U is the solution at the node.
	U float64
	// W is the weight function value at the node.
	W float64
	// ShapeFunc is the shape function for the node
	ShapeFunc ShapeFunc
}

func (n *Node) Set(u, w float64) {
	n.U = u
	n.W = w
}

func (n *Node) Value(refx []float64, id CoordId) float64 {
	return n.ShapeFunc.Value(refx, id) * n.U
}
func (n *Node) Weight(refx []float64, id CoordId) float64 {
	return n.ShapeFunc.Value(refx, id) * n.W
}

// ValueDeriv returns the partial derivatives (i.e. gradient) contribution to the solution of the
// node at the given reference coordinates.  If deriv is not nil the result is stored in it and
// deriv is returned - otherwise a new slice is allocated.
func (n *Node) ValueDeriv(refx, deriv []float64, id CoordId) []float64 {
	d := n.ShapeFunc.Deriv(refx, deriv, id)
	for i := range d {
		d[i] *= n.U
	}
	return d
}

// WeightDeriv returns the partial derivatives (i.e. gradient) contribution to the weight function
// of the node at the given reference coordinates.  If deriv is not nil the result is stored in it
// and deriv is returned - otherwise a new slice is allocated.
func (n *Node) WeightDeriv(refx, deriv []float64, id CoordId) []float64 {
	d := n.ShapeFunc.Deriv(refx, deriv, id)
	for i := range d {
		d[i] *= n.W
	}
	return d
}

type ShapeFunc interface {
	Value(refx []float64, id CoordId) float64

	// Deriv calculates and returns the partial derivatives at the given reference coordinates for
	// each dimension.  If deriv is not nil, the resuts are stored in it and deriv is returned -
	// otherwise a new slice is allocated and returned.
	Deriv(refx, deriv []float64, id CoordId) []float64
}

type LagrangeNDCache map[struct{ order, index int }]*LagrangeND

func (c LagrangeNDCache) New(order, index int) *LagrangeND {
	key := struct{ order, index int }{order, index}
	if _, ok := c[key]; !ok {
		c[key] = &LagrangeND{Index: index, Order: order}
	}
	return c[key]
}

type LagrangeND struct {

	// Index indicates for which of the nodes the shape function takes on the value 1.0.  For
	// Index=0, x=-1 and y=-1.  Subsequent (increasing) Index numbers indicate the nodes running
	// from left to right (increasing x) in dimensions 1 to ndim. Index runs from zero to
	// (Order+1)^ndim-1 inclusive.  Boundary nodes can be identified by the following createrion:
	//
	//    Index/(Order+1)^n + Index%(Order+1)^n == 0 or Order
	Index int
	Order int
	Safe  bool
	// xindices stores pre-computed xindex values in the u *= (xx-x0)/(xindex-x0) shape function terms
	xindices []float64
	// currpos caches (Index/stride)%(Order+1) where stride is (Order+1)^dim.
	currpos []int
	// upart caches memory used for calculating (partial) derivaties
	upart []float64
	// ordermults[i] caches x0=(-1+float64(i)*2/Order) where i indexes the
	// node-position related to the order (from zero to Order-1 inclusive).
	ordermults []float64

	valCache       []float64
	valCacheHave   []bool
	derivCache     [][]float64
	derivCacheHave []bool
}

func (fn *LagrangeND) init(ndim int) {
	if len(fn.xindices) == ndim {
		return
	}

	n := fn.Order + 1
	if fn.Index > pow(n, ndim)-1 {
		panic("incompatible Index, Order, and dimension")
	}
	fn.xindices = make([]float64, ndim)
	fn.currpos = make([]int, ndim)
	fn.upart = make([]float64, ndim)
	stride := 1
	for i := range fn.xindices {
		currpos := (fn.Index / stride) % n
		fn.currpos[i] = currpos
		fn.xindices[i] = -1 + 2*float64(currpos)/float64(fn.Order)
		stride *= n
	}
	fn.ordermults = make([]float64, n)
	for i := range fn.ordermults {
		fn.ordermults[i] = -1 + 2*float64(i)/float64(fn.Order)
	}

	nquadpoints := pow(fn.Order, ndim) + 2*ndim*pow(fn.Order, ndim-1) // rough upper bound estimate
	fn.derivCacheHave = make([]bool, nquadpoints)
	fn.derivCache = make([][]float64, nquadpoints)
	fn.valCache = make([]float64, nquadpoints)
	fn.valCacheHave = make([]bool, nquadpoints)
	for i := range fn.derivCache {
		fn.derivCache[i] = make([]float64, ndim)
	}
}

func (fn *LagrangeND) Value(refx []float64, id CoordId) float64 {
	ndim := len(refx)
	n := fn.Order + 1
	fn.init(ndim)

	if id >= 0 && len(fn.valCache) > int(id) && fn.valCacheHave[id] {
		return fn.valCache[id]
	}

	u := 1.0
	for d, xx := range refx {
		xindex := fn.xindices[d]
		currpos := fn.currpos[d]
		for i := 0; i < n; i++ {
			if i != currpos {
				x0 := fn.ordermults[i]
				u *= (xx - x0) / (xindex - x0)
			}
		}
	}

	if id >= 0 {
		for len(fn.valCache) <= int(id) {
			fn.valCache = append(fn.valCache, 0)
			fn.valCacheHave = append(fn.valCacheHave, false)
		}
		fn.valCacheHave[id] = true
		fn.valCache[id] = u
	}
	return u
}

func (fn LagrangeND) Deriv(refx, deriv []float64, id CoordId) []float64 {
	ndim := len(refx)
	n := fn.Order + 1
	fn.init(ndim)

	if deriv == nil {
		deriv = make([]float64, ndim)
	} else {
		for i := range deriv {
			deriv[i] = 0
		}
	}

	if id >= 0 && len(fn.derivCache) > int(id) && fn.derivCacheHave[id] {
		cache := fn.derivCache[id]
		for i := range deriv {
			deriv[i] = cache[i]
		}
		return deriv
	}

	u := 1.0
	for i := 0; i < n; i++ {
		x0 := fn.ordermults[i]
		for d, xx := range refx {
			if i != fn.currpos[d] {
				xindex := fn.xindices[d]
				a := 1 / (xindex - x0)
				term := xx - x0
				fn.upart[d] = a * term
				// compute 1/(xindex-x0)*u + (xx-x0)/(xindex-x0)*deriv[d]:
				deriv[d] = a * (u + term*deriv[d])
			} else {
				fn.upart[d] = 1.0
			}
		}

		for d := range deriv {
			u *= fn.upart[d]
			for i, ux := range fn.upart {
				if d != i {
					deriv[d] *= ux
				}
			}
		}
	}

	if id >= 0 {
		for len(fn.derivCache) <= int(id) {
			fn.derivCache = append(fn.derivCache, make([]float64, ndim))
			fn.derivCacheHave = append(fn.derivCacheHave, false)
		}
		fn.derivCacheHave[id] = true
		for i := range deriv {
			cache := fn.derivCache[id]
			cache[i] = deriv[i]
		}
	}
	return deriv
}
