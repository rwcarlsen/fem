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

func (n *Node) Value(refx []float64) float64 {
	return n.ShapeFunc.Value(refx) * n.U
}
func (n *Node) Weight(refx []float64) float64 {
	return n.ShapeFunc.Value(refx) * n.W
}

// ValueDeriv returns the partial derivatives (i.e. gradient) contribution to the solution of the
// node at the given reference coordinates.  If deriv is not nil the result is stored in it and
// deriv is returned - otherwise a new slice is allocated.
func (n *Node) ValueDeriv(refx, deriv []float64) []float64 {
	d := n.ShapeFunc.Deriv(refx, deriv)
	for i := range d {
		d[i] *= n.U
	}
	return d
}

// WeightDeriv returns the partial derivatives (i.e. gradient) contribution to the weight function
// of the node at the given reference coordinates.  If deriv is not nil the result is stored in it
// and deriv is returned - otherwise a new slice is allocated.
func (n *Node) WeightDeriv(refx, deriv []float64) []float64 {
	d := n.ShapeFunc.Deriv(refx, deriv)
	for i := range d {
		d[i] *= n.W
	}
	return d
}

type ShapeFunc interface {
	Value(refx []float64) float64
	// Deriv calculates and returns the partial derivatives at the given reference coordinates for
	// each dimension.  If deriv is not nil, the resuts are stored in it and deriv is returned -
	// otherwise a new slice is allocated and returned.
	Deriv(refx, deriv []float64) []float64
}

type Lagrange1D struct {
	// Index identifies the interpolation point or (virtual) nodes where the shape function is
	// equal to 1.0.
	Index int
	// Polynomial order of the shape function.
	Order int
}

func (fn Lagrange1D) Value(refx []float64) float64 {
	xx, u := refx[0], 1.0
	xindex := -1 + float64(fn.Index)*2/float64(fn.Order)
	for i := 0; i < fn.Order+1; i++ {
		if i == fn.Index {
			continue
		}
		x0 := -1 + float64(i)*2/float64(fn.Order)
		u *= (xx - x0) / (xindex - x0)
	}
	return u
}

func (fn Lagrange1D) Deriv(refx, deriv []float64) []float64 {
	if deriv == nil {
		deriv = make([]float64, 1)
	}

	xx, u := refx[0], 1.0
	dudx := 0.0
	xindex := -1 + float64(fn.Index)*2/float64(fn.Order)
	for i := 0; i < fn.Order+1; i++ {
		if i == fn.Index {
			continue
		}
		x0 := -1 + float64(i)*2/float64(fn.Order)
		dudx = 1/(xindex-x0)*u + (xx-x0)/(xindex-x0)*dudx
		u *= (xx - x0) / (xindex - x0)
	}
	deriv[0] = dudx
	return deriv
}

type Lagrange2D struct {
	// Index indicates for which of the nodes the shape function takes on the value 1.0.  For
	// Index=0, x=-1 and y=-1.  Subsequent (increasing) Index numbers indicate the nodes running
	// counter-clockwise from left to right (increasing x) in rows from bottom to top (increasing
	// y).  Index runs from zero to (Order+1)^2-1 inclusive.
	// Boundary nodes can be identified by the following createria:
	//
	//    * Bottom: Index/3==0
	//    * Top: Index/3==Order
	//    * Left: Index%3==O
	//    * Right: Index%3==Order
	Index int
	Order int
}

func (fn Lagrange2D) Value(refx []float64) float64 {
	n := fn.Order + 1
	if fn.Index > n*n-1 {
		panic("incompatible Index and Order")
	}

	xx, yy, u := refx[0], refx[1], 1.0

	xindex := -1 + float64(fn.Index%n)*2/float64(fn.Order)
	yindex := -1 + float64(fn.Index/n)*2/float64(fn.Order)
	for i := 0; i < n; i++ {
		if i != fn.Index%n {
			x0 := -1 + 2*float64(i)/float64(fn.Order)
			u *= (xx - x0) / (xindex - x0)
		}
		if i != fn.Index/n {
			y0 := -1 + 2*float64(i)/float64(fn.Order)
			u *= (yy - y0) / (yindex - y0)
		}
	}

	return u
}

func (fn Lagrange2D) Deriv(refx, deriv []float64) []float64 {
	n := fn.Order + 1
	if fn.Index > n*n-1 {
		panic("incompatible Index and Order")
	}

	if deriv == nil {
		deriv = make([]float64, 2)
	}

	xx, yy := refx[0], refx[1]
	u, dudx, dudy := 1.0, 0.0, 0.0

	xindex := -1 + float64(fn.Index%n)*2/float64(fn.Order)
	yindex := -1 + float64(fn.Index/n)*2/float64(fn.Order)
	for i := 0; i < n; i++ {
		x0 := -1 + 2*float64(i)/float64(fn.Order)
		y0 := x0
		ux, uy := 1.0, 1.0
		if i != fn.Index%n {
			ux = (xx - x0) / (xindex - x0)
		}
		if i != fn.Index/n {
			uy = (yy - y0) / (yindex - y0)
		}

		if i != fn.Index%n {
			dudx = 1/(xindex-x0)*u + (xx-x0)/(xindex-x0)*dudx
		}

		if i != fn.Index/n {
			dudy = 1/(yindex-y0)*u + (yy-y0)/(yindex-y0)*dudy
		}
		dudx *= uy
		dudy *= ux
		u *= ux * uy
	}

	deriv[0] = dudx
	deriv[1] = dudy
	return deriv
}

type LagrangeND struct {
	// Index indicates for which of the nodes the shape function takes on the value 1.0.  For
	// Index=0, x=-1 and y=-1.  Subsequent (increasing) Index numbers indicate the nodes running
	// counter-clockwise from left to right (increasing x) in rows from bottom to top (increasing
	// y).  Index runs from zero to (Order+1)^2-1 inclusive.
	// Boundary nodes can be identified by the following createria:
	//
	//    * Bottom: Index/3==0
	//    * Top: Index/3==Order
	//    * Left: Index%3==O
	//    * Right: Index%3==Order
	Index int
	Order int
	Safe  bool
	// xindices stores pre-computed xindex values in the u *= (xx-x0)/(xindex-x0) shape function terms
	xindices []float64
	// currpos caches (fn.Index/stride)%(fn.Order+1) where stride is (fn.Order+1)^dim.
	currpos []int
	// upart caches memory used for calculating (partial) derivaties
	upart []float64
}

var nodecache = map[struct{ order, index int }]*LagrangeND{}

func NewLagrangeND(order, index int) *LagrangeND {
	key := struct{ order, index int }{order, index}
	if _, ok := nodecache[key]; !ok {
		nodecache[key] = &LagrangeND{Index: index, Order: order}
	}
	return nodecache[key]
}

func (fn *LagrangeND) init(ndim int) {
	if len(fn.xindices) != ndim {
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
	}
}

func (fn *LagrangeND) Value(refx []float64) float64 {
	ndim := len(refx)
	n := fn.Order + 1
	fn.init(ndim)

	u := 1.0

	ordermult := 2 / float64(fn.Order)
	for d, xx := range refx {
		xindex := fn.xindices[d]
		currpos := fn.currpos[d]
		for i := 0; i < n; i++ {
			if i != currpos {
				x0 := -1 + float64(i)*ordermult
				u *= (xx - x0) / (xindex - x0)
			}
		}
	}

	return u
}

func (fn LagrangeND) Deriv(refx, deriv []float64) []float64 {
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

	u := 1.0
	for i := 0; i < n; i++ {
		x0 := -1 + 2*float64(i)/float64(fn.Order)
		for d, xx := range refx {
			xindex := fn.xindices[d]
			currpos := fn.currpos[d]

			if i != currpos {
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

	return deriv
}
