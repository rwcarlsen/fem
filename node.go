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
func (n *Node) Weight(refx []float64) float64 { return n.ShapeFunc.Value(refx) * n.W }
func (n *Node) ValueDeriv(refx []float64) []float64 {
	d := n.ShapeFunc.Deriv(refx)
	for i := range d {
		d[i] *= n.U
	}
	return d
}
func (n *Node) WeightDeriv(refx []float64) []float64 {
	d := n.ShapeFunc.Deriv(refx)
	for i := range d {
		d[i] *= n.W
	}
	return d
}

type ShapeFunc interface {
	Value(refx []float64) float64
	Deriv(refx []float64) []float64
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

func (fn Lagrange1D) Deriv(refx []float64) []float64 {
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
	return []float64{dudx}
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

func (fn Lagrange2D) Deriv(refx []float64) []float64 {
	n := fn.Order + 1
	if fn.Index > n*n-1 {
		panic("incompatible Index and Order")
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

	return []float64{dudx, dudy}
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
}

func pow(a, b int) int {
	if b == 0 {
		return 1
	}
	return a * pow(a, b-1)
}

func (fn LagrangeND) Value(refx []float64) float64 {
	ndim := len(refx)
	n := fn.Order + 1
	if fn.Index > pow(n, ndim)-1 {
		panic("incompatible Index, Order, and dimension")
	}

	u := 1.0

	for d, xx := range refx {
		stride := pow(ndim, d)
		xindex := -1 + float64((fn.Index/stride)%n)*2/float64(fn.Order)
		for i := 0; i < n; i++ {
			if i != (fn.Index/stride)%n {
				x0 := -1 + 2*float64(i)/float64(fn.Order)
				u *= (xx - x0) / (xindex - x0)
			}
		}
	}

	return u
}

func (fn LagrangeND) Deriv(refx []float64) []float64 {
	ndim := len(refx)
	n := fn.Order + 1
	if fn.Index > pow(n, ndim)-1 {
		panic("incompatible Index, Order, and dimension")
	}

	strides := make([]int, ndim)
	for i := range strides {
		strides[i] = pow(n, i)
	}

	u := 1.0
	deriv := make([]float64, ndim)
	upart := make([]float64, ndim)
	for i := 0; i < n; i++ {
		x0 := -1 + 2*float64(i)/float64(fn.Order)
		for d, xx := range refx {
			stride := strides[d]
			xindex := -1 + 2*float64((fn.Index/stride)%n)/float64(fn.Order)

			if i != (fn.Index/stride)%n {
				upart[d] = (xx - x0) / (xindex - x0)
				deriv[d] = 1/(xindex-x0)*u + (xx-x0)/(xindex-x0)*deriv[d]
			} else {
				upart[d] = 1.0
			}
		}

		for d := range deriv {
			u *= upart[d]
			for i, ux := range upart {
				if d != i {
					deriv[d] *= ux
				}
			}
		}
	}

	return deriv
}
