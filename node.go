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

type Bilinear struct {
	// Index indicates which of the four corners the shape function takes on the value 1.0:
	//
	//    (x=-1, y=-1): Index=1
	//    (x= 1, y=-1): Index=1
	//    (x=-1, y= 1): Index=1
	//    (x= 1, y= 1): Index=1
	Index int
	// Polynomial order of the shape function.
	Order int
}

func (fn Bilinear) Value(refx []float64) float64 {
	panic("unimplemented")
}

func (fn Bilinear) Deriv(refx []float64) []float64 {
	panic("unimplemented")
}
