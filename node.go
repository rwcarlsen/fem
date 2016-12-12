package main

import "math"

// Node represents a node and its interpolant within a finite element.
type Node interface {
	// X returns the global/absolute position of the node.
	X() []float64
	// Sample returns the value of the node's shape/solution function at x.
	Sample(x []float64) float64
	// Weight returns the value of the node'd weight/test function at x.
	Weight(x []float64) float64
	// DerivSample returns the partial derivative for the dim'th dimension
	// of the node's shape function at x.
	DerivSample(x []float64, dim int) float64
	// DerivWeight returns the partial derivative for the dim'th dimension of
	// the node's weight function at x.
	DerivWeight(x []float64, dim int) float64
	// Set normalizes the node's shape/solution and weight/test function to be
	// equal to sample and weight at the node's position X().
	Set(sample, weight float64)
}

// LagrangeNode1D represents a finite element node.  It holds a polynomial shape
// function that can be sampled.  It represents a shape function of the
// following form:
//
//    (x - x2)    (x - x3)    (x - x4)
//    --------- * --------- * ---------  * ...
//    (x1 - x2)   (x1 - x3)   (x1 - x4)
//
type LagrangeNode1D struct {
	// Index identifies the interpolation point in Xvals where the node's
	// shape function is equal to U or W for the solution and weight
	// respectively.
	Index int
	// Xvals represent all the interpolation points where the node's
	// polynomial shape function is fixed/specified (i.e. either zero or U/W).
	// When constructing elements, all the points in Xvals must each have a
	// node that the Index is set to.
	Xvals []float64
	// U is the value of the node's solution shape function at Xvals[Index].
	U float64
	// W is the value of the node's weight shape function at Xvals[Index].
	W float64
}

// NewLagrangeNode returns a lagrange interpolating polynomial shape function
// backed node with U and W set to 1.0.  xs represents the interpolation
// points and xIndex identifies the point in xs where the polynomial is equal
// to U/W instead of zero.
func NewLagrangeNode(xIndex int, xs []float64) *LagrangeNode1D {
	return &LagrangeNode1D{
		Index: xIndex,
		Xvals: append([]float64{}, xs...),
		U:     1,
		W:     1,
	}
}

func (n *LagrangeNode1D) X() []float64 { return []float64{n.Xvals[n.Index]} }

func (n *LagrangeNode1D) Set(sample, weight float64) { n.U, n.W = sample, weight }

func (n *LagrangeNode1D) Sample(x []float64) float64 {
	xx, u := x[0], n.U
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return u
}

func (n *LagrangeNode1D) Weight(x []float64) float64 { return n.Sample(x) / n.U * n.W }

func (n *LagrangeNode1D) DerivSample(x []float64, dim int) float64 {
	xx, u := x[0], n.U
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()[0]-x0)*u + (xx-x0)/(n.X()[0]-x0)*dudx
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return dudx
}

func (n *LagrangeNode1D) DerivWeight(x []float64, dim int) float64 {
	xx, u := x[0], n.U
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()[0]-x0)*u + (xx-x0)/(n.X()[0]-x0)*dudx
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return dudx
}

type BilinearNode2D struct {
	X1, Y1 float64
	X2, Y2 float64
	U, W   float64
}

func (n *BilinearNode2D) X() []float64 { return []float64{n.X1, n.Y1} }

func (n *BilinearNode2D) Set(sample, weight float64) { n.U, n.W = sample, weight }

func (n *BilinearNode2D) Sample(x []float64) float64 {
	xx, yy, u := x[0], x[1], n.U
	u *= (xx - n.X2) / math.Abs(n.X1-n.X2)
	u *= (yy - n.Y2) / math.Abs(n.Y1-n.Y2)
	return u
}

func (n *BilinearNode2D) Weight(x []float64) float64 { return n.Sample(x) / n.U * n.W }

func (n *BilinearNode2D) DerivSample(x []float64, dim int) float64 {
	if dim == 0 {
		return n.U / (n.X1 - n.X2)
	} else {
		return n.U / (n.Y1 - n.Y2)
	}
}

func (n *BilinearNode2D) DerivWeight(x []float64, dim int) float64 {
	if dim == 0 {
		return n.W / (n.X1 - n.X2)
	} else {
		return n.W / (n.Y1 - n.Y2)
	}
}
