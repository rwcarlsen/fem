package main

import "errors"

// Node represents a node and its interpolant within a finite element.
type Node interface {
	// X returns the global/absolute position of the node.
	X() []float64
	// Sample returns the value of the node's shape/solution function at x.
	Sample(x []float64) float64
	// Weight returns the value of the node'd weight/test function at x.
	Weight(x []float64) float64
	// DerivSample returns the partial derivative for each dimension
	// of the node's shape function at x.
	DerivSample(x []float64) []float64
	// DerivWeight returns the partial derivative for each dimension of
	// the node's weight function at x.
	DerivWeight(x []float64) []float64
	// Set normalizes the node's shape/solution and weight/test function to be
	// equal to sample and weight at the node's position X().
	Set(sample, weight float64)
}

// LagrangeNode represents a finite element node.  It holds a polynomial shape
// function that can be sampled.  It represents a shape function of the
// following form:
//
//    (x - x2)    (x - x3)    (x - x4)
//    --------- * --------- * ---------  * ...
//    (x1 - x2)   (x1 - x3)   (x1 - x4)
//
type LagrangeNode struct {
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
func NewLagrangeNode(xIndex int, xs []float64) *LagrangeNode {
	return &LagrangeNode{
		Index: xIndex,
		Xvals: append([]float64{}, xs...),
		U:     1,
		W:     1,
	}
}

func (n *LagrangeNode) X() []float64 { return []float64{n.Xvals[n.Index]} }

func (n *LagrangeNode) Set(sample, weight float64) { n.U, n.W = sample, weight }

func (n *LagrangeNode) Sample(x []float64) float64 {
	xx, u := x[0], n.U
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return u
}

func (n *LagrangeNode) Weight(x []float64) float64 { return n.Sample(x) / n.U * n.W }

func (n *LagrangeNode) DerivSample(x []float64) []float64 {
	xx, u := x[0], n.U
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()[0]-x0)*u + (xx-x0)/(n.X()[0]-x0)*dudx
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return []float64{dudx}
}

func (n *LagrangeNode) DerivWeight(x []float64) []float64 {
	xx, u := x[0], n.U
	dudx := 0.0
	for i, x0 := range n.Xvals {
		if i == n.Index {
			continue
		}
		dudx = 1/(n.X()[0]-x0)*u + (xx-x0)/(n.X()[0]-x0)*dudx
		u *= (xx - x0) / (n.X()[0] - x0)
	}
	return []float64{dudx}
}

type BilinQuadNode struct {
	X1, Y1    float64
	X2, Y2    float64
	X3, Y3    float64
	X4, Y4    float64
	U, W      float64
	Transform *ProjectiveTransform
}

// NewBilinQuadNode creates a bilinearly interpolated node for the points
// x1,y2; x2,y2; x3,y3; x4,y4.  The point x1,y1 is the "primary" point and
// node location (i.e. the only one of the four points where the shape
// function is non-zero).  The points 1 thorugh 4 must be in counter-clockwise
// order.
func NewBilinQuadNode(x1, y1, x2, y2, x3, y3, x4, y4 float64) (*BilinQuadNode, error) {
	trans, err := NewProjectiveTransform(
		[][]float64{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}},
		[][]float64{{x1, y1}, {x2, y2}, {x3, y3}, {x4, y4}},
	)
	if err != nil {
		return nil, errors.New("failed to compute coordinate transform")
	}

	return &BilinQuadNode{
		X1: x1, X2: x2, X3: x3, X4: x4,
		Y1: y1, Y2: y2, Y3: y3, Y4: y4,
		U: 1, W: 1,
		Transform: trans,
	}, nil
}

func (n *BilinQuadNode) X() []float64               { return []float64{n.X1, n.Y1} }
func (n *BilinQuadNode) Set(sample, weight float64) { n.U, n.W = sample, weight }

func (n *BilinQuadNode) Sample(xv []float64) float64 {
	x, y := xv[0], xv[1]
	xx, yy := n.Transform.Reverse(x, y)

	u := n.U
	u *= (xx - 1) / (-1 - 1)
	u *= (yy - 1) / (-1 - 1)
	return u
}

func (n *BilinQuadNode) Weight(x []float64) float64 { return n.Sample(x) / n.U * n.W }

func (n *BilinQuadNode) DerivSample(xv []float64) []float64 {
	x, y := xv[0], xv[1]
	e, eta := n.Transform.Reverse(x, y)
	jac := n.Transform.Jacobian(x, y)

	dude := 1 / 4 * (eta - 1)
	dxde, dyde := jac.At(0, 0), jac.At(0, 1)
	dxdn, dydn := jac.At(1, 0), jac.At(1, 1)
	dudx := n.U * (dude/dxde + dudn/dxdn)
	dudy := n.U * (dude/dyde + dudn/dydn)
	return []float64{dudx, dudy}
}

func (n *BilinQuadNode) DerivWeight(x []float64) []float64 {
	dwdx := n.DerivSample(x)
	dwdx[0] *= n.W / n.U
	dwdx[1] *= n.W / n.U
	return dwdx
}

type BilinRectNode struct {
	X1, Y1 float64
	X2, Y2 float64
	U, W   float64
}

// NewBilinRectNode creates a new 2D rectangular node with bilinear
// interpolation with U and W initialized to 1.0. The point x1,y1 is the
// "primary" point where the node takes on its value U.  The node evaluates to
// zero at all other corner points (x1,y2; x2,y1; x2,y2).
func NewBilinRectNode(x1, y1, x2, y2 float64) *BilinRectNode {
	return &BilinRectNode{x1, y1, x2, y2, 1.0, 1.0}
}

func (n *BilinRectNode) X() []float64               { return []float64{n.X1, n.Y1} }
func (n *BilinRectNode) Set(sample, weight float64) { n.U, n.W = sample, weight }

func (n *BilinRectNode) Sample(x []float64) float64 {
	xx, yy, u := x[0], x[1], n.U
	u *= (xx - n.X2) / (n.X1 - n.X2)
	u *= (yy - n.Y2) / (n.Y1 - n.Y2)
	return u
}

func (n *BilinRectNode) Weight(x []float64) float64 { return n.Sample(x) / n.U * n.W }

func (n *BilinRectNode) DerivSample(x []float64) []float64 {
	return []float64{n.U / (n.X1 - n.X2), n.U / (n.Y1 - n.Y2)}
}

func (n *BilinRectNode) DerivWeight(x []float64) []float64 {
	return []float64{n.W / (n.X1 - n.X2), n.W / (n.Y1 - n.Y2)}
}
