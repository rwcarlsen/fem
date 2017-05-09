package main

type Node struct {
	X         []float64
	Solution  float64
	Weight    float64
	ShapeFunc ShapeFunc
}

func (n *Node) Solution

type ShapeFunc interface {
	Value(n *Node, refx []float64) float64
}

type Lagrange1D struct {
	// Index identifies the interpolation point or (virtual) nodes where the shape function is
	// equal to 1.0.
	Index int
	// Polynomial order of the shape function.
	Order int
}

func (n *Lagrange1D) Value(refx []float64) float64 {
	xx, u := refx[0], 1.0
	for i := 0; i < n.Order; i++ {
		if i == n.Index {
			continue
		}
		x0 := -1 + float64(i)*2/float64(n.Order)
		u *= (xx - x0) / (-1 - x0)
	}
	return u
}

func (n *Lagrange1D) Deriv(refx []float64) []float64 {
	xx, u := refx[0], 1.0
	dudx := 0.0
	for i := 0; i < n.Order; i++ {
		if i == n.Index {
			continue
		}
		x0 := -1 + float64(i)*2/float64(n.Order)
		dudx = 1/(-1-x0)*u + (xx-x0)/(-1-x0)*dudx
		u *= (xx - x0) / (-1 - x0)
	}
	return []float64{dudx}
}

//
//import "github.com/gonum/integrate/quad"
//
//// Node represents a node and its interpolant within a finite element.
//type Node interface {
//	// X returns the global/absolute position of the node.
//	X() []float64
//	// Solution returns the value of node's basis function at x.
//	Solution(x []float64) float64
//	// DerivSolution returns the partial derivative for each dimension
//	// of the node's shape function at x.
//	DerivSolution(order int, x []float64) []float64
//	// Weight returns the value of the node'd weight/test function at x.
//	Weight(x []float64) float64
//	// DerivWeight returns the partial derivative for each dimension of
//	// the node's weight function at x.
//	DerivWeight(order int, x []float64) []float64
//	// Set normalizes the node's shape/solution and weight/test function to be
//	// equal to sample and weight at the node's position X().
//	Set(sample, weight float64)
//}
//
//type Element interface {
//	// Coords maps the reference coordinates (between -1 and 1) to the real coordinates of the
//	// element.  The ref and real slices have the same length.
//	Coords(ref []float64) (real []float64)
//}
//
//func IntegrateVol()
//
//type Lagrange1D struct {
//	X1, X2 float64
//}
//
//func (l *Lagrange1D) Coords(ref []float64) (real []float64) {
//	r := reference[0]
//	return (l.X1*(1-r) + l.X2*(1+r)) / 2
//}
//
//type Kernel interface {
//	Value(p *KernelParams) float64
//}
//
//type Integrator interface {
//	Integrate(e Element, k Kernel) float64
//}
//
//type GausQuad1D struct{}
//
//func (gi *GausQuad1D) Integrate(e Element, k Kernel) float64 {
//	fn := func(ref float64) float64 {
//		x := e.Coords(ref)
//		xs := []float64{x}
//		var w, u Node = e.nodes[wNode], nil
//		pars := &KernelParams{X: xs, W: w.Weight(xs), GradW: w.DerivWeight(xs)}
//		if uNode < 0 {
//			return k.VolInt(pars)
//		}
//		u = e.nodes[uNode]
//		pars.U = u.Sample(xs)
//		pars.GradU = u.DerivSample(xs)
//		return k.VolIntU(pars)
//	}
//	return quad.Fixed(fn, e.left(), e.right(), len(e.nodes), quad.Legendre{}, 0)
//}
//
//// HeatConduction implements 1D heat conduction physics.
//// TODO: update the Boundary... methods to handle multi-dimensions
//type HeatConduction struct {
//	// X is the node/mesh points.
//	X []float64
//	// K is thermal conductivity (W/m/K).
//	K Valer
//	// S is heat source strength (W/m^3).
//	S Valer
//	// Boundary holds the non-dirichlet boundary conditions for the problem.
//	Boundary Boundary
//}
//
//func (hc *HeatConduction) IsDirichlet(x []float64) (bool, float64) {
//	return hc.Boundary.Type(x) == Dirichlet, hc.Boundary.Val(x)
//}
//
//func (hc *HeatConduction) VolIntU(p *KernelParams) float64 {
//	return Dot(p.GradW, p.GradU) * hc.K.Val(p.X)
//}
//
//func (hc *HeatConduction) VolInt(p *KernelParams) float64 {
//	return p.W * hc.S.Val(p.X)
//}
//
//func (hc *HeatConduction) BoundaryIntU(p *KernelParams) float64 { return 0 }
//
//func (hc *HeatConduction) BoundaryInt(p *KernelParams) float64 {
//	return p.W * hc.Boundary.Val(p.X)
//}
