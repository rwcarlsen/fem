package main

type BoundaryType int

const (
	Dirichlet BoundaryType = iota
	Neumann
)

type Boundary struct {
	Type BoundaryType
	Val  float64
}

// Dot performs a vector*vector dot product.
func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("inconsistent lengths for dot product")
	}
	v := 0.0
	for i := range a {
		v += a[i] * b[i]
	}
	return v
}

func DirichletBC(u float64) *Boundary   { return &Boundary{Dirichlet, u} }
func NeumannBC(gradU float64) *Boundary { return &Boundary{Neumann, gradU} }

type KernelParams struct {
	// X is the position the kernel is being evaluated at.
	X []float64
	// U is value of the solution shape function if solving a linear system
	// and it is the current guess for the solution (i.e. shape function val
	// times current solution guess).  For linear systems, an explicit solve
	// will actually determine the nodal U values directly.  For nonlinear
	// systems, an Newton or similar will be used to iterate toward better U
	// guesses.
	U float64
	// GradU holds the (partial) derivatives of the solution shape function.
	GradU []float64
	// W holds the value of the weight/test function
	W float64
	// GradW holds the (partial) derivatives of the weight/test function
	GradW []float64
}

type Kernel interface {
	// Kernel returns the value of the volumetric integration portion of the weak form
	// (i.e. everything but the boundary/surface integration).
	VolIntU(p *KernelParams) float64
	VolInt(p *KernelParams) float64
	BoundaryIntU(p *KernelParams) float64
	BoundaryInt(p *KernelParams) float64
	IsDirichlet(xs []float64) (bool, float64)
}

type Valer interface {
	Val(x []float64) float64
}

type ConstVal float64

func (p ConstVal) Val(x []float64) float64 { return float64(p) }

// LinVals only works for 1D problems
type LinVals struct {
	X []float64
	Y []float64
}

func (p *LinVals) Val(x []float64) float64 {
	xx := x[0]
	for i := 0; i < len(p.X)-1; i++ {
		x1, x2 := p.X[i], p.X[i+1]
		y1, y2 := p.Y[i], p.Y[i+1]
		if x1 <= xx && xx <= x2 {
			return y2 + (xx-x1)/(x2-x1)*(y2-y1)
		}
	}
	if xx < p.X[0] {
		return p.Y[0]
	}
	return p.Y[len(p.Y)-1]
}

// SecVal only works for 1D problems.
type SecVal struct {
	X []float64
	Y []float64
}

func (p *SecVal) Val(x []float64) float64 {
	xx := x[0]
	for i := 0; i < len(p.X)-1; i++ {
		x1, x2 := p.X[i], p.X[i+1]
		y := p.Y[i]
		if x1 <= xx && xx <= x2 {
			return y
		}
	}
	if xx < p.X[0] {
		return p.Y[0]
	}
	return p.Y[len(p.Y)-1]
}

// HeatConduction implements 1D heat conduction physics.
// TODO: update the Boundary... methods to handle multi-dimensions
type HeatConduction struct {
	// X is the node/mesh points.
	X []float64
	// K is thermal conductivity (W/m/K).
	K Valer
	// S is heat source strength (W/m^3).
	S     Valer
	Left  *Boundary
	Right *Boundary
}

func (hc *HeatConduction) IsDirichlet(x []float64) (bool, float64) {
	if x[0] == hc.X[0] && hc.Left.Type == Dirichlet {
		return true, hc.Left.Val
	} else if x[0] == hc.X[len(hc.X)-1] && hc.Right.Type == Dirichlet {
		return true, hc.Right.Val
	}
	return false, 0
}

func (hc *HeatConduction) VolIntU(p *KernelParams) float64 {
	return Dot(p.GradW, p.GradU) * hc.K.Val(p.X)
}

func (hc *HeatConduction) VolInt(p *KernelParams) float64 {
	return p.W * hc.S.Val(p.X)
}

func (hc *HeatConduction) BoundaryIntU(p *KernelParams) float64 { return 0 }

func (hc *HeatConduction) BoundaryInt(p *KernelParams) float64 {
	if p.X[0] == hc.X[0] {
		return p.W * hc.Left.Val
	} else if p.X[0] == hc.X[len(hc.X)-1] {
		return -1 * p.W * hc.Right.Val
	}
	return 0
}
