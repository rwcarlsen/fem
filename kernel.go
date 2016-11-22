package main

type BoundaryType int

const (
	Essential BoundaryType = iota
	Dirichlet
)

type Boundary struct {
	Type BoundaryType
	Val  float64
}

func EssentialBC(u float64) *Boundary     { return &Boundary{Essential, u} }
func DirichletBC(gradU float64) *Boundary { return &Boundary{Dirichlet, gradU} }

const DefaultPenalty = 1e14

type KernelParams struct {
	X []float64
	// U is value of the solution shape function if solving a linear system
	// and it is the current guess for the solution (i.e. shape function val
	// times current solution guess).  For linear systems, an explicit solve
	// will actually determine the nodal U values directly.  For nonlinear
	// systems, an Newton or similar will be used to iterate toward better U
	// guesses.
	U     float64
	GradU float64
	// W holds the value of the weight/test function
	W float64
	// GradW holds the derivative of the weight/test function
	GradW float64
	// Penalty represents a penalty factor for converting essential boundary
	// conditions to natural/traction boundary conditions.
	Penalty float64
}

type Kernel interface {
	// Kernel returns the value of the volumetric integration portion of the weak form
	// (i.e. everything but the boundary/surface integration).
	VolIntU(p *KernelParams) float64
	VolInt(p *KernelParams) float64
	BoundaryIntU(p *KernelParams) float64
	BoundaryInt(p *KernelParams) float64
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
	// K is thermal conductivity between node points.  len(K) == len(X)-1.
	// K[i] is the thermal conductivity between X[i] and X[i+1].
	K Valer
	// S is heat source between node points.  len(S) == len(X)-1.
	// S[i] is the thermal conductivity between X[i] and X[i+1].
	S Valer
	// Area is the cross section area of the conduction medium
	Area  float64
	Left  *Boundary
	Right *Boundary
}

func (hc *HeatConduction) VolIntU(p *KernelParams) float64 {
	return p.GradW * hc.K.Val(p.X) * hc.Area * p.GradU
}
func (hc *HeatConduction) VolInt(p *KernelParams) float64 {
	return p.W * hc.S.Val(p.X)
}

func (hc *HeatConduction) BoundaryIntU(p *KernelParams) float64 {
	if p.X[0] != hc.X[0] && p.X[0] != hc.X[len(hc.X)-1] {
		return 0
	}

	if p.X[0] == hc.X[0] {
		if hc.Left.Type == Dirichlet {
			return 0
		}
		return p.W * hc.Area * p.Penalty * p.U
	} else {
		if hc.Right.Type == Dirichlet {
			return 0
		}
		return p.W * hc.Area * p.Penalty * p.U
	}
}

func (hc *HeatConduction) BoundaryInt(p *KernelParams) float64 {
	if p.X[0] != hc.X[0] && p.X[0] != hc.X[len(hc.X)-1] {
		return 0
	}

	if p.X[0] == hc.X[0] {
		if hc.Left.Type == Essential {
			return p.W * hc.Area * p.Penalty * hc.Left.Val
		}
		return p.W * hc.Area * hc.Left.Val
	} else {
		if hc.Right.Type == Essential {
			return p.W * hc.Area * p.Penalty * hc.Right.Val
		}
		return -1 * p.W * hc.Area * hc.Right.Val
	}
}
