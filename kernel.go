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
	X float64
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

// K
type Kernel interface {
	// Kernel returns the value of the volumetric integration portion of the weak form
	// (i.e. everything but the boundary/surface integration).
	VolIntU(p *KernelParams) float64
	VolInt(p *KernelParams) float64
	BoundaryIntU(p *KernelParams) float64
	BoundaryInt(p *KernelParams) float64
}

type HeatConduction struct {
	// X is the node/mesh points.
	X []float64
	// K is thermal conductivity between node points.  len(K) == len(X)-1.
	// K[i] is the thermal conductivity between X[i] and X[i+1].
	K []float64
	// S is heat source between node points.  len(S) == len(X)-1.
	// S[i] is the thermal conductivity between X[i] and X[i+1].
	S []float64
	// Area is the cross section area of the conduction medium
	Area  float64
	Left  *Boundary
	Right *Boundary
}

func (hc *HeatConduction) VolIntU(p *KernelParams) float64 {
	for i := 0; i < len(hc.X)-1; i++ {
		if hc.X[i] <= p.X && p.X <= hc.X[i+1] {
			return p.GradW * hc.K[i] * hc.Area * p.GradU
		}
	}
	return 1e100 // insulating boundary
}
func (hc *HeatConduction) VolInt(p *KernelParams) float64 {
	for i := 0; i < len(hc.X)-1; i++ {
		if hc.X[i] <= p.X && p.X <= hc.X[i+1] {
			return p.W * hc.S[i]
		}
	}
	return 0
}

func (hc *HeatConduction) BoundaryIntU(p *KernelParams) float64 {
	if p.X != hc.X[0] && p.X != hc.X[len(hc.X)-1] {
		return 0
	}

	if p.X == hc.X[0] {
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
	if p.X != hc.X[0] && p.X != hc.X[len(hc.X)-1] {
		return 0
	}

	if p.X == hc.X[0] {
		if hc.Left.Type == Essential {
			return p.W * hc.Area * p.Penalty * hc.Left.Val
		}
		return p.W * hc.Area * hc.Left.Val
	} else {
		if hc.Right.Type == Essential {
			return p.W * hc.Area * p.Penalty * hc.Right.Val
		}
		return p.W * hc.Area * hc.Right.Val
	}
}

type SpringKernel struct {
	X []float64
	K []float64
}

func (k *SpringKernel) VolIntU(p *KernelParams) float64 {
	for i := 1; i < len(k.X); i++ {
		if k.X[i-1] <= p.X && p.X <= k.X[i] {
			return p.GradW * p.GradU * k.K[i-1]
		}
	}
	return 1e100
}
func (k *SpringKernel) VolInt(p *KernelParams) float64       { return 0 }
func (k *SpringKernel) BoundaryIntU(p *KernelParams) float64 { return 0 }
func (k *SpringKernel) BoundaryInt(p *KernelParams) float64  { return 0 }
