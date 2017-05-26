package main

type BoundaryType int

const (
	Dirichlet BoundaryType = iota
	Neumann
	Interior
)

type Boundary interface {
	Type(x []float64) BoundaryType
	Val(x []float64) float64
}

type Boundary1D struct {
	Left      float64
	LeftVal   float64
	LeftType  BoundaryType
	Right     float64
	RightVal  float64
	RightType BoundaryType
}

func NewBoundary1D(xs []float64, lval, rval float64, ltype, rtype BoundaryType) *Boundary1D {
	return &Boundary1D{
		Left:      xs[0],
		Right:     xs[len(xs)-1],
		LeftVal:   lval,
		RightVal:  rval,
		LeftType:  ltype,
		RightType: rtype,
	}
}

func (b *Boundary1D) Type(x []float64) BoundaryType {
	if x[0] == b.Left {
		return b.LeftType
	} else if x[0] == b.Right {
		return b.RightType
	}
	return Interior
}

func (b *Boundary1D) Val(x []float64) float64 {
	if x[0] == b.Left {
		return b.LeftVal
	} else if x[0] == b.Right {
		if b.RightType == Neumann {
			// negative on rightval is for polarity of boundary integral
			return -b.RightVal
		}
		return b.RightVal
	}
	return 0
}

// Boundary2D represents a 2D boundary defined by nodes connected with
// straight lines.
type Boundary2D struct {
	// X holds the X coordinate sequence of the problem boundary.  The last
	// coordinate connects to the first.
	X []float64
	// Y holds the Y coordinate sequence of the problem boundary. The last
	// coordinate connects to the first.
	Y []float64
	// Types holds the sequence of the problem boundary condition types.
	// Type[i] is the BC type for the path between points (X[i],Y[i]) and
	// (X[i+1],Y[i+1]) where the last X,Y point wraps/connects to the first.
	Types []BoundaryType
	// Val holds boundary condition values. Val[i] is the BC for the path
	// between points (X[i],Y[i]) and (X[i+1],Y[i+1]) where the last X,Y point
	// wraps/connects to the first.
	Vals []float64
	// Tol is the distance within which a point is considered on the boundary.
	Tol float64
}

func (b *Boundary2D) Append(x, y float64, t BoundaryType, val float64) {
	b.X = append(b.X, x)
	b.Y = append(b.Y, y)
	b.Types = append(b.Types, t)
	b.Vals = append(b.Vals, val)
}

// loc returns the index (into Type/Val) of the point x on the boundary.
// It returns -1 if x is not on the boundary.
func (b *Boundary2D) loc(x []float64) int {
	xx, yy := x[0], x[1]
	for i := range b.X {
		x1, x2 := b.X[i], b.X[0] // wrap last node to 1st node
		y1, y2 := b.Y[i], b.Y[0] // wrap last node to 1st node
		if i+1 < len(b.X) {
			x2 = b.X[i+1]
			y2 = b.Y[i+1]
		}

		if x2 < x1 {
			x1, x2 = x2, x1
		}
		if y2 < y1 {
			y1, y2 = y2, y1
		}

		if xx+b.Tol < x1 || x2 < xx-b.Tol || yy+b.Tol < y1 || y2 < yy-b.Tol {
			continue
		}

		proj := vecProject(x, []float64{x1, y1}, []float64{x2, y2})
		norm := vecL2Norm(vecSub(proj, x))
		if norm < b.Tol {
			return i
		}
	}
	return -1
}

func (b *Boundary2D) Type(x []float64) BoundaryType {
	i := b.loc(x)
	if i == -1 {
		return Interior
	}
	return b.Types[i]
}

func (b *Boundary2D) Val(x []float64) float64 {
	i := b.loc(x)
	if i == -1 {
		return 0.0
	}
	return b.Vals[i]
}

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
	// K is thermal conductivity (W/m/K).
	K Valer
	// S is heat source strength (W/m^3).
	S Valer
	// Boundary holds the non-dirichlet boundary conditions for the problem.
	Boundary Boundary
}

func (hc *HeatConduction) IsDirichlet(x []float64) (bool, float64) {
	return hc.Boundary.Type(x) == Dirichlet, hc.Boundary.Val(x)
}

func (hc *HeatConduction) VolIntU(p *KernelParams) float64 {
	return Dot(p.GradW, p.GradU) * hc.K.Val(p.X)
}

func (hc *HeatConduction) VolInt(p *KernelParams) float64 {
	return p.W * hc.S.Val(p.X)
}

func (hc *HeatConduction) BoundaryIntU(p *KernelParams) float64 { return 0 }

func (hc *HeatConduction) BoundaryInt(p *KernelParams) float64 {
	return p.W * hc.Boundary.Val(p.X)
}
