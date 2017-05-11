package main

import "github.com/gonum/matrix/mat64"

type GaussSeidel struct {
	MaxIter int
	Tol     float64
}

func (g *GaussSeidel) solveRow(i int, row, b, soln *mat64.Vector) {
	acceleration := 1.8 // between 1.0 and 2.0
	xold := soln.At(i, 0)
	soln.SetVec(i, 0)
	Ainverse := 1 / row.At(i, 0)
	xnew := (1-acceleration)*xold +
		acceleration*Ainverse*(b.At(i, 0)-mat64.Dot(row, soln))
	soln.SetVec(i, xnew)
}

func (g *GaussSeidel) forwardIter(Ai func(row int) *mat64.Vector, b, soln *mat64.Vector) {
	for i := 0; i < b.Len(); i++ {
		g.solveRow(i, Ai(i), b, soln)
	}
}

func (g *GaussSeidel) backwardIter(Ai func(row int) *mat64.Vector, b, soln *mat64.Vector) {
	for i := b.Len() - 1; i >= 0; i-- {
		g.solveRow(i, Ai(i), b, soln)
	}
}

func (g *GaussSeidel) Solve(Ai func(row int) *mat64.Vector, b *mat64.Vector) (soln *mat64.Vector, niter int) {
	prev := mat64.NewVector(b.Len(), nil)
	soln = mat64.NewVector(b.Len(), nil)
	soln.CloneVec(b)

	n := 0
	for ; n < g.MaxIter; n++ {
		g.forwardIter(Ai, b, soln)
		g.backwardIter(Ai, b, soln)

		// check convergence
		var diff mat64.Vector
		diff.SubVec(soln, prev)
		if er := mat64.Norm(&diff, 2) / mat64.Norm(soln, 2); er < g.Tol {
			break
		}
		prev.CloneVec(soln)
	}
	return soln, n
}
