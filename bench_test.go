package main

import (
	"math/rand"
	"testing"
)

func BenchmarkMeshBuild(b *testing.B) {
	b.Run("nodes=10", benchMeshBuildN(10))
	b.Run("nodes=100", benchMeshBuildN(100))
	b.Run("nodes=1000", benchMeshBuildN(1000))
}

func BenchmarkSolve(b *testing.B) {
	b.Run("nodes=10", benchSolveN(10))
	b.Run("nodes=100", benchSolveN(100))
	b.Run("nodes=1000", benchSolveN(1000))
	b.Run("nodes=10000", benchSolveN(10000))
}

func BenchmarkInterpolate(b *testing.B) {
	b.Run("nodes=10", benchInterpolateN(10))
	b.Run("nodes=100", benchInterpolateN(100))
	b.Run("nodes=1000", benchInterpolateN(1000))
}

func benchMeshBuildN(n int) func(b *testing.B) {
	return func(b *testing.B) {
		degree := 2
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = float64(i) / float64(n)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := NewMeshSimple1D(xs, degree)
			if err != nil {
				b.Error(err)
			}
		}
	}
}

func benchSolveN(n int) func(b *testing.B) {
	return func(b *testing.B) {
		degree := 2
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = float64(i) / float64(n)
		}
		mesh, err := NewMeshSimple1D(xs, degree)
		if err != nil {
			b.Error(err)
		}

		hc := &HeatConduction{
			X: xs,
			K: ConstVal(2),  // W/(m*C)
			S: ConstVal(50), // W/m^3
			Boundary: &Boundary1D{
				Left:      xs[0],
				LeftVal:   0, // deg C
				LeftType:  Dirichlet,
				Right:     xs[len(xs)-1],
				RightVal:  5, // W/m^2
				RightType: Neumann,
			},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			mesh.Solve(hc)
		}
	}
}

func benchInterpolateN(n int) func(b *testing.B) {
	return func(b *testing.B) {
		degree := 2
		xs := make([]float64, n+1)
		for i := 0; i < n+1; i++ {
			xs[i] = float64(i) / float64(n)
		}
		mesh, err := NewMeshSimple1D(xs, degree)
		if err != nil {
			b.Error(err)
		}

		hc := &HeatConduction{
			X: xs,
			K: ConstVal(2),  // W/(m*C)
			S: ConstVal(50), // W/m^3
			Boundary: &Boundary1D{
				Left:      xs[0],
				LeftVal:   0, // deg C
				LeftType:  Dirichlet,
				Right:     xs[len(xs)-1],
				RightVal:  5, // W/m^2
				RightType: Neumann,
			},
		}
		mesh.Solve(hc)

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			mesh.Interpolate([]float64{rand.Float64()})
		}
	}
}
