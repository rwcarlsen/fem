package main

import (
	"math"

	"github.com/gonum/integrate/quad"
	"github.com/gonum/matrix/mat64"
)

func QuadLegendre(ndim int, f func([]float64) float64, min, max float64, n int, xs, weights []float64) float64 {
	if n <= 0 {
		panic("quad: non-positive number of locations")
	} else if min > max {
		panic("quad: min > max")
	} else if min == max {
		return 0
	} else if ndim == 0 {
		return f(nil)
	}

	if len(xs) != n {
		xs = make([]float64, n)
	}
	if len(weights) != n {
		weights = make([]float64, n)
	}

	rule := quad.Legendre{}
	rule.FixedLocations(xs, weights, min, max)

	dims := make([]int, ndim)
	for i := range dims {
		dims[i] = n
	}
	perms := Permute(nil, dims...)

	fullxs := make([]float64, ndim)
	var integral float64
	for _, perm := range perms {
		w := 1.0
		for d, i := range perm {
			fullxs[d] = xs[i]
			w *= weights[i]
		}
		integral += w * f(fullxs)
	}

	return integral
}

func det(m *mat64.Dense) float64 {
	ndim, _ := m.Dims()
	if ndim == 2 {
		a := m.At(0, 0)
		b := m.At(0, 1)
		c := m.At(1, 0)
		d := m.At(1, 1)
		return (a*d - b*c)
	}
	return mat64.Det(m)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func minInt(vals ...int) int {
	minv := vals[0]
	for _, v := range vals[1:] {
		if v < minv {
			minv = v
		}
	}
	return minv
}

func min(vals ...float64) float64 {
	v := vals[0]
	for _, val := range vals[1:] {
		if val < v {
			v = val
		}
	}
	return v
}

func max(vals ...float64) float64 {
	v := vals[0]
	for _, val := range vals[1:] {
		if val > v {
			v = val
		}
	}
	return v
}

func PosEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	return vecL2Norm(vecSub(nil, a, b)) <= tol
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

// vecProject projects multidimensional point p onto the line connecting points
// end1 and end2 and returns the projected point p.  All three vectors and the return vector have the same
// length/dimension.
func vecProject(p []float64, end1, end2 []float64) []float64 {
	if len(p) != len(end1) || len(end1) != len(end2) {
		panic("inconsistent lengths for vector projection")
	}

	s := vecSub(nil, end2, end1)
	v := vecSub(nil, p, end1)
	vDotS := Dot(v, s)
	sDotS := Dot(s, s)

	proj := make([]float64, len(p))
	for i := range s {
		proj[i] = s[i] * vDotS / sDotS
	}
	return vecSub(nil, proj, vecMult(end1, -1)) // add back end1
}

func vecL2Norm(vec []float64) float64 {
	return math.Sqrt(Dot(vec, vec))
}

func vecMult(v []float64, mult float64) []float64 {
	for i := range v {
		v[i] *= mult
	}
	return v
}

func vecSub(dst, a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("inconsistent lengths for vector subtraction")
	}
	if dst == nil {
		dst = make([]float64, len(a))
	}
	for i := range a {
		dst[i] = a[i] - b[i]
	}
	return dst
}

func Permute(skip func([]int) bool, dimensions ...int) [][]int {
	return permute(skip, dimensions, []int{})
}

func permute(skip func([]int) bool, dimensions []int, prefix []int) [][]int {
	set := make([][]int, 0)

	if len(dimensions) == 1 {
		for i := 0; i < dimensions[0]; i++ {
			val := append(append([]int{}, prefix...), i)
			set = append(set, val)
		}
		return set
	}

	max := dimensions[0]
	for i := 0; i < max; i++ {
		newprefix := append(prefix, i)
		if skip != nil && skip(newprefix) {
			continue
		}
		set = append(set, permute(skip, dimensions[1:], newprefix)...)
	}
	return set
}

func pow(a, b int) int {
	if a == 1 || b == 0 {
		return 1
	}

	v := 1
	for i := 0; i < b; i++ {
		v *= a
	}
	return v
}
