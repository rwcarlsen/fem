package main

import "math"

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
	return vecL2Norm(vecSub(a, b)) <= tol
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
// end1 and end2.  All three vectors and the return vector have the same
// length/dimension.
func vecProject(p []float64, end1, end2 []float64) []float64 {
	if len(p) != len(end1) || len(end1) != len(end2) {
		panic("inconsistent lengths for vector projection")
	}

	s := vecSub(end2, end1)
	v := vecSub(p, end1)
	vDotS := Dot(v, s)
	sDotS := Dot(s, s)

	proj := make([]float64, len(p))
	for i := range s {
		proj[i] = s[i] * vDotS / sDotS
	}
	return proj
}

func vecL2Norm(vec []float64) float64 {
	tot := 0.0
	for _, v := range vec {
		tot += v * v
	}
	return math.Sqrt(tot)
}

func vecMult(v []float64, mult float64) []float64 {
	for i := range v {
		v[i] *= mult
	}
	return v
}

func vecSub(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("inconsistent lengths for vector subtraction")
	}
	diff := make([]float64, len(a))
	for i := range a {
		diff[i] = a[i] - b[i]
	}
	return diff
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
