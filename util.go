package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

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

type ProjectiveTransform struct {
	srcToDst *mat64.Dense
	dstToSrc *mat64.Dense
}

// NewProjectiveTransform computes the transofmration matrix that maps the
// quadrilateral specified by src coordinates to the quadrilateral specified
// by dst coordinates.  All slice lengths must be 4, sub-slices hold pairs of
// xy coordinates.  To map from src coordinates to dst coordinates, multiply
// the returned transformation  matrix against a length 2 column vector
// holding the x,y coordinates.
func NewProjectiveTransform(src, dst [][]float64) (*ProjectiveTransform, error) {
	A := projectivePart(src)
	B := projectivePart(dst)

	var Ainv mat64.Dense
	var Binv mat64.Dense
	var forward mat64.Dense
	var backward mat64.Dense

	err := Ainv.Inverse(A)
	if err != nil {
		return nil, err
	}
	err = Binv.Inverse(B)
	if err != nil {
		return nil, err
	}

	forward.Mul(B, &Ainv)
	backward.Mul(A, &Binv)
	return &ProjectiveTransform{srcToDst: &forward, dstToSrc: &backward}, nil
}

func projectivePart(pts [][]float64) *mat64.Dense {
	A := mat64.NewDense(3, 3, nil)
	b := mat64.NewDense(3, 1, []float64{pts[3][0], pts[3][1], 1})
	for c, val := range pts[:3] {
		x, y := val[0], val[1]
		A.Set(0, c, x)
		A.Set(1, c, y)
		A.Set(2, c, 1)
	}

	var u mat64.Dense
	err := u.Solve(A, b)
	if err != nil {
		panic(err)
	}
	for r := range pts[:3] {
		A.Set(r, 0, A.At(r, 0)*u.At(0, 0))
		A.Set(r, 1, A.At(r, 1)*u.At(1, 0))
		A.Set(r, 2, A.At(r, 2)*u.At(2, 0))
	}
	return A
}

func (at *ProjectiveTransform) Transform(x, y float64) (float64, float64) {
	u := mat64.NewDense(3, 1, []float64{x, y, 1})
	u.Mul(at.srcToDst, u)
	return u.At(0, 0) / u.At(2, 0), u.At(1, 0) / u.At(2, 0)
}

func (at *ProjectiveTransform) Reverse(xp, yp float64) (float64, float64) {
	u := mat64.NewDense(3, 1, []float64{xp, yp, 1})
	u.Mul(at.dstToSrc, u)
	return u.At(0, 0) / u.At(2, 0), u.At(1, 0) / u.At(2, 0)
}

func (pt *ProjectiveTransform) JacobianDet(x, y float64) float64 {
	return mat64.Det(pt.Jacobian(x, y))
}

func (pt *ProjectiveTransform) Jacobian(x, y float64) *mat64.Dense {
	return pt.jacobian(pt.srcToDst, x, y)
}

func (pt *ProjectiveTransform) ReverseJacobianDet(x, y float64) float64 {
	return mat64.Det(pt.ReverseJacobian(x, y))
}

func (pt *ProjectiveTransform) ReverseJacobian(x, y float64) *mat64.Dense {
	return pt.jacobian(pt.dstToSrc, x, y)
}

func (pt *ProjectiveTransform) jacobian(trans *mat64.Dense, x, y float64) *mat64.Dense {
	e, n := pt.Transform(x, y)
	A := mat64.NewDense(4, 4, nil)

	a1 := trans.At(0, 0)
	a2 := trans.At(0, 1)
	b1 := trans.At(1, 0)
	b2 := trans.At(1, 1)
	c1 := trans.At(2, 0)
	c2 := trans.At(2, 1)
	c3 := trans.At(2, 2)

	A.Set(0, 0, a1-c1*e)
	A.Set(0, 1, a2)
	A.Set(1, 2, a1-c1*e)
	A.Set(1, 3, a2-c2*e)

	A.Set(2, 2, b1-c1*n)
	A.Set(2, 3, b2)
	A.Set(3, 0, b1-c1*n)
	A.Set(3, 1, b2-c2*n)

	v := c1*x + c2*y + c3
	b := mat64.NewVector(4, []float64{v, 0, v, 0})

	var partials mat64.Vector

	partials.SolveVec(A, b)

	jacobian := mat64.NewDense(2, 2, nil)
	jacobian.Set(0, 0, partials.At(0, 0))
	jacobian.Set(0, 1, partials.At(1, 0))
	jacobian.Set(1, 0, partials.At(2, 0))
	jacobian.Set(1, 1, partials.At(3, 0))

	return jacobian
}
