package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type rowmult struct {
	row   int
	pivot int
}

func scaleRow(Ab *mat64.Dense, pivot int) {
	data := Ab.RawRowView(pivot)
	mult := 1 / data[pivot]
	for i := range data {
		data[i] *= mult
	}
}

func rowEliminate(Ab *mat64.Dense, pivot int, pivotdata []float64, row int) {
	data := Ab.RawRowView(row)
	mult := data[pivot]
	data[pivot] = 0
	for j := pivot + 1; j < len(data); j++ {
		data[j] -= mult * pivotdata[j]
	}
}

func swapRowNonzero(Ab *mat64.Dense, pivot int, idxs []int, tol float64) {
	r, _ := Ab.Dims()
	for i := pivot; i < r; i++ {
		if Ab.At(i, pivot) > tol {
			idxs[pivot], idxs[i] = idxs[i], idxs[pivot]
			pivotrow := Ab.RawRowView(pivot)
			swaprow := make([]float64, len(pivotrow))
			copy(swaprow, Ab.RawRowView(i))
			Ab.SetRow(i, pivotrow)
			Ab.SetRow(pivot, swaprow)
		}
	}
	panic("singular system")
}

func ParallelSolve(A, b *mat64.Dense) *mat64.Dense {
	var Ab mat64.Dense
	Ab.Augment(A, b)
	//fmt.Printf("    %v\n", mat64.Formatted(&Ab, mat64.Prefix("    ")))

	r, c := A.Dims()
	idxs := make([]int, r)
	for i := range idxs {
		idxs[i] = i
	}
	naug := c + 1

	for i := 0; i < r; i++ {
		if Ab.At(i, i) == 0 {
			swapRowNonzero(&Ab, i, idxs, 1e-10)
		}
		scaleRow(&Ab, i)
		pivotdata := Ab.RawRowView(i)
		for j := 0; j < c; j++ {
			if i == j {
				continue
			}

			//rowEliminate(&Ab, i, pivotdata, j)
			//pivot, row := i, j
			data := Ab.RawRowView(j)
			mult := data[i]
			data[i] = 0
			for k := i + 1; k < naug; k++ {
				data[k] -= mult * pivotdata[k]
			}
			//fmt.Printf("pivot=%v, row=%v\n", i, j)
			//fmt.Printf("    %.5v\n", mat64.Formatted(&Ab, mat64.Prefix("    ")))
		}
	}

	// reswap to original order
	x := Ab.ColView(c)
	for i, idx := range idxs {
		a, b := x.At(i, 0), x.At(idx, 0)
		x.SetVec(i, b)
		x.SetVec(idx, a)
	}

	return mat64.DenseCopyOf(x)
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
	e, n := pt.Transform(x, y)
	A := mat64.NewDense(4, 4, nil)

	a1 := pt.srcToDst.At(0, 0)
	a2 := pt.srcToDst.At(0, 1)
	b1 := pt.srcToDst.At(1, 0)
	b2 := pt.srcToDst.At(1, 1)
	c1 := pt.srcToDst.At(2, 0)
	c2 := pt.srcToDst.At(2, 1)
	c3 := pt.srcToDst.At(2, 2)

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
