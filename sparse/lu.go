package sparse

func IncompleteLU(A Matrix) PreconditionerFunc {
	size, _ := A.Dims()
	lu := &LU{
		L: RestrictByPattern{Matrix: NewSparse(size), Pattern: A},
		U: RestrictByPattern{Matrix: NewSparse(size), Pattern: A},
	}
	lu.Factorize(A)
	return func(z, r []float64) {
		lu.Solve(r, z)
	}
}

// LU performs holds an LU factorization for a matrix A used for solving Ax=b.
type LU struct {
	L, U Matrix
}

func (lu *LU) Factorize(A Matrix) *LU {
	size, _ := A.Dims()
	if lu.L == nil {
		lu.L = NewSparse(size)
	}
	if lu.U == nil {
		lu.U = NewSparse(size)
	}

	Copy(lu.U, A)

	for j := 0; j < size; j++ {
		lu.L.Set(j, j, 1)
		piv := j
		ApplyPivot(lu.U, nil, j, piv, -1, lu.L)
	}

	return lu
}

func (lu *LU) Solve(b, result []float64) ([]float64, error) {
	if result == nil {
		result = make([]float64, len(b))
	}

	// Solve Ly = b via forward substitution
	y := make([]float64, len(b))
	for i := 0; i < len(b); i++ {
		tot := 0.0
		div := 0.0
		for _, nonzero := range lu.L.SweepRow(i) {
			if nonzero.I == nonzero.J {
				div = nonzero.Val
			} else {
				tot += y[nonzero.J] * nonzero.Val
			}
		}
		y[i] = (b[i] - tot) / div
	}

	// Solve Ux = y via backward substitution
	for i := len(b) - 1; i >= 0; i-- {
		tot := 0.0
		div := 0.0
		for _, nonzero := range lu.U.SweepRow(i) {
			if nonzero.I == nonzero.J {
				div = nonzero.Val
			} else {
				tot += result[nonzero.J] * nonzero.Val
			}
		}
		result[i] = (y[i] - tot) / div
	}
	return result, nil
}
