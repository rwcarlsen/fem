package sparse

import "sort"

// ApplyPivot uses the given pivot row to multiply and add to all other rows
// in A either above or below the pivot (dir = -1 for below pivot and 1 for
// above pivot) in order to zero out the given column.  The appropriate
// operations are also performed on b to keep it in sync.  If L is not nil, the
// multipliers used for each row are stored in it.  len(multipliers) should be
func ApplyPivot(A Matrix, b []float64, col int, piv int, dir int, L Matrix) {
	pval := A.At(piv, col)
	bval := b[piv]
	for _, nonzero := range A.SweepCol(col) {
		i := nonzero.I
		aij := nonzero.Val
		cond := ((dir < 0) && i > piv) || ((dir > 0) && i < piv) || (dir == 0 && i != piv)
		if i != piv && cond {
			mult := -aij / pval
			if L != nil {
				L.Set(i, col, mult)
			}
			RowCombination(A, piv, i, mult)
			b[i] += bval * mult
		}
	}
	return mult
}

func vecAdd(result, a, b []float64) {
	if len(a) != len(b) {
		panic("inconsistent lengths for vector subtraction")
	}
	for i := range a {
		result[i] = a[i] + b[i]
	}
}

func vecSub(result, a, b []float64) {
	if len(a) != len(b) {
		panic("inconsistent lengths for vector subtraction")
	}
	for i := range a {
		result[i] = a[i] - b[i]
	}
}

// dot performs a vector*vector dot product.
func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("inconsistent lengths for dot product")
	}
	v := 0.0
	for i := range a {
		v += a[i] * b[i]
	}
	return v
}

func vecMult(v []float64, mult float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = mult * v[i]
	}
	return result
}

// RCM provides an alternate degree-of-freedom reordering in assembled matrix
// that provides better bandwidth properties for solvers.
func RCM(A Matrix) []int {
	size, _ := A.Dims()

	degreemap := make([]int, size)
	for i := range degreemap {
		degreemap[i] = i
	}

	sort.SliceStable(degreemap, func(i, j int) bool {
		return len(A.SweepRow(degreemap[i])) > len(A.SweepRow(degreemap[j]))
	})
	startrow := degreemap[0]

	// breadth-first search across adjacency/connections between nodes/dofs
	nextlevel := []int{startrow}
	mapping := make(map[int]int, size)
	mapping[startrow] = 0
	for n := 0; n < size; n++ {
		if len(nextlevel) == 0 {
			// Matrix does not represent a fully connected graph. We need to
			// choose a random dof/index that we haven't remapped yet to start
			// from
			for _, k := range degreemap {
				if _, ok := mapping[k]; !ok {
					nextlevel = []int{k}
					break
				}
			}
		}

		nextlevel = nextRCMLevel(A, mapping, nextlevel)
		if len(mapping) >= size {
			break
		}
	}

	slice := make([]int, size)

	reverse := make([]int, size)
	count := size - 1
	for i := range reverse {
		reverse[i] = count
		count--
	}

	for from, to := range mapping {
		slice[from] = reverse[to]
	}
	return slice
}

func nextRCMLevel(A Matrix, mapping map[int]int, ii []int) []int {
	var nextlevel []int
	size, _ := A.Dims()
	tmp := []int{}
	for _, i := range ii {
		tmp = tmp[:0]
		for _, nonzero := range A.SweepRow(i) {
			j := nonzero.J
			if _, ok := mapping[j]; !ok {
				tmp = append(tmp, j)
				if len(mapping)+len(tmp) >= size {
					break
				}
			}
		}

		// sort tmp and insert into mapping batched by src row
		sort.SliceStable(tmp, func(i, j int) bool {
			return len(A.SweepRow(tmp[i])) > len(A.SweepRow(tmp[j]))
		})
		for _, index := range tmp {
			mapping[index] = len(mapping)
			nextlevel = append(nextlevel, index)
		}
	}

	return nextlevel
}
