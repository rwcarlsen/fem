package sparse

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
func RCM(A *Matrix) []int {
	size, _ := A.Dims()
	mapping := make(map[int]int, size)

	degreemap := make([]int, size)
	for i := range degreemap {
		degreemap[i] = i
	}

	sort.SliceStable(degreemap, func(i, j int) bool {
		return len(A.NonzeroCols(degreemap[i])) < len(A.NonzeroCols(degreemap[j]))
	})
	startrow := degreemap[0]

	// breadth-first search across adjacency/connections between nodes/dofs
	nextlevel := []int{startrow}
	for n := 0; n < size; n++ {
		if len(nextlevel) == 0 {
			// Matrix must not represent a fully connected graph. We need to choose a random dof/index
			// that we haven't remapped yet to start from
			for _, k := range degreemap {
				if _, ok := mapping[k]; !ok {
					nextlevel = []int{k}
					break
				}
			}
		}

		for _, i := range nextlevel {
			if _, ok := mapping[i]; !ok {
				mapping[i] = len(mapping)
			}
		}
		if len(mapping) >= size {
			break
		}
		nextlevel = nextRCMLevel(A, mapping, nextlevel)
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

func nextRCMLevel(A *Matrix, mapping map[int]int, ii []int) []int {
	var nextlevel []int
	for _, i := range ii {
		for j := range A.NonzeroCols(i) {
			if _, ok := mapping[j]; !ok {
				nextlevel = append(nextlevel, j)
			}
		}
	}
	return nextlevel
}
